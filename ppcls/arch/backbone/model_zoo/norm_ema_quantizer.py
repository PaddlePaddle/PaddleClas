import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as distributed
from einops import rearrange, repeat

from .modeling_finetune import zeros_, ones_, Identity

def l2norm(t):
    return F.normalize(t, p=2, axis=-1)

def ema_inplace(moving_avg, new, decay):
    x = moving_avg * decay
    x = x + new*(1-decay)
    moving_avg = paddle.create_parameter(shape=x.shape, 
                        dtype=str(x.numpy().dtype), 
                        default_initializer=paddle.nn.initializer.Assign(x))

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = paddle.randperm(num_samples)[:num]
    else:
        indices = paddle.randint(0, num_samples, [num,])

    return samples[indices]

def kmeans(samples, num_clusters, num_iters = 10, use_cosine_sim = False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') \
                    - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(axis = -1)

        buckets = dists.max(axis = -1).indices
        bins = paddle.bincount(buckets, minlength = num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype = dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d = dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = paddle.where(zero_mask[..., None], means, new_means)

    return means, bins


class EmbeddingEMA(nn.Layer):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5, kmeans_init=True, codebook_init_path=''):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps 
        if codebook_init_path == '':   
            if not kmeans_init:
                weight = paddle.randn([num_tokens, codebook_dim])
                weight = l2norm(weight)
            else:
                weight = paddle.zeros([num_tokens, codebook_dim])
            self.register_buffer('initted', paddle.to_tensor([not kmeans_init], dtype='float32'))
        else:
            print(f"load init codebook weight from {codebook_init_path}")
            codebook_ckpt_weight = paddle.load(codebook_init_path, map_location='cpu')
            weight = codebook_ckpt_weight.clone()
            self.register_buffer('initted', paddle.to_tensor([True]))
        
        self.weight = paddle.create_parameter(shape=weight.shape, 
                                        dtype=str(weight.numpy().dtype), 
                                        default_initializer=paddle.nn.initializer.Assign(weight))
        self.cluster_size = self.create_parameter(shape=[num_tokens], default_initializer=zeros_)
        self.add_parameter("cluster_size", self.cluster_size)
        self.embed_avg = paddle.create_parameter(shape=weight.shape, 
                                        dtype=str(weight.numpy().dtype), 
                                        default_initializer=paddle.nn.initializer.Assign(weight))
        self.update = True

    def init_embed_(self, data):
        if self.initted:
            return
        print("Performing Kemans init for codebook")
        embed, cluster_size = kmeans(data, self.num_tokens, 10, use_cosine_sim=True)
        self.weight = paddle.create_parameter(shape=embed.shape, 
                                    dtype=str(embed.numpy().dtype), 
                                    default_initializer=paddle.nn.initializer.Assign(embed))
        self.cluster_size = paddle.create_parameter(shape=cluster_size.shape, 
                                    dtype=str(cluster_size.numpy().dtype), 
                                    default_initializer=paddle.nn.initializer.Assign(cluster_size))
        self.initted = paddle.create_parameter(shape=[1], 
                                    dtype="bool", 
                                    default_initializer=paddle.nn.initializer.Assign(paddle.to_tensor([True])))                            
        
    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        x = self.cluster_size.multiply(self.decay)
        x = x.add(new_cluster_size*(1 - self.decay))
        self.cluster_size = paddle.create_parameter(shape=x.shape, 
                                        dtype=str(x.numpy().dtype), 
                                        default_initializer=paddle.nn.initializer.Assign(x))

    def embed_avg_ema_update(self, new_embed_avg): 
        x = self.cluster_size.multiply(self.decay)
        x = x.add(new_embed_avg*(1 - self.decay))
        self.embed_avg = paddle.create_parameter(shape=x.shape, 
                                    dtype=str(x.numpy().dtype), 
                                    default_initializer=paddle.nn.initializer.Assign(x))
    
    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
            )
        #normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        # embed_normalized = l2norm(self.embed_avg / smoothed_cluster_size.unsqueeze(1))
        self.weight = paddle.create_parameter(shape=embed_normalized.shape, 
                                    dtype=str(embed_normalized.numpy().dtype), 
                                    default_initializer=paddle.nn.initializer.Assign(embed_normalized))


def norm_ema_inplace(moving_avg, new, decay):
    x = moving_avg.multiply(paddle.to_tensor(decay))
    x = x.add(new*(1 - decay))
    x = l2norm(x)
    moving_avg = paddle.create_parameter(shape=x.shape, 
                                dtype=str(x.numpy().dtype), 
                                default_initializer=paddle.nn.initializer.Assign(x))
    


class NormEMAVectorQuantizer(nn.Layer):
    def __init__(self, n_embed, embedding_dim, beta, decay=0.99, eps=1e-5, 
                statistic_code_usage=True, kmeans_init=False, codebook_init_path=''):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.decay = decay
        
        # learnable = True if orthogonal_reg_weight > 0 else False
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps, kmeans_init, codebook_init_path)

        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            self.register_buffer('cluster_size', paddle.zeros([n_embed]))
        # if distributed.is_available() and distributed.is_initialized():
        #     print("ddp is enable, so use ddp_reduce to sync the statistic_code_usage for each gpu!")
        #     self.all_reduce_fn = distributed.all_reduce
        # else:
        #     self.all_reduce_fn = Identity
        # self.all_reduce_fn = paddle.distributed.all_reduce()
        
    def reset_cluster_size(self, device):
        if self.statistic_code_usage:
            self.register_buffer('cluster_size', paddle.zeros([self.num_tokens]))
            self.cluster_size = self.cluster_size.to(device)

    def _masked_fill(self, x, mask, value):
        y = paddle.full(x.shape, value, x.dtype)
        return paddle.where(mask, y, x)

    def forward(self, z):
         # reshape z -> (batch, height, width, channel) and flatten
        #z, 'b c h w -> b h w c'
        b, c, h, w = z.shape
        z = paddle.reshape(z, [b, h, w, c])
        # z = rearrange(z, 'b c h w -> b h w c')
        z = l2norm(z)
        z_flattened = z.reshape([-1, self.codebook_dim])
        
        self.embedding.init_embed_(z_flattened)

        d = z_flattened.pow(2).sum(axis=1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(axis=1) - 2 * \
            paddle.einsum('bd,nd->bn', z_flattened, self.embedding.weight) # 'n d -> d n'

        encoding_indices = paddle.argmin(d, axis=1)

        z_q = self.embedding(encoding_indices).reshape(z.shape)

        encodings = F.one_hot(encoding_indices, self.num_tokens).astype(z.dtype)

        if not self.training:
            with paddle.no_grad():
                cluster_size = encodings.sum(0)
                # self.all_reduce_fn(cluster_size)
                if paddle.distributed.get_world_size() > 1:
                    paddle.distributed.all_reduce(cluster_size)
                ema_inplace(self.cluster_size, cluster_size, self.decay)
        
        if self.training and self.embedding.update:
            # EMA cluster size

            bins = encodings.sum(0)
            # self.all_reduce_fn(bins)
            if paddle.distributed.get_world_size() > 1:
                paddle.distributed.all_reduce(bins)

            # self.embedding.cluster_size_ema_update(bins)
            ema_inplace(self.cluster_size, bins, self.decay)

            zero_mask = (bins == 0)
            # bins = bins.masked_fill(zero_mask, 1.)
            bins = self._masked_fill(bins, zero_mask, 1.)

            embed_sum = z_flattened.t() @ encodings
            # self.all_reduce_fn(embed_sum)
            if paddle.distributed.get_world_size() > 1:
                paddle.distributed.all_reduce(embed_sum)
                        
            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = l2norm(embed_normalized)
            
            embed_normalized = paddle.where(zero_mask[..., None], self.embedding.weight,
                                           embed_normalized)
            norm_ema_inplace(self.embedding.weight, embed_normalized, self.decay)

        # compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z) 
        
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        #z_q, 'b h w c -> b c h w'
        b, h, w, c = z_q.shape
        z_q = paddle.reshape(z_q, [b, c, h, w])
        # z_q = rearrange(z_q, 'b h w c -> b c h w')
        return z_q, loss, encoding_indices
