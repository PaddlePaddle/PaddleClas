# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Source-Code-License-NC
# PaddlePaddle port of NVLabs MambaVision.

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = [x.shape[0]] + [1] * (len(x.shape) - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)
    return (x / keep_prob) * random_tensor


class DropPath(nn.Layer):

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm2D(nn.Layer):

    def __init__(self, num_channels, epsilon=1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.weight = self.create_parameter(
            shape=[num_channels],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.bias = self.create_parameter(
            shape=[num_channels],
            default_initializer=nn.initializer.Constant(0.0),
        )

    def forward(self, x):
        mean = paddle.mean(x, axis=1, keepdim=True)
        var = paddle.mean((x - mean) * (x - mean), axis=1, keepdim=True)
        x = (x - mean) * paddle.rsqrt(var + self.epsilon)
        weight = self.weight.reshape([1, -1, 1, 1])
        bias = self.bias.reshape([1, -1, 1, 1])
        return x * weight + bias


class ChannelFirstLayerNorm(nn.Layer):

    def __init__(self, num_channels, epsilon=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, epsilon=epsilon)

    def forward(self, x):
        x = x.transpose([0, 2, 3, 1])
        x = self.norm(x)
        return x.transpose([0, 3, 1, 2])


class TorchCompatibleLayerNorm(nn.LayerNorm):
    """Use PyTorch's CUDA LayerNorm reduction order for strict eval audits."""

    def __init__(self,
                 normalized_shape,
                 epsilon=1e-5,
                 implementation="paddle"):
        super().__init__(normalized_shape, epsilon=epsilon)
        if implementation != "paddle":
            raise ValueError(
                f"Unsupported LayerNorm implementation: {implementation!r}")
        self.implementation = implementation
        self.torch_epsilon = float(epsilon)

    def forward(self, x):
        # Explicit dispatch is required by Paddle dy2static; zero-argument super()
        # cannot be transformed in this overridden LayerNorm path.
        return nn.LayerNorm.forward(self, x)


class Mlp(nn.Layer):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def window_partition(x, window_size):
    b, c, h, w = x.shape
    x = x.reshape(
        [b, c, h // window_size, window_size, w // window_size, window_size])
    windows = x.transpose([0, 2, 4, 3, 5,
                           1]).reshape([-1, window_size * window_size, c])
    return windows


def window_reverse(windows, window_size, height, width):
    b = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.reshape([
        b,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    ])
    x = x.transpose([0, 5, 1, 3, 2,
                     4]).reshape([b, windows.shape[2], height, width])
    return x


class Downsample(nn.Layer):

    def __init__(self, dim, keep_dim=False):
        super().__init__()
        dim_out = dim if keep_dim else 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2D(dim, dim_out, 3, stride=2, padding=1, bias_attr=False), )

    def forward(self, x):
        return self.reduction(x)


class PatchEmbed(nn.Layer):

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2D(in_chans,
                      in_dim,
                      3,
                      stride=2,
                      padding=1,
                      bias_attr=False),
            nn.BatchNorm2D(in_dim, epsilon=1e-4),
            nn.ReLU(),
            nn.Conv2D(in_dim, dim, 3, stride=2, padding=1, bias_attr=False),
            nn.BatchNorm2D(dim, epsilon=1e-4),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class ConvBlock(nn.Layer):

    def __init__(
        self,
        dim,
        drop_path=0.0,
        layer_scale=None,
        kernel_size=3,
        gelu_impl="paddle",
    ):
        super().__init__()
        if gelu_impl != "paddle":
            raise ValueError(f"Unsupported GELU implementation: {gelu_impl!r}")
        self.gelu_impl = gelu_impl
        self.conv1 = nn.Conv2D(dim,
                               dim,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=1)
        self.norm1 = nn.BatchNorm2D(dim, epsilon=1e-5)
        self.conv2 = nn.Conv2D(dim,
                               dim,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=1)
        self.norm2 = nn.BatchNorm2D(dim, epsilon=1e-5)
        use_layer_scale = layer_scale is not None and isinstance(
            layer_scale, (int, float))
        self.layer_scale = use_layer_scale
        if use_layer_scale:
            self.gamma = self.create_parameter(
                shape=[dim],
                default_initializer=nn.initializer.Constant(
                    float(layer_scale)),
            )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.gelu(x, approximate=True)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.reshape([1, -1, 1, 1])
        x = residual + self.drop_path(x)
        return x


def selective_scan_paddle(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
    implementation="sequential",
):
    if implementation not in {"sequential", "parallel"}:
        raise ValueError(
            f"Unsupported selective scan implementation: {implementation!r}")
    dtype_in = u.dtype
    u = u.astype("float32")
    delta = delta.astype("float32")
    if delta_bias is not None:
        delta = delta + delta_bias.astype("float32").reshape([1, -1, 1])
    if delta_softplus:
        delta = F.softplus(delta)

    batch = u.shape[0]
    dim = A.shape[0]
    dstate = A.shape[1]
    B = B.astype("float32")
    C = C.astype("float32")

    deltaA = paddle.exp(paddle.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = paddle.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    if implementation == "sequential":
        x = paddle.zeros([batch, dim, dstate], dtype=A.dtype)
        ys = []
        last_state = None
        for i in range(u.shape[2]):
            x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :]
            y = paddle.einsum("bdn,bn->bd", x, C[:, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            ys.append(y)
        y = paddle.stack(ys, axis=2)
    else:
        # Affine recurrences compose associatively:
        # (a2, b2) o (a1, b1) = (a2 * a1, b2 + a2 * b1).
        # A Hillis-Steele scan reduces L sequential GPU launch groups to log2(L)
        # vectorized groups. It changes floating-point reduction order, so it is
        # an opt-in performance backend rather than the strict-alignment default.
        scan_a = deltaA
        scan_b = deltaB_u
        offset = 1
        sequence_length = u.shape[2]
        while offset < sequence_length:
            prev_a = scan_a[:, :, :-offset, :]
            prev_b = scan_b[:, :, :-offset, :]
            tail_a = scan_a[:, :, offset:, :]
            tail_b = scan_b[:, :, offset:, :]
            scan_a = paddle.concat([scan_a[:, :, :offset, :], tail_a * prev_a],
                                   axis=2)
            scan_b = paddle.concat(
                [scan_b[:, :, :offset, :], tail_b + tail_a * prev_b], axis=2)
            offset *= 2
        last_state = scan_b[:, :, -1, :]
        y = paddle.einsum("bdln,bnl->bdl", scan_b, C)
    out = y if D is None else y + u * D.reshape([1, -1, 1])
    if z is not None:
        out = out * F.silu(z)
    out = out.astype(dtype_in)
    return out if not return_last_state else (out, last_state)


class MambaVisionMixer(nn.Layer):

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        scan_impl="sequential",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model /
                                 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.scan_impl = scan_impl

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias_attr=bias)
        self.x_proj = nn.Linear(
            self.d_inner // 2,
            self.dt_rank + self.d_state * 2,
            bias_attr=False,
        )
        self.dt_proj = nn.Linear(self.dt_rank,
                                 self.d_inner // 2,
                                 bias_attr=True)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            self.dt_proj.weight.set_value(
                paddle.full(self.dt_proj.weight.shape,
                            dt_init_std,
                            dtype=self.dt_proj.weight.dtype))
        elif dt_init == "random":
            self.dt_proj.weight.set_value(
                paddle.uniform(
                    self.dt_proj.weight.shape,
                    min=-dt_init_std,
                    max=dt_init_std,
                    dtype=self.dt_proj.weight.dtype,
                ))
        else:
            raise NotImplementedError(dt_init)

        dt = paddle.exp(
            paddle.rand([self.d_inner // 2], dtype="float32") *
            (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = paddle.clip(dt, min=dt_init_floor)
        inv_dt = dt + paddle.log(-paddle.expm1(-dt))
        self.dt_proj.bias.set_value(inv_dt)

        A = paddle.arange(1, self.d_state + 1, dtype="float32")
        A = paddle.tile(A.reshape([1, -1]), [self.d_inner // 2, 1])
        self.A_log = self.create_parameter(
            shape=A.shape,
            default_initializer=nn.initializer.Assign(paddle.log(A)),
        )
        self.D = self.create_parameter(
            shape=[self.d_inner // 2],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias_attr=bias)

        conv_bias_attr = False if conv_bias // 2 == 0 else True
        self.conv1d_x = nn.Conv1D(
            self.d_inner // 2,
            self.d_inner // 2,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=self.d_inner // 2,
            bias_attr=conv_bias_attr,
        )
        self.conv1d_z = nn.Conv1D(
            self.d_inner // 2,
            self.d_inner // 2,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=self.d_inner // 2,
            bias_attr=conv_bias_attr,
        )

    def forward(self, hidden_states):
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = xz.transpose([0, 2, 1])
        x, z = paddle.chunk(xz, 2, axis=1)
        A = -paddle.exp(self.A_log.astype("float32"))

        x = F.silu(self.conv1d_x(x))
        z = F.silu(self.conv1d_z(z))
        x_dbl = self.x_proj(
            x.transpose([0, 2, 1]).reshape([-1, self.d_inner // 2]))
        dt, B, C = paddle.split(x_dbl,
                                [self.dt_rank, self.d_state, self.d_state],
                                axis=-1)
        dt = self.dt_proj(dt).reshape([-1, seqlen,
                                       self.d_inner // 2]).transpose([0, 2, 1])
        B = B.reshape([-1, seqlen, self.d_state]).transpose([0, 2, 1])
        C = C.reshape([-1, seqlen, self.d_state]).transpose([0, 2, 1])
        y = selective_scan_paddle(
            x,
            dt,
            A,
            B,
            C,
            self.D.astype("float32"),
            z=None,
            delta_bias=self.dt_proj.bias.astype("float32"),
            delta_softplus=True,
            return_last_state=False,
            implementation=self.scan_impl,
        )
        y = paddle.concat([y, z], axis=1)
        y = y.transpose([0, 2, 1])
        out = self.out_proj(y)
        return out


class Attention(nn.Layer):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape([b, n, 3, self.num_heads, self.head_dim])
        qkv = qkv.transpose([2, 0, 3, 1, 4])
        q, k, v = paddle.unstack(qkv, axis=0)
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = paddle.matmul(attn, v)
        x = x.transpose([0, 2, 1, 3]).reshape([b, n, c])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):

    def __init__(
        self,
        dim,
        num_heads,
        counter,
        transformer_blocks,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Mlp_block=Mlp,
        layer_scale=None,
        scan_impl="sequential",
    ):
        super().__init__()
        norm_impl = "paddle"
        block_norm = (TorchCompatibleLayerNorm(dim, implementation=norm_impl)
                      if norm_layer is nn.LayerNorm else norm_layer(dim))
        self.norm1 = block_norm
        if counter in transformer_blocks:
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                norm_layer=norm_layer,
            )
        else:
            self.mixer = MambaVisionMixer(
                d_model=dim,
                d_state=8,
                d_conv=3,
                expand=1,
                scan_impl=scan_impl,
            )

        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = (TorchCompatibleLayerNorm(dim, implementation=norm_impl)
                      if norm_layer is nn.LayerNorm else norm_layer(dim))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        use_layer_scale = layer_scale is not None and isinstance(
            layer_scale, (int, float))
        if use_layer_scale:
            self.gamma_1 = self.create_parameter(
                shape=[dim],
                default_initializer=nn.initializer.Constant(
                    float(layer_scale)),
            )
            self.gamma_2 = self.create_parameter(
                shape=[dim],
                default_initializer=nn.initializer.Constant(
                    float(layer_scale)),
            )
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x):
        gamma_1 = self.gamma_1 if self.gamma_1 is not None else 1.0
        gamma_2 = self.gamma_2 if self.gamma_2 is not None else 1.0
        x = x + self.drop_path(gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(gamma_2 * self.mlp(self.norm2(x)))
        return x


class MambaVisionLayer(nn.Layer):

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        conv=False,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        layer_scale_conv=None,
        transformer_blocks=None,
        scan_impl="sequential",
    ):
        super().__init__()
        transformer_blocks = transformer_blocks or []
        self.conv = conv
        conv_gelu_impl = "paddle"
        if conv:
            self.blocks = nn.LayerList([
                ConvBlock(
                    dim=dim,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list) else drop_path,
                    layer_scale=layer_scale_conv,
                    gelu_impl=conv_gelu_impl,
                ) for i in range(depth)
            ])
            self.transformer_block = False
        else:
            self.blocks = nn.LayerList([
                Block(
                    dim=dim,
                    counter=i,
                    transformer_blocks=transformer_blocks,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list) else drop_path,
                    layer_scale=layer_scale,
                    scan_impl=scan_impl,
                ) for i in range(depth)
            ])
            self.transformer_block = True

        self.downsample = None if not downsample else Downsample(dim=dim)
        self.window_size = window_size

    def forward(self, x, return_stage_output=False):
        _, _, h, w = x.shape
        pad_r = 0
        pad_b = 0
        hp = h
        wp = w

        if self.transformer_block:
            pad_r = (self.window_size -
                     w % self.window_size) % self.window_size
            pad_b = (self.window_size -
                     h % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = F.pad(
                    x,
                    pad=[0, pad_r, 0, pad_b],
                    mode="constant",
                    value=0.0,
                    data_format="NCHW",
                )
                _, _, hp, wp = x.shape
            x = window_partition(x, self.window_size)

        for blk in self.blocks:
            x = blk(x)

        if self.transformer_block:
            x = window_reverse(x, self.window_size, hp, wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :h, :w]

        stage_output = x
        x = x if self.downsample is None else self.downsample(x)
        if return_stage_output:
            return x, stage_output
        return x
