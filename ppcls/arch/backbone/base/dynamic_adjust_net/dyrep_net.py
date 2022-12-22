import paddle
import paddle.nn as nn

from ..rep_blocks import DiverseBranchBlock
from .....utils import logger
from .....utils.misc import AverageMeter
from .base import DynamicAdjustNet


class DyRepNet(DynamicAdjustNet):
    def __init__(self,
                 net,
                 dbb_branches=[1, 1, 1, 1, 1, 1, 1],
                 grow_metric="grad_norm",
                 adjust_interval=5,
                 max_adjust_epochs=100):
        super().__init__()
        self.net = net

        if grow_metric not in ('grad_norm', 'snip', 'synflow', 'random'):
            msg = "DynamicRep only support "
            raise Exception(msg)
        self.grow_metric = grow_metric
        self.dbb_branches = dbb_branches

        # valid dbb branches for conv with unequal shapes of input and output
        self.dbb_branches_unequal = [
            v if i not in (0, 4, 5, 6) else 0
            for i, v in enumerate(dbb_branches)
        ]

        self.adjust_interval = adjust_interval
        self.max_adjust_epochs = max_adjust_epochs

        # dict for recording the metric of each conv modules
        self._metric_records = {}
        self._weight_records = {}
        self.new_param_group = None
        self.last_growed_module = 'none'

    def post_iter(self):
        self._record_metrics()
        # TODO(gaotingquan): debug
        # logger.warning(f"records: {self._metric_records}")

    def post_epoch(self, epoch_id, engine):
        if epoch_id < self.max_adjust_epochs:
            if epoch_id % self.adjust_interval == 0:
                # adjust
                logger.info("DyRep: adjust model.")
                self._adjust_net()
                logger.info("DyRep: adjust done.")
                self._reset_optimizer(engine)

                # for distributed
                if engine.config["Global"]["distributed"]:
                    engine.model = paddle.DataParallel(engine.model)

    def _get_module(self, path):
        path_split = path.split('.')
        m = self.net
        for key in path_split:
            if not hasattr(m, key):
                return None
            m = getattr(m, key)
        return m

    def _record_metrics(self):
        for k, m in self.net.named_sublayers():
            if not isinstance(m, nn.Conv2D) \
                    or m._kernel_size[0] != m._kernel_size[1] \
                    or m._kernel_size[0] == 1 \
                    or k.count('dbb') >= 2:
                # Requirements for growing the module:
                # 1. the module is a nn.Conv2D module;
                # 2. it must has the same kernel_size (>1) in `h` and `w` axes;
                # 3. we restrict the number of growths in each layer.
                continue

            if m.weight.grad is None:
                continue
            grad = m.weight.grad.reshape([-1])
            weight = m.weight.reshape([-1])

            if self.grow_metric == 'grad_norm':
                metric_val = grad.norm().item()
            elif self.grow_metric == 'snip':
                metric_val = (grad * weight).abs().sum().item()
            elif self.grow_metric == 'synflow':
                metric_val = (grad * weight).sum().item()
            elif self.grow_metric == 'random':
                metric_val = random.random()

            if k not in self._metric_records:
                # TODO: AverageMeter not support dist
                # self._metric_records[k] = AverageMeter(dist=True)
                self._metric_records[k] = AverageMeter()
            self._metric_records[k].update(metric_val)

    def _grow(self, metric_records_sorted, topk=1):
        if len(metric_records_sorted) == 0:
            return
        for i in range(topk):
            conv_to_grow = metric_records_sorted[i][0]
            logger.info('grow: {}'.format(conv_to_grow))
            len_parent_str = conv_to_grow.rfind('.')
            if len_parent_str != -1:
                parent = conv_to_grow[:len_parent_str]
                conv_key = conv_to_grow[len_parent_str + 1:]
                # get the target conv module and its parent
                parent_m = self._get_module(parent)
            else:
                conv_key = conv_to_grow
                parent_m = self.net
            conv_m = getattr(parent_m, conv_key, None)
            # replace target conv module with DBB
            conv_m_padding = conv_m._padding
            conv_m_kernel_size = conv_m._kernel_size[0]

            if conv_m_padding == conv_m_kernel_size // 2:
                dbb_branches = self.dbb_branches.copy()
            else:
                dbb_branches = self.dbb_branches_unequal.copy()
            dbb_block = DiverseBranchBlock(
                conv_m._in_channels,
                conv_m._out_channels,
                conv_m_kernel_size,
                stride=conv_m._stride,
                groups=conv_m._groups,
                padding=conv_m_padding,
                ori_conv=conv_m,
                branches=dbb_branches,
                use_bn=True,
                # bn=nn.BatchNorm2D,
                # TODO(gaotingquan): recal_bn_fn is not implemented
                # recal_bn_fn=self.recal_bn_fn
            )
            setattr(parent_m, conv_key, dbb_block)
            dbb_block._reset_dbb(conv_m.weight, conv_m.bias)
            self.last_growed_module = conv_to_grow
        logger.info(str(self.net))

    def _cut(self, dbb_key, cut_branches, remove_bn=False):
        dbb_m = self._get_module(dbb_key)
        if dbb_m is None:
            return
        if sum(cut_branches) == 1:
            # only keep the original 3x3 conv
            parent = self._get_module(dbb_key[:dbb_key.rfind('.')])
            weight, bias = dbb_m.get_actual_kernel()
            conv = nn.Conv2D(
                dbb_m.in_channels,
                dbb_m.out_channels,
                dbb_m.kernel_size,
                stride=dbb_m.stride,
                groups=dbb_m.groups,
                padding=dbb_m.padding,
                bias=True).cuda()
            conv.weight.data = weight
            conv.bias.data = bias
            setattr(parent, dbb_key[dbb_key.rfind('.') + 1:], conv)
        else:
            dbb_m.cut_branch(cut_branches)

    def _reset_optimizer(self):
        return
        ''' TODO(gaotingquan): optimizer should be reset the parameters of model
        param_groups = get_params(self.net, lr=0.1, weight_decay=1e-5, filter_bias_and_bn=self.filter_bias_and_bn, sort_params=True)

        # remove the states of removed paramters
        for optim in self.optimizer:
            assert len(param_groups) == len(optim.param_groups)
        for param_group, param_group_old in zip(param_groups, self.optimizer.param_groups):
            params, params_old = param_group['params'], param_group_old['params']
            params = set(params)
            for param_old in params_old:
                if param_old not in params:
                    if param_old in self.optimizer.state:
                        del self.optimizer.state[param_old]
            param_group_old['params'] = param_group['params']
        '''

    def _adjust_net(self):
        records = {}
        for key in self._metric_records:
            records[key] = self._metric_records[key].avg

        metric_records_sorted = sorted(
            records.items(), key=lambda item: item[1], reverse=True)

        logger.info('metric: {}'.format(metric_records_sorted))
        self._grow(metric_records_sorted)

        # reset records
        self._metric_records = {}

        for k, m in self.net.named_sublayers():
            if isinstance(m, DiverseBranchBlock):
                weights = m.branch_weights()
                logger.info(k + ': ' + str(weights))
                valid_weights = paddle.to_tensor([
                    x for x in weights[:3] + weights[4:] if x not in [-1, 1]
                ])
                if valid_weights.std() > 0.02:
                    mean = valid_weights.mean()
                    # cut those branches less than 0.1
                    need_cut = False
                    cut_branches = [1] * len(weights)
                    for idx in range(len(weights)):
                        if weights[idx] < mean and weights[idx] < 0.1:
                            cut_branches[idx] = 0
                            if weights[idx] != -1:
                                need_cut = True
                    if need_cut:
                        self._cut(k, cut_branches)
                        logger.info(f'cut: {k}, new branches: {cut_branches}')
