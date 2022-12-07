"""
backbone option "WideResNet"

"""

from audioop import bias
from turtle import forward
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from sympy import false


class PSBatchNorm2d(nn.BatchNorm2D):
    """
    How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)
    """
    def __init__(self, 
                 num_features, 
                 alpha=0.1, 
                 eps=1e-05, 
                 momentum=0.001, 
                 affine=True,
                 track_running_stats=True):
        super().__init__(num_features, momentum=momentum, epsilon=eps)
        self.alpha = alpha

    def forward(self, x):
        """
        forward
        Args:
            x: (n, c, h, w)
        """
        return super().forward(x) + self.alpha


class BasicBlock(nn.Layer):
    """
    set basic block
    """
    def __init__(self, in_channel, out_channel, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2D(in_channel, momentum=0.001, 
                                #   weight_attr=nn.initializer.Constant(value=1.), 
                                #   bias_attr=nn.initializer.Constant(value=0.)
                                  )
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv1 = nn.Conv2D(in_channel, out_channel, kernel_size=3, stride=stride,
                               padding=1, bias_attr=False, weight_attr=nn.initializer.KaimingNormal(nonlinearity='leaky_relu'))
        
        self.bn2 = nn.BatchNorm2D(out_channel, momentum=0.001,
                                #   weight_attr=nn.initializer.Constant(value=1.), 
                                #   bias_attr=nn.initializer.Constant(value=0.)
                                  )
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2D(out_channel, out_channel, kernel_size=3, stride=1,
                               padding=1, bias_attr=False, weight_attr=nn.initializer.KaimingNormal(nonlinearity='leaky_relu'))
        
        self.drop_rate = drop_rate
        self.equalInOut = (in_channel == out_channel)
        self.convShortcut = (not self.equalInOut) and nn.Conv2D(in_channel, 
                                                                out_channel, 
                                                                kernel_size=1, 
                                                                padding=0, 
                                                                stride=stride,
                                                                bias_attr=False,
                                                                weight_attr=nn.initializer.KaimingNormal(nonlinearity='leaky_relu')) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        """
        Args:
            x: (n, c, h, w)
        """
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))

        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.trainiing)
        out = self.conv2(out)
        return paddle.add(x if self.equalInOut else self.convShortcut(x), out)

        
class NetworkBlock(nn.Layer):
    """
    make network block
    """
    def __init__(self, nb_layers, 
                       in_channel, 
                       out_channel, 
                       block, 
                       stride, 
                       drop_rate=0.0, 
                       activate_before_residual=False):
        """
        make network block
        """
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_channel, out_channel, nb_layers, stride, drop_rate, activate_before_residual
        )

    def _make_layer(self, block, in_channel, out_channel, nb_layers, stride, drop_rate, activate_before_residual):
        """
        make block list
        """
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_channel or out_channel, out_channel,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (n, c, h, w)
        """
        return self.layer(x)


class Normalize(nn.Layer):
    """ Ln normalization copied from
    https://github.com/salesforce/CoMatch
    """
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        """
        Args:
            x: (n, c, h, w)
        """
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x / norm
        return out


class WideResNet(nn.Layer):
    """
    WideResNet network
    """
    def __init__(self,
                 num_classes,
                 depth=28,
                 widen_factor=2,
                 drop_rate=0.0,
                 proj=False,
                 proj_after=False,
                 low_dim=64):
        super(WideResNet, self).__init__()
        self.wide_factor = widen_factor
        self.depth = depth
        self.drop_rate = drop_rate
        self.proj = proj
        self.proj_after = proj_after
        self.low_dim = low_dim

        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        block = BasicBlock
        self.conv1 = nn.Conv2D(3, channels[0], kernel_size=3, stride=1, padding=1, bias_attr=False, weight_attr=nn.initializer.KaimingNormal(nonlinearity='leaky_relu'))

        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, 1, drop_rate, activate_before_residual=True
        )

        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate
        )

        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate
        )

        self.bn1 = nn.BatchNorm2D(channels[3], momentum=0.001, 
                                #   weight_attr=nn.initializer.Constant(value=1.), 
                                #   bias_attr=nn.initializer.Constant(value=0.)
                                  )
        self.relu = nn.LeakyReLU(negative_slope=0.1)

        if self.proj_after:
            self.fc = nn.Linear(self.low_dim, num_classes,
                                weight_attr=nn.initializer.XavierNormal(),
                                bias_attr=nn.initializer.Constant(value=0.))
        else:
            self.fc = nn.Linear(channels[3], num_classes,
                                weight_attr=nn.initializer.XavierNormal(),
                                bias_attr=nn.initializer.Constant(value=0.))

        if self.proj:
            self.l2norm = Normalize(2)
            self.fc1 = nn.Linear(64 * self.wide_factor, 64 * self.wide_factor,
                                 weight_attr=nn.initializer.XavierNormal(),
                                 bias_attr=nn.initializer.Constant(value=0.))
            self.relu_mlp = nn.LeakyReLU(negative_slope=0.1)
            self.fc2 = nn.Linear(64 * self.wide_factor, self.low_dim,
                                 weight_attr=nn.initializer.XavierNormal(),
                                 bias_attr=nn.initializer.Constant(value=0.))

        self.init_weights()

    def init_weights(self):
        """
        reset weight
        """
        # for m in self.sublayers:
        #     if isinstance(m, nn.Conv2D):
        pass

    def forward(self, x):
        """
        Args:
            x: (n, c, h, w)
        """
        feat = self.conv1(x)
        feat = self.block1(feat)
        feat = self.block2(feat)
        feat = self.block3(feat)
        feat = self.relu(self.bn1(feat))
        feat = F.adaptive_avg_pool2d(feat, 1)
        feat = feat.squeeze()

        if not self.training:
            return self.fc(feat)

        if self.proj:
            pfeat = self.fc1(feat)
            pfeat = self.relu_mlp(pfeat)
            pfeat = self.fc2(pfeat)
            pfeat = self.l2norm(pfeat)

            if self.proj_after:
                out = self.fc(pfeat)
            else:
                out = self.fc(feat)
            return out, pfeat

        out = self.fc(feat)
        return out


def WideResNetCCSSL(num_classes,
                    **kwargs):
    return WideResNet(num_classes=num_classes, **kwargs)      