from paddle import nn
import paddle
from typing import Optional, List
import paddle.nn.functional as F


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(
            self,
            in_channel,
            out_channel,
            stride=1,
            downsample: Optional[nn.Layer] = None,
            base_width: int = 64,
            nonlin: bool = True,
            embedding_dim: Optional[int] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if base_width / 64 > 1:
            raise ValueError("Base width >64 does not work for BasicBlock")
        if embedding_dim is not None:
            out_channel = embedding_dim

        self.conv1 = nn.Conv2D(in_channel, out_channel, kernel_size=3, padding=1,
                               stride=stride, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(out_channel, momentum=0.1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channel, out_channel, kernel_size=3, padding=1,
                               stride=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channel, momentum=0.1)
        self.downsample = downsample
        self.stride = stride
        self.nonlin = nonlin

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.nonlin:
            out = self.relu(out)
        return out


class Bottleneck(nn.Layer):
    expansion = 4
    def __init__(
            self,
            in_channel,
            out_channel,
            stride=1,
            downsample: Optional[nn.Layer] = None,
            base_width: int = 64,
            nonlin: bool = True,
            embedding_dim: Optional[int] = None,
    ):
        super(Bottleneck, self).__init__()
        super(Bottleneck, self).__init__()
        width = int(out_channel * base_width / 64)
        if embedding_dim is not None:
            out_dim = embedding_dim
        else:
            out_dim = out_channel * self.expansion

        self.conv1 = nn.Conv2D(in_channel, width, kernel_size=1, stride=1,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(width, momentum=0.1)
        self.conv2 = nn.Conv2D(width, width, kernel_size=3, padding=1,
                               stride=stride, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(width, momentum=0.1)
        self.conv3 = nn.Conv2D(width, out_dim, kernel_size=1, stride=1,
                               bias_attr=False)
        self.bn3 = nn.BatchNorm2D(out_dim, momentum=0.1)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.nonlin = nonlin

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.nonlin:
            out = self.relu(out)
        return out


class ResNet(nn.Layer):
    def __init__(
            self,
            block: nn.Layer,
            layers: List[int],
            num_classes: int = 1000,
            base_width: int = 64,
            embedding_dim: Optional[int] = None,
            last_nonlin: bool = True,
            norm_feature: bool = False,
    ) -> None:
        super(ResNet, self).__init__()

        self.in_channel = 64
        self.output_shape = [embedding_dim, 1, 1]
        self.is_normalized = norm_feature
        self.base_width = base_width
        if self.base_width // 64 > 1:
            print(f"==> Using {self.base_width // 64}x wide model")

        if embedding_dim is not None:
            print("Using given embedding dimension = {}".format(embedding_dim))
            self.embedding_dim = embedding_dim
        else:
            self.embedding_dim = 512 * block.expansion

        self.conv1 = nn.Conv2D(3, 64, 7, stride=2, padding=3, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64, momentum=0.1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], 64 * block.expansion)
        self.layer2 = self._make_layer(block, 128, layers[1], 128 * block.expansion, 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 256 * block.expansion, 2)
        self.layer4 = self._make_layer(block, 512, layers[3], self.embedding_dim, 2, last_nonlin)

        self.avgpool = nn.AdaptiveAvgPool2D(1)

        self.fc = nn.Conv2D(self.embedding_dim, num_classes, kernel_size=1,
                            stride=1, bias_attr=False)

    def _make_layer(
            self,
            block: nn.Layer,
            out_channel: int,
            blocks: int,
            embedding_dim: int,
            stride: int = 1,
            nonlin: bool = True
    ):
        downsample = None
        if stride != 1 or self.in_channel != out_channel * block.expansion:
            dconv = nn.Conv2D(self.in_channel, out_channel * block.expansion,
                              kernel_size=1,
                              stride=stride, bias_attr=False)
            dbn = nn.BatchNorm2D(out_channel * block.expansion, momentum=0.1)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        last_downsample = None

        layers = []
        if blocks == 1:  # If this layer has only one-block
            if stride != 1 or self.in_channel != embedding_dim:
                dconv = nn.Conv2D(self.in_channel, embedding_dim, kernel_size=1,
                                  stride=stride, bias_attr=False)
                dbn = nn.BatchNorm2D(embedding_dim, momentum=0.1)
                if dbn is not None:
                    last_downsample = nn.Sequential(dconv, dbn)
                else:
                    last_downsample = dconv
            layers.append(
                block(
                    self.in_channel, out_channel, stride, last_downsample,
                    self.base_width, nonlin, embedding_dim
                )
            )
            return nn.Sequential(*layers)
        else:
            layers.append(
                block(self.in_channel, out_channel, stride, downsample, base_width=self.base_width)
            )
        self.in_channel = out_channel * block.expansion
        for i in range(1, blocks - 1):
            layers.append(
                block(self.in_channel, out_channel, base_width=self.base_width)
            )

        if self.in_channel != embedding_dim:
            dconv = nn.Conv2D(self.in_channel, embedding_dim, stride=1,
                              kernel_size=1,
                              bias_attr=False)
            dbn = nn.BatchNorm2D(embedding_dim, momentum=0.1)
            if dbn is not None:
                last_downsample = nn.Sequential(dconv, dbn)
            else:
                last_downsample = dconv
        layers.append(
            block(self.in_channel, out_channel,
                  downsample=last_downsample, base_width=self.base_width,
                  nonlin=nonlin, embedding_dim=embedding_dim)
        )

        return nn.Sequential(*layers)

    def forward(self, x: paddle.Tensor) -> (paddle.Tensor, paddle.Tensor):
        x = self.conv1(x)

        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        feature = self.avgpool(x)
        if self.is_normalized:
            feature = F.normalize(feature)

        x = self.fc(feature)
        x = x.reshape((x.shape[0], -1))

        return x, feature


def ResNet18(num_classes: int,
             embedding_dim: int,
             last_nonlin: bool = True,
             **kwargs) -> nn.Layer:
    """Get a ResNet18 model.
    
    Args:
        num_classes: Number of classes in the dataset.
        embedding_dim: Size of the output embedding dimension.
        last_nonlin: Whether to apply non-linearity before output.
        **kwargs: 

    Returns: ResNet18 Model
    """

    return ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        last_nonlin=last_nonlin
    )


def ResNet50(num_classes: int,
             embedding_dim: int,
             last_nonlin: bool = True,
             **kwargs) -> nn.Layer:
    """Get a ResNet50 model.

    :param num_classes: Number of classes in the dataset.
    :param embedding_dim: Size of the output embedding dimension.
    :param last_nonlin: Whether to apply non-linearity before output.
    :return: ResNet18 Model.
    """
    return ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        last_nonlin=last_nonlin
    )


def ResNet101(num_classes: int,
              embedding_dim: int,
              last_nonlin: bool = True,
              **kwargs) -> nn.Layer:
    """Get a ResNet101 model.

    :param num_classes: Number of classes in the dataset.
    :param embedding_dim: Size of the output embedding dimension.
    :param last_nonlin: Whether to apply non-linearity before output.
    :return: ResNet18 Model.
    """
    return ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        last_nonlin=last_nonlin
    )


def WideResNet50_2(num_classes: int,
                   embedding_dim: int,
                   last_nonlin: bool = True,
                   **kwargs) -> nn.Layer:
    """Get a WideResNet50 model.

    :param num_classes: Number of classes in the dataset.
    :param embedding_dim: Size of the output embedding dimension.
    :param last_nonlin: Whether to apply non-linearity before output.
    :return: ResNet18 Model.
    """
    return ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        base_width=64 * 2,
        embedding_dim=embedding_dim,
        last_nonlin=last_nonlin
    )


def WideResNet101_2(num_classes: int,
                    embedding_dim: int,
                    last_nonlin: bool = True,
                    **kwargs) -> nn.Layer:
    """Get a WideResNet101 model.

    :param num_classes: Number of classes in the dataset.
    :param embedding_dim: Size of the output embedding dimension.
    :param last_nonlin: Whether to apply non-linearity before output.
    :return: ResNet18 Model.
    """
    return ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        num_classes=num_classes,
        base_width=64 * 2,
        embedding_dim=embedding_dim,
        last_nonlin=last_nonlin
    )


class ConvBlock(nn.Layer):
    """Convenience convolution module."""

    def __init__(self,
                 channels_in: int,
                 channels_out: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 normalizer: Optional[nn.Layer] = nn.BatchNorm2D,
                 activation: Optional[nn.Layer] = nn.ReLU) -> None:
        """Construct a ConvBlock module."""
        super().__init__()

        self.conv = nn.Conv2D(
            channels_in, channels_out,
            kernel_size=kernel_size, stride=stride,
            bias_attr=normalizer is None,
            padding=kernel_size // 2
        )
        if normalizer is not None:
            self.normalizer = normalizer(channels_out)
        else:
            self.normalizer = None
        if activation is not None:
            self.activation = activation()
        else:
            self.activation = None

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Apply forward pass."""
        x = self.conv(x)
        if self.normalizer is not None:
            x = self.normalizer(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class MLP_BN_SIDE_PROJECTION(nn.Layer):
    """FCT transformation module."""

    def __init__(self,
                 old_embedding_dim: int,
                 new_embedding_dim: int,
                 side_info_dim: int,
                 inner_dim: int = 2048,
                 **kwargs) -> None:
        """Construct MLP_BN_SIDE_PROJECTION module.
        """
        super().__init__()

        self.inner_dim = inner_dim
        self.p1 = nn.Sequential(
            ConvBlock(old_embedding_dim, 2 * old_embedding_dim),
            ConvBlock(2 * old_embedding_dim, 2 * new_embedding_dim),
        )

        self.p2 = nn.Sequential(
            ConvBlock(side_info_dim, 2 * side_info_dim),
            ConvBlock(2 * side_info_dim, 2 * new_embedding_dim),
        )

        self.mixer = nn.Sequential(
            ConvBlock(4 * new_embedding_dim, self.inner_dim),
            ConvBlock(self.inner_dim, self.inner_dim),
            ConvBlock(self.inner_dim, new_embedding_dim, normalizer=None,
                      activation=None)
        )

    def forward(self,
                old_feature: paddle.Tensor,
                side_info: paddle.Tensor) -> paddle.Tensor:
        """Apply forward pass.
        """
        x1 = self.p1(old_feature)
        x2 = self.p2(side_info)
        return self.mixer(paddle.concat([x1, x2], axis=1))


class FCTTrans(nn.Layer):
    def __init(self, *args, **kwargs) -> None:
        self.old_model = paddle.jit.load(old_model_path)
        self.new_model = paddle.jit.load(new_model_path)
        self.side_model = paddle.jit.load(side_model_path)
        self.old_model.eval()
        self.new_model.eval()
        self.side_model.eval()
        self.model = MLP_BN_SIDE_PROJECTION(*trans_config)

    def forward(self, images, labels) -> None:
        with paddle.no_rgrad():
            old_fea = self.old_model.forward(images)
            new_fea = self.new_model.forward(images)
            side_info = self.side_model.forward(images)

        recycled_fea = self.model(old_fea, side_info)
        return new_fea, recycled_fea


class FCT(nn.Layer):
    def __init__(self, *args, **kwargs) -> None:
        self.backbone = ResNet50(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)

