import paddle.nn as nn


class Normalize(nn.Layer):
    """ Ln normalization copied from
    https://github.com/salesforce/CoMatch
    """

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.divide(norm)
        return out


class FRFNNeck(nn.Layer):
    def __init__(self, num_features, low_dim, **kwargs):
        super(FRFNNeck, self).__init__()
        self.l2norm = Normalize(2)
        self.fc1 = nn.Linear(num_features, num_features)
        self.relu_mlp = nn.LeakyReLU(negative_slope=0.1)
        self.fc2 = nn.Linear(num_features, low_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu_mlp(x)
        x = self.fc2(x)
        x = self.l2norm(x)
        return x