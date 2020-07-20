import paddle
import torch
import numpy as np
import paddle.fluid as fluid

def instance_std_paddle(input, epsilon=1e-5):
    v = paddle.var(input, axis=[2,3], keepdim=True )
    v = paddle.expand_as(v, input)
    return  paddle.sqrt(v+epsilon)

def instance_std(x, eps=1e-5):
    var = torch.var(x, dim = (2, 3), keepdim=True).expand_as(x)
    if torch.isnan(var).any():
        var = torch.zeros(var.shape)
    return torch.sqrt(var + eps)

def group_std_paddle(input, groups=32, epsilon=1e-5):
    #N, C, H, W = paddle.shape(input)
    N,C,H,W = input.shape
    #print(N,C,H,W)
    input = paddle.reshape(input, [N, groups, C//groups, H, W])
    v = paddle.var(input, axis=[2,3,4], keepdim=True)
    v = paddle.expand_as(v, input)
    return paddle.reshape(paddle.sqrt(v+epsilon),(N,C,H,W))

def group_std(x, groups = 32, eps = 1e-5):
    N, C, H, W = x.size()
    x = torch.reshape(x, (N, groups, C // groups, H, W))
    var = torch.var(x, dim = (2, 3, 4), keepdim = True).expand_as(x)
    return torch.reshape(torch.sqrt(var + eps), (N, C, H, W))



class EvoNorm(fluid.dygraph.Layer):
    def __init__(self, channels, version='B0', affine=True, non_linear=True, groups=32, epsilon=1e-5,momentum=0.9, training=True):
        super(EvoNorm, self).__init__()
        self.channels = channels
        self.affine = affine
        self.version = version
        self.non_linear = non_linear
        self.groups = groups
        self.epsilon = epsilon
        self.training = training 
        self.momentum = momentum

        if self.affine:

            self.gamma = self.create_parameter([1, self.channels, 1, 1],
                default_initializer=fluid.initializer.Constant(value=1.0))
            self.beta = self.create_parameter([1, self.channels, 1, 1],
                default_initializer=fluid.initializer.Constant(value=0.0))
            if self.non_linear:
                self.v = self.create_parameter([1, self.channels, 1, 1],
                    default_initializer=fluid.initializer.Constant(value=1.0))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
            self.register_buffer('v', None)

        #self.running_var = self.create_parameter([1, self.channels, 1, 1],
        #        default_initializer=fluid.initializer.Constant(value=0.0))
        #self.running_var.stop_gradient = True
        #self.register_buffer('running_var', self.create_parameter([1, self.channels, 1, 1],
        #           default_initializer=fluid.initializer.Constant(value=1.0)))
        self.register_buffer('running_var', paddle.fluid.layers.ones(shape=[1,self.channels,1,1], dtype='float32'))

    def forward(self, input):
        if self.version == 'S0':
            if self.non_linear:
                num = input * paddle.fluid.layers.sigmoid(self.v * input)
                return num / group_std_paddle(input, groups=self.groups, epsilon=self.epsilon) * self.gamma + self.beta
            else:
                return input * self.gamma + self.beta
        if self.version == 'B0':
            if self.training:
                var = paddle.var(input, axis=[0,2,3], unbiased=False, keepdim=True)
                self.running_var = self.running_var * self.momentum
                self.running_var = self.running_var + (1- self.momentum) * var
            else:
                var = self.running_var
            if self.non_linear:
                den = paddle.elementwise_max(paddle.sqrt((var+self.epsilon)), self.v * input + instance_std_paddle(input, epsilon=self.epsilon))
                return input / den * self.gamma + self.beta
            else:
                return input * self.gamma + self.beta

