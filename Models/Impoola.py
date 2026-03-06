import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_init_kaiming_uniform(layer, a=0, nonlinearity='relu'):
    nn.init.kaiming_uniform_(layer.weight, a=a, nonlinearity=nonlinearity)
    # nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
    nn.init.constant_(layer.bias, 0)
    return layer


def layer_init_conv_torch_standard(layer):
    nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
    if layer.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        if fan_in != 0:
            bound = 1 / np.sqrt(fan_in)
            layer.uniform_(layer.bias, -bound, bound)
    return layer


def layer_init_xavier_uniform(layer, gain=1.0):
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)
    return layer


def layer_init_orthogonal(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init_normed(layer, norm_dim, scale=1.0):
    with torch.no_grad():
        layer.weight.data *= scale / layer.weight.norm(dim=norm_dim, p=2, keepdim=True)
        layer.bias *= 0
    return layer


def activation_factory(activation):
    if isinstance(activation, str):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'rrelu':
            return nn.RReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'silu':
            return nn.SiLU()
        else:
            raise NotImplementedError
    else:
        return activation()


class ResidualBlock(nn.Module):
    def __init__(self, channels, scale=1.0, use_layer_init_normed=False, activation='relu'):
        super().__init__()

        kernel_size = 3
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding='same')
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding='same')
        layer_init_orthogonal(self.conv0)
        layer_init_orthogonal(self.conv1)

        # if use_layer_init_normed:
        #     layer_init_normed(self.conv0, norm_dim=1, scale=scale)
        #     layer_init_normed(self.conv1, norm_dim=1, scale=scale)
        # else:
        #     with torch.no_grad():
        #         self.conv0.weight.data *= scale
        #         self.conv1.weight.data *= scale

        self.activation0 = activation_factory(activation)
        self.activation1 = activation_factory(activation)

    def forward(self, x):
        inputs = x
        x = self.activation0(x)
        x = self.conv0(x)
        x = self.activation1(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels, scale, use_layer_init_normed=False, activation='relu'):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels

        conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3,
                         padding="same")
        self.conv = conv
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        nblocks = 2
        scale = scale / np.sqrt(nblocks)
        self.res_block0 = ResidualBlock(self._out_channels, scale=scale, use_layer_init_normed=use_layer_init_normed,
                                        activation=activation)
        self.res_block1 = ResidualBlock(self._out_channels, scale=scale, use_layer_init_normed=use_layer_init_normed,
                                        activation=activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return self._out_channels, (h + 1) // 2, (w + 1) // 2


class Imapala_Backbone(nn.Module):
    def __init__(
            self, 
            shape,
            cnn_filters=(16, 32, 32),
            activation='relu',
            use_AvgPool=True,
            pooling_size=1,
    ):
        super().__init__()
        
        scale = 1 / np.sqrt(len(cnn_filters))
        
        # Build CNN layers
        cnn_layers = []
        current_shape = shape
        
        for out_channels in cnn_filters:
            conv_seq = ConvSequence(
                current_shape, 
                out_channels, 
                scale=scale,
                activation=activation
            )
            cnn_layers.append(conv_seq)
            current_shape = conv_seq.get_output_shape()
        
        # Add final activation
        cnn_layers.append(activation_factory(activation))
        
        # Add pooling if requested
        if use_AvgPool:
            # Correct calculation for tuples/torch.Size
            final_shape = tuple(s // 2**len(cnn_filters) for s in shape[-2:])
            stride = tuple(max(1, s // pooling_size) for s in final_shape)
            kernel_size = tuple(max(1, s - (pooling_size - 1) * st) for s, st in zip(final_shape, stride))
            cnn_layers.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride))
            # cnn_layers.append(nn.AdaptiveAvgPool2d((pooling_size, pooling_size)))
        
        self.network = nn.Sequential(*cnn_layers)
    
    def forward(self, x):
        return self.network(x)

def build_continuous_actor(input_dim, output_dim, NonLinearity=nn.SiLU):
    class DoubleHead(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.mu = layer_init_orthogonal(nn.Linear(in_dim, out_dim), std=1.0)
            self.log_std = layer_init_orthogonal(nn.Linear(in_dim, out_dim), std=0.01, bias_const=np.log(0.5))
        def forward(self, x):
            return self.mu(x), self.log_std(x)
    actor_net = nn.Sequential(
        layer_init_orthogonal(nn.Linear(input_dim, 128), std=np.sqrt(2)), # Gain approx per SiLU
        NonLinearity(),
        layer_init_orthogonal(nn.Linear(128, 128), std=np.sqrt(2)),
        NonLinearity(),
        DoubleHead(in_dim=128, out_dim=output_dim)
    )
    return actor_net

def build_discrete_actor(input_dim, output_dim, NonLinearity=nn.SiLU):
    actor_net = nn.Sequential(
        layer_init_orthogonal(nn.Linear(input_dim, 128), std=np.sqrt(2)), # Gain approx per SiLU
        NonLinearity(),
        layer_init_orthogonal(nn.Linear(128, 128), std=np.sqrt(2)),
        NonLinearity(),
        layer_init_orthogonal(nn.Linear(128, output_dim), std=0.01)
    )
    return actor_net

def build_critic(input_dim, NonLinearity=nn.SiLU):
    critic_net = nn.Sequential(
        layer_init_orthogonal(nn.Linear(input_dim, 128), std=np.sqrt(2)),
        NonLinearity(),
        layer_init_orthogonal(nn.Linear(128, 128), std=np.sqrt(2)),
        NonLinearity(),
        # Gain = 1.0 per il Critic
        layer_init_orthogonal(nn.Linear(128, 1), std=1.0)
    )
    return critic_net

