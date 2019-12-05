
import torch as t, torch.nn as nn
from einops.layers.torch import Rearrange


def insert_bnorm(layers, init_gain=False, eps=1e-5, ignore_last_layer=True):
  """Inserts batch-norm layers after each convolution/linear layer in a list of layers."""
  last = True
  for (idx, layer) in reversed(list(enumerate(layers))):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
      if ignore_last_layer and last:
        last = False  # do not insert batch-norm after last linear/conv layer
      else:
        if isinstance(layer, nn.Conv2d):
          bnorm = nn.BatchNorm2d(layer.out_channels, eps=eps)
        elif isinstance(layer, nn.Linear):
          bnorm = nn.BatchNorm1d(layer.out_features, eps=eps)
        
        if init_gain:
          bnorm.weight.data[:] = 1.0  # instead of uniform sampling

        layers.insert(idx + 1, bnorm)
  return layers

def init_conv(layer, std=0.01):
  """Initialize a conv layer with zero bias and gaussian-sampled weights"""
  layer.weight.data = t.normal(mean=t.zeros_like(layer.weight), std=std)
  layer.bias.data.zero_()

def get_two_layers_cnn(args, input_channels=2):
  """Return a 2-layers CNN"""
  layers = [
    nn.Conv2d(input_channels, 20, kernel_size=3),
    nn.ReLU(inplace=False),
    nn.Conv2d(20, 16, kernel_size=3),
  ]
  
  init_conv(layers[0], std=0.01)
  init_conv(layers[2], std=0.01)

  if args.bn:
    insert_bnorm(layers, ignore_last_layer=False, init_gain=True, eps=1e-4)

  return nn.Sequential(*layers)
