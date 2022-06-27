import torch
import torch.nn as nn

from .affine import Affine


class BatchNorm(nn.Module):
    """Batch Normalization applied to Last dimension instead of 2nd dimension"""

    def __init__(self, *args, **kwargs):
        super(BatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        return x

    def fuse_bn(self):
        dim = self.bn.bias.size(-1)
        new_layer = Affine(dim)
        
        bn = self.bn
        w = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        b = bn.bias - bn.running_mean * w

        new_layer.alpha = nn.Parameter(w.view(new_layer.alpha.size()))
        new_layer.beta = nn.Parameter(b.view(new_layer.beta.size()))

        return new_layer

def fold_bn_linear(bn: BatchNorm, linear: nn.Linear):
    bn = bn.bn
    w = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    b = linear.weight @ (bn.bias - bn.running_mean * w)

    linear.weight = nn.Parameter(linear.weight * w)
    linear.bias = nn.Parameter(linear.bias + b)

    return linear
