import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, StudentT
import math

class BatchedLinear(nn.Module):
    def __init__(self, in_features, out_features, num_parallels, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_parallels = num_parallels

        self.weight = nn.Parameter(torch.Tensor(num_parallels, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_parallels, 1, out_features))
        else:
            self.register_parameter("bias", None)
    
    def forward(self, x):
        # x: (num_parallels, nseq_len, in_features)

        out = torch.bmm(x, self.weight)
        if self.bias is not None:
            out += self.bias
        
        return out
    
class RBFKernel(object):
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: batch_size * num_points * dim  [B,N,Dx=1]
    def __call__(self, x):
        length = 0.1 + (self.max_length-0.1) \
                * torch.rand([x.shape[0], 1, 1, 1], device=x.device)
        scale = 0.1 + (self.max_scale-0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)

        # batch_size * num_points * num_points * dim  [B,N,N,1]
        dist = (x.unsqueeze(-2) - x.unsqueeze(-3))/length

        # batch_size * num_points * num_points  [B,N,N]
        cov = scale.pow(2) * torch.exp(-0.5 * dist.pow(2).sum(-1)) \
                + self.sigma_eps**2 * torch.eye(x.shape[-2]).to(x.device)

        return cov  # [B,N,N]

class Matern52Kernel(object):
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: batch_size * num_points * dim
    def __call__(self, x):
        length = 0.1 + (self.max_length-0.1) \
                * torch.rand([x.shape[0], 1, 1, 1], device=x.device)
        scale = 0.1 + (self.max_scale-0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)

        # batch_size * num_points * num_points
        dist = torch.norm((x.unsqueeze(-2) - x.unsqueeze(-3))/length, dim=-1)

        cov = scale.pow(2)*(1 + math.sqrt(5.0)*dist + 5.0*dist.pow(2)/3.0) \
                * torch.exp(-math.sqrt(5.0) * dist) \
                + self.sigma_eps**2 * torch.eye(x.shape[-2]).to(x.device)

        return cov

class PeriodicKernel(object):
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        #self.p = p
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: batch_size * num_points * dim
    def __call__(self, x):
        p = 0.1 + 0.4*torch.rand([x.shape[0], 1, 1], device=x.device)
        length = 0.1 + (self.max_length-0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)
        scale = 0.1 + (self.max_scale-0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)

        dist = x.unsqueeze(-2) - x.unsqueeze(-3)
        cov = scale.pow(2) * torch.exp(\
                - 2*(torch.sin(math.pi*dist.abs().sum(-1)/p)/length).pow(2)) \
                + self.sigma_eps**2 * torch.eye(x.shape[-2]).to(x.device)

        return cov