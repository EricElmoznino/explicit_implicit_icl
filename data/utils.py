import torch
import torch.nn as nn

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