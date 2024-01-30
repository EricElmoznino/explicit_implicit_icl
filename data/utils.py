import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, StudentT
import math
from typing import Tuple
from torch import Tensor
from tqdm import tqdm

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

class HodgkinHuxleyODE(nn.Module):
    def __init__(
        self,
        g_leak: float = 0.1,
        g_bar_Na: float = 4.0,
        g_bar_K: float = 1.5,
        g_bar_M: float = 0.07,
        C: float = 1.0,
        E_Na: float = 53,
        E_K: float = -107,
        E_leak: float = -70,
        V0: float = -70.0,
        Vt: float = -60.0,
        tau_max: float = 6e2,
        I_on: float = 0.0,
        I_off: float = np.inf,
        curr_level=5e-4,
    ) -> None:
        super().__init__()
        self.batch_size = 1
        self.register_buffer("gbar_Na", torch.as_tensor(g_bar_Na))
        self.register_buffer("gbar_K", torch.as_tensor(g_bar_K))
        self.register_buffer("gbar_M", torch.as_tensor(g_bar_M))
        self.register_buffer("C", torch.as_tensor(C))
        self.register_buffer("E_leak", torch.as_tensor(E_leak))
        self.register_buffer("E_Na", torch.as_tensor(E_Na))
        self.register_buffer("E_K", torch.as_tensor(E_K))
        self.register_buffer("g_leak", torch.as_tensor(g_leak))
        self.register_buffer("Vt", torch.as_tensor(Vt))
        self.register_buffer("V0", torch.as_tensor(V0))
        self.register_buffer("tau_max", torch.as_tensor(tau_max))
        self.register_buffer("I_on", torch.as_tensor(I_on))
        self.register_buffer("I_off", torch.as_tensor(I_off))
        self.register_buffer("curr_level", torch.as_tensor(curr_level))
        self.register_buffer(
            "A_soma", torch.as_tensor(3.141592653589793 * ((70.0 * 1e-4) ** 2))
        )

    @torch.jit.export  # type: ignore
    def get_initial_state(self):
        V0 = self.V0.repeat(self.batch_size, 1)  # type: ignore
        return V0, self.n_inf(V0), self.m_inf(V0), self.h_inf(V0), self.p_inf(V0)

    @torch.jit.export  # type: ignore
    def efun(self, z: Tensor) -> Tensor:
        mask = torch.abs(z) < 1e-4
        new_z = torch.zeros_like(z, device=z.device)
        new_z[mask] = 1 - z[mask] / 2
        new_z[~mask] = z[~mask] / (torch.exp(z[~mask]) - 1)
        return new_z

    @torch.jit.export  # type: ignore
    def I_in(self, t: float) -> Tensor:
        if t > self.I_on and t < self.I_off:  # type: ignore
            return self.curr_level / self.A_soma  # type: ignore
        else:
            return torch.zeros(1, device=self.A_soma.device)  # type: ignore

    @torch.jit.export  # type: ignore
    def alpha_m(self, x: Tensor) -> Tensor:
        v1 = x - self.Vt - 13.0
        return 0.32 * self.efun(-0.25 * v1) / 0.25

    @torch.jit.export  # type: ignore
    def beta_m(self, x: Tensor) -> Tensor:
        v1 = x - self.Vt - 40
        return 0.28 * self.efun(0.2 * v1) / 0.2

    @torch.jit.export  # type: ignore
    def alpha_h(self, x: Tensor) -> Tensor:
        v1 = x - self.Vt - 17.0
        return 0.128 * torch.exp(-v1 / 18.0)

    @torch.jit.export  # type: ignore
    def beta_h(self, x: Tensor) -> Tensor:
        v1 = x - self.Vt - 40.0
        return 4.0 / (1 + torch.exp(-0.2 * v1))

    @torch.jit.export  # type: ignore
    def alpha_n(self, x: Tensor) -> Tensor:
        v1 = x - self.Vt - 15.0
        return 0.032 * self.efun(-0.2 * v1) / 0.2

    @torch.jit.export  # type: ignore
    def beta_n(self, x: Tensor) -> Tensor:
        v1 = x - self.Vt - 10.0
        return 0.5 * torch.exp(-v1 / 40)

    @torch.jit.export  # type: ignore
    def tau_n(self, x: Tensor) -> Tensor:
        return 1 / (self.alpha_n(x) + self.beta_n(x))

    @torch.jit.export  # type: ignore
    def n_inf(self, x: Tensor) -> Tensor:
        return self.alpha_n(x) / (self.alpha_n(x) + self.beta_n(x))

    @torch.jit.export  # type: ignore
    def tau_m(self, x: Tensor) -> Tensor:
        return 1 / (self.alpha_m(x) + self.beta_m(x))

    @torch.jit.export  # type: ignore
    def m_inf(self, x: Tensor) -> Tensor:
        return self.alpha_m(x) / (self.alpha_m(x) + self.beta_m(x))

    @torch.jit.export  # type: ignore
    def tau_h(self, x: Tensor) -> Tensor:
        return 1 / (self.alpha_h(x) + self.beta_h(x))

    @torch.jit.export  # type: ignore
    def h_inf(self, x: Tensor) -> Tensor:
        return self.alpha_h(x) / (self.alpha_h(x) + self.beta_h(x))

    @torch.jit.export  # type: ignore
    def p_inf(self, x: Tensor) -> Tensor:
        v1 = x + 35.0
        return 1.0 / (1.0 + torch.exp(-0.1 * v1))

    @torch.jit.export  # type: ignore
    def tau_p(self, x: Tensor) -> Tensor:
        v1 = x + 35.0
        return self.tau_max / (3.3 * torch.exp(0.05 * v1) + torch.exp(-0.05 * v1))

    def forward(
        self, t: float, state: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        V, n, m, h, p = state

        dV = (
            (m**3) * self.gbar_Na * h * (self.E_Na - V)
            + (n**4) * self.gbar_K * (self.E_K - V)
            + self.gbar_M * p * (self.E_K - V)
            + self.g_leak * (self.E_leak - V)
            + self.I_in(t)
        )
        dV = self.C * dV
        dn = (self.n_inf(V) - n) / self.tau_n(V)
        dm = (self.m_inf(V) - m) / self.tau_m(V)
        dh = (self.h_inf(V) - h) / self.tau_h(V)
        dp = (self.p_inf(V) - p) / self.tau_p(V)

        return dV, dn, dm, dh, dp
    