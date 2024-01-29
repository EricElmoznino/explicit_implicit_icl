import torch
from utils import HodgkinHuxleyODE
from torchdiffeq import odeint
import pickle
from tqdm import tqdm

if __name__ == "__main__":
    N_TIMESTEPS = 1000
    T_MAX = 120

    params = torch.stack(torch.meshgrid(torch.linspace(0,40,80), torch.linspace(0,40,80))).transpose(0,-1).flatten(0,-2)
    params = [{'g_bar_Na': e[0], 'g_bar_K': e[1]} for e in params]
    data = torch.zeros((len(params), N_TIMESTEPS))

    for i, p in tqdm(enumerate(params)):
        hh_ode = torch.jit.script(HodgkinHuxleyODE(**p))
        t = torch.linspace(0.0, T_MAX, N_TIMESTEPS, device='cpu')
        V, _, _, _, _ = odeint(
            hh_ode,
            hh_ode.get_initial_state(),
            t,
            method='bosh3',
            atol=1e-4,
            rtol=1e-3
        )
        data[i] = V.squeeze()

    with open('hh_data.pkl', 'wb') as f:
        pickle.dump({'data': data, 'params': params}, f)