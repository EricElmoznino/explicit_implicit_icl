import os

import torch
from torch.nn import Linear
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.utils.data import DataLoader

import sys
import hydra
from omegaconf import OmegaConf

sys.path.append('/home/mila/l/leo.gagnon/explicit_implicit_icl')

def main(experiment):

    conf = OmegaConf.load(f"configs/experiment/{experiment}/explicit_mlp.yaml")
    dm = hydra.utils.instantiate(conf.data)

    data = dm.train_data
    # To be able to train a probe on the context 
    data.min_context = 72
    data.max_context = 72

    # To get the IO shape of probe
    (x_c, y_c), (x_q, y_q), w = next(iter(data))
    assert w != None, "This experiment doesn't have a latent to decode"

    c_dim = x_c[0].numel() + y_c[0].numel()
    z_dim = w[0].numel()

    probe = Linear(c_dim, z_dim)
    opt = Adam(probe.parameters())

    # Train
    for epoch in range(conf.trainer.max_epochs):
        dl = DataLoader(data, batch_size=None)
        for batch in dl:

            opt.zero_grad()
            
            if experiment == 'raven':
                x_c, x_q, y_q_candidates, y_q_label, rule = batch
                c, w = x_c, rule
            else:
                (x_c, y_c), (x_q, y_q), w = batch
                c = torch.cat([x_c.flatten(1), y_c.flatten(1)], dim=-1)
            w_pred = probe(c).view(*w.shape)
            loss = mse_loss(w_pred, w)

            loss.backward()
            opt.step()

    # Val
    for key, dl in zip(dm.val_data.keys(), dm.val_dataloader()):
        for batch in dl:
            if experiment == 'raven':
                x_c, x_q, y_q_candidates, y_q_label, rule = batch
                c, w = x_c, rule
            else:
                (x_c, y_c), (x_q, y_q), w = batch
                c = torch.cat([x_c.flatten(1), y_c.flatten(1)], dim=-1)
            w_pred = probe(c).view(*w.shape)
            loss = mse_loss(w_pred, w)
            print(f'{key} val loss : {loss}')
                


if __name__ == "__main__":
    main('raven')
