import os

import torch
from torch.nn import Linear
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.utils.data import DataLoader

os.chrdir("../..")

import sys
import hydra
from omegaconf import OmegaConf


def main(experiment):

    conf = OmegaConf.load(f"configs/experiment/{experiment}.yaml")
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

            (x_c, y_c), (x_q, y_q), w = batch
            c = torch.cat([x_c, y_c], dim=1).view(x_c.shape[0], -1)
            w_pred = probe(c).view(*w.shape)
            loss = mse_loss(w_pred, w)

            loss.backward()
            opt.step()

    # Val
    for key, dl in zip(dm.val_data.keys(), dm.val_dataloader()):
        for batch in dl:
            (x_c, y_c), (x_q, y_q), w = batch
            c = torch.cat([x_c, y_c], dim=1).view(x_c.shape[0], -1)
            w_pred = probe(c).view(*w.shape)
            loss = mse_loss(w_pred, w)
            print(f'{key} val loss : {loss}')
                


if __name__ == "__main__":
    main(sys.argv[1])
