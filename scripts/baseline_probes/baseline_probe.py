import os

import torch
from torch.nn import Linear
from torch.nn.functional import mse_loss, cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassExactMatch
import sys
import hydra
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import json

from tqdm import tqdm

sys.path.append("/home/mila/l/leo.gagnon/explicit_implicit_icl")


def train_probe(experiment):

    conf = OmegaConf.load(f"configs/experiment/{experiment}/explicit_mlp.yaml")
    dm = hydra.utils.instantiate(conf.data)
    if experiment == "raven":
        dm.setup(None)
        raven_acc = MulticlassExactMatch(num_classes=dm.n_rules)

    data = dm.train_data
    # To be able to train a probe on the context
    if experiment != "raven":
        data.min_context = 72
        data.max_context = 72

    # To get the IO shape of probe
    if experiment == "raven":
        x_c, x_q, x_q_candidates, y_q_label, rule = next(iter(data))
        c_dim = x_c.numel()
        z_dim = dm.n_attributes * dm.n_rules
    else:
        (x_c, y_c), (x_q, y_q), w = next(iter(data))
        assert w != None, "This experiment doesn't have a latent to decode"
        c_dim = x_c[0].numel() + y_c[0].numel()
        z_dim = w[0].numel()

    probe = Linear(c_dim, z_dim).to("cuda")
    opt = Adam(probe.parameters())

    # Train
    for epoch in tqdm(
        range(
            50
            if experiment
            in ["raven", "lowrankmlp_regression", "lowrankmlp_classification"]
            else conf.trainer.max_epochs
        )
    ):
        dl = DataLoader(
            data, batch_size=conf.data.batch_size if experiment == "raven" else None
        )
        for batch in dl:

            opt.zero_grad()

            if experiment == "raven":
                x_c, x_q, y_q_candidates, y_q_label, rule = batch
                c = x_c.flatten(1)
                rule_pred = probe(c.cuda()).view(-1, dm.n_rules, dm.n_attributes).cpu()
                loss = cross_entropy(rule_pred, rule)
            else:
                (x_c, y_c), (x_q, y_q), w = batch
                c = torch.cat([x_c.cpu().flatten(1), y_c.cpu().flatten(1)], dim=-1)
                w_pred = probe(c.cuda()).view(*w.shape).cpu()
                loss = mse_loss(w_pred, w)

            loss.backward()
            opt.step()

        if experiment == "raven":
            keys = ["IID", "OOD"]
        else:
            keys = dm.val_data.keys()

        metrics = []
        for key, dl in zip(keys, dm.val_dataloader()):
            key_metric = []
            for batch in dl:
                if experiment == "raven":
                    x_c, x_q, y_q_candidates, y_q_label, rule = batch
                    c = x_c.flatten(1)
                    rule_pred = (
                        probe(c.cuda()).view(-1, dm.n_rules, dm.n_attributes).cpu()
                    )
                    key_metric += [float(raven_acc(rule_pred, rule))]
                else:
                    (x_c, y_c), (x_q, y_q), w = batch
                    c = torch.cat([x_c.cpu().flatten(1), y_c.cpu().flatten(1)], dim=-1)
                    w_pred = probe(c.cuda()).view(*w.shape).cpu()
                    loss = mse_loss(w_pred, w)
                    key_metric += [float(loss)]
                    
            metrics += [sum(key_metric) / len(key_metric)]

    return metrics


if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]
    experiments = [
        "lowrankmlp_classification",
        "lowrankmlp_regression",
        "linear_regression",
        "linear_classification",
        "raven",
        "sinusoid_regression",
    ]
    all_metrics = []
    for experiment in experiments:
        seed_metrics = []
        for seed in seeds:
            seed_everything(seed)
            seed_metrics += [train_probe(experiment)]
        all_metrics += [seed_metrics]

    with open("probe_data_.json", "w") as f:
        json.dump(all_metrics, f)
