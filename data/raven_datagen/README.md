# rule-RAVEN

This folder contains the source code of `rule-RAVEN` dataset. To reproduce the dataset we used in the experiment, please refer to the following step:

```shell
python main.py # generate all data splits in our paper
```

You can see that the following folders are generated in `./data`:

- `warmup_[20, 30, 40]`: Data for training the speaker in stage one. The number at the end represents attribute values $N$.

- `[l1, l2]_[inpo, expo]_[20, 30, 40, 80]`: Data splits for generalization. The number at the end represents attribute values $N$. The four generalization accuracies in the paper correspond to the following table.

	| Generalization Level |       Accuracy of       |
	| :------------------: | :---------------------: |
	|         `ID`         | `l2_inpo`, `validation` |
	|      `Inpo-ood`      |    `l2_inpo`, `test`    |
	|    `Expo-ood-L1`     |    `l1_expo`, `test`    |
	|    `Expo-ood-L2`     |    `l2_expo`, `test`    |

	

- `ablation`: Data for comparing `rule-RAVEN` and `I-RAVEN`.