from .task import Task
import argparse
import os

from .const_iid_inpo_expo_l2_40 import config_iid_inpo, config_iid_inpo_core

import os

parser = argparse.ArgumentParser(description="rule RAVEN")
parser.add_argument("--data-dir", type=str, default="./data")
parser.add_argument("--samples-per-rule", type=int, default=100)
parser.add_argument("--test-samples-per-rule", type=int, default=20)

args = parser.parse_args()


def main():
    generalization_40_args = [
        {
            "mode": "iid_inpo",
            "config": config_iid_inpo,
            "core_config": config_iid_inpo_core,
            "samples_per_rule": args.samples_per_rule,
            "test_samples_per_rule": args.test_samples_per_rule,
            "data_dir": os.path.join(args.data_dir, "l2_inpo_40"),
        },
    ]

    # generalization
    for a in generalization_40_args:
        t = Task("center_single", **a)
        t.generate_pkl()


if __name__ == "__main__":
    main()
