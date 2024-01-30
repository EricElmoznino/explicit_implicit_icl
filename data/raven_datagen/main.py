#############################################################
# Obtained from: https://openreview.net/forum?id=gx20B4ItIw #
#############################################################


from rule_RAVEN.task import Task
import argparse
import os

# N = 20
import rule_RAVEN.const_iid_inpo_expo_l2_20 as const_2_20
import rule_RAVEN.const_expo_l1_20 as const_1_20
# N = 30
import rule_RAVEN.const_iid_inpo_expo_l2_30 as const_2_30
import rule_RAVEN.const_expo_l1_30 as const_1_30
# N = 40
import rule_RAVEN.const_iid_inpo_expo_l2_40 as const_2_40
import rule_RAVEN.const_expo_l1_40 as const_1_40
# N = 80
import rule_RAVEN.const_iid_inpo_expo_l2_80 as const_2_80

import os

parser = argparse.ArgumentParser(description='rule RAVEN')
parser.add_argument('--data-dir', type=str, default="./data")
parser.add_argument('--samples-per-rule', type=int, default=100)
parser.add_argument('--test-samples-per-rule', type=int, default=20)

args = parser.parse_args()

def main():
    
    generalization_20_args = [
        {
            'mode': 'iid_inpo',
            'config': const_2_20.config_iid_inpo,
            'core_config': const_2_20.config_iid_inpo_core,
            'samples_per_rule': args.samples_per_rule,
            'test_samples_per_rule': args.test_samples_per_rule,
            'data_dir': os.path.join(args.data_dir, "l2_inpo_20")
        },
        {
            'mode': 'ood_expo_l1',
            'config': const_2_20.config_ood_expo_l1,
            'core_config': const_2_20.config_ood_expo_l1_core,
            'samples_per_rule': args.test_samples_per_rule,
            'test_samples_per_rule': args.test_samples_per_rule,
            'data_dir': os.path.join(args.data_dir, "l2_expo_20")
        },
        {
            'mode': 'iid_inpo',
            'config': const_1_20.config_iid_inpo,
            'core_config': const_1_20.config_iid_inpo_core,
            'samples_per_rule': args.samples_per_rule,
            'test_samples_per_rule': args.test_samples_per_rule,
            'data_dir': os.path.join(args.data_dir, "l1_inpo_20")
        },
        {
            'mode': 'ood_expo_l1',
            'config': const_1_20.config_ood_expo_l1,
            'core_config': const_1_20.config_ood_expo_l1_core,
            'samples_per_rule': args.test_samples_per_rule,
            'test_samples_per_rule': args.test_samples_per_rule,
            'data_dir': os.path.join(args.data_dir, "l1_expo_20")
        },
    ]

    generalization_30_args = [
        {
            'mode': 'iid_inpo',
            'config': const_2_30.config_iid_inpo,
            'core_config': const_2_30.config_iid_inpo_core,
            'samples_per_rule': args.samples_per_rule,
            'test_samples_per_rule': args.test_samples_per_rule,
            'data_dir': os.path.join(args.data_dir, "l2_inpo_30")
        },
        {
            'mode': 'ood_expo_l1',
            'config': const_2_30.config_ood_expo_l1,
            'core_config': const_2_30.config_ood_expo_l1_core,
            'samples_per_rule': args.test_samples_per_rule,
            'test_samples_per_rule': args.test_samples_per_rule,
            'data_dir': os.path.join(args.data_dir, "l2_expo_30")
        },
        {
            'mode': 'iid_inpo',
            'config': const_1_30.config_iid_inpo,
            'core_config': const_1_30.config_iid_inpo_core,
            'samples_per_rule': args.samples_per_rule,
            'test_samples_per_rule': args.test_samples_per_rule,
            'data_dir': os.path.join(args.data_dir, "l1_inpo_30")
        },
        {
            'mode': 'ood_expo_l1',
            'config': const_1_30.config_ood_expo_l1,
            'core_config': const_1_30.config_ood_expo_l1_core,
            'samples_per_rule': args.test_samples_per_rule,
            'test_samples_per_rule': args.test_samples_per_rule,
            'data_dir': os.path.join(args.data_dir, "l1_expo_30")
        },
    ]

    generalization_40_args = [
        {
            'mode': 'iid_inpo',
            'config': const_2_40.config_iid_inpo,
            'core_config': const_2_40.config_iid_inpo_core,
            'samples_per_rule': args.samples_per_rule,
            'test_samples_per_rule': args.test_samples_per_rule,
            'data_dir': os.path.join(args.data_dir, "l2_inpo_40")
        },
        {
            'mode': 'ood_expo_l1',
            'config': const_2_40.config_ood_expo_l1,
            'core_config': const_2_40.config_ood_expo_l1_core,
            'samples_per_rule': args.test_samples_per_rule,
            'test_samples_per_rule': args.test_samples_per_rule,
            'data_dir': os.path.join(args.data_dir, "l2_expo_40")
        },
        {
            'mode': 'iid_inpo',
            'config': const_1_40.config_iid_inpo,
            'core_config': const_1_40.config_iid_inpo_core,
            'samples_per_rule': args.samples_per_rule,
            'test_samples_per_rule': args.test_samples_per_rule,
            'data_dir': os.path.join(args.data_dir, "l1_inpo_40")
        },
        {
            'mode': 'ood_expo_l1',
            'config': const_1_40.config_ood_expo_l1,
            'core_config': const_1_40.config_ood_expo_l1_core,
            'samples_per_rule': args.test_samples_per_rule,
            'test_samples_per_rule': args.test_samples_per_rule,
            'data_dir': os.path.join(args.data_dir, "l1_expo_40")
        },
    ]

    generalization_80_args = [
        {
            'mode': 'iid_inpo',
            'config': const_2_80.config_iid_inpo,
            'core_config': const_2_80.config_iid_inpo_core,
            'samples_per_rule': args.samples_per_rule,
            'test_samples_per_rule': args.test_samples_per_rule,
            'data_dir': os.path.join(args.data_dir, "l2_inpo_80")
        },
        {
            'mode': 'ood_expo_l1',
            'config': const_2_80.config_ood_expo_l1,
            'core_config': const_2_80.config_ood_expo_l1_core,
            'samples_per_rule': args.test_samples_per_rule,
            'test_samples_per_rule': args.test_samples_per_rule,
            'data_dir': os.path.join(args.data_dir, "l2_expo_80")
        },
    ]
    
    # generalization
    for ga in [generalization_20_args, generalization_30_args, generalization_40_args, generalization_80_args]:
        for a in ga:
            t = Task("center_single", **a)
            t.generate_pkl()
    fix_list = [
        i 
        for i in os.listdir(args.data_dir)
        if "expo" in i
    ] # inpo and expo share the same train and valid
    for i in fix_list:
        src_dir = os.path.join(args.data_dir, i.replace("expo", "inpo"))
        dst_dir = os.path.join(args.data_dir, i)
        for fn in ["train", "validation"]:
            cmd = "cp %s %s" % (os.path.join(src_dir, "%s_visual.pkl" % fn), dst_dir)
            os.system(cmd)
    

if __name__ == "__main__":
    main()
