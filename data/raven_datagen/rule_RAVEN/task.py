from itertools import permutations, combinations, product

import copy
from .rule import create_rule, RuleGroup, deserialize_rule_key
import random
import tqdm
import pickle
import numpy as np
import os

# random.seed(19990809)


class Task(object):
    def __init__(self, task, mode, config, data_dir, core_config=None, samples=None, samples_per_rule=None, test_samples_per_rule=None, debug=False):
        assert task in ['center_single']
        self.task = task
        self.mode = mode
        self.samples = samples
        self.samples_per_rule = samples_per_rule
        self.test_samples_per_rule = test_samples_per_rule
        self.data_dir = data_dir
        self.config = config  # used to generate candidates
        # used to generate data, only used in method: generate_iid/ood
        self.core_config = core_config

        self.value_space = {}  # attr -> space
        self.projected_value_space = {}  # attr -> rule -> space
        for attr in self.config[self.task]['attrs']:
            self.value_space.update(self.generate_value_space(attr))

        if debug:
            for k in self.value_space:
                print(k, len(self.value_space[k]))

        for attr in self.config[self.task]['attrs']:
            self.projected_value_space.update(
                self.project_value_space_to_rule(attr))

        if debug:
            for attr in self.projected_value_space:
                for rule in self.projected_value_space[attr]:
                    print(attr, rule, len(
                        self.projected_value_space[attr][rule]))

    def generate_value_space(self, attr):
        '''
        generate space for a specific attr
        '''
        # col1_value_range = list(
        #     range(1, 1 + len(self.config[self.task]['value'][attr])))
        # col2_value_range = list(
        #     range(1, 1 + len(self.config[self.task]['value'][attr])))
        # col3_value_range = list(
        #     range(1, 1 + len(self.config[self.task]['value'][attr])))
        col1_value_range = copy.deepcopy(self.config[self.task]['value'][attr])
        col2_value_range = copy.deepcopy(self.config[self.task]['value'][attr])
        col3_value_range = copy.deepcopy(self.config[self.task]['value'][attr])
        space = list(
            product(*[col1_value_range, col2_value_range, col3_value_range]))

        return {attr: space}

    def project_value_space_to_rule(self, attr):
        attr_value_space = self.value_space[attr]
        rule_space = self.config[self.task]['rule'][attr]

        result = {attr: {}}
        bitarray = [False] * len(attr_value_space)
        # store index, find all correct point
        for r in rule_space:
            rule_obj = create_rule(*((attr,) + r))
            points = []
            for i, v in enumerate(attr_value_space):
                if rule_obj.check(*v):
                    bitarray[i] = True
                    points.append(v)
            result[attr].update({rule_obj.to_dict_key(): points})

        illegal = [self.value_space[attr][i]
                   for i in range(len(attr_value_space)) if bitarray[i] == False]
        # result[attr].update({'illegal': illegal})

        # check begin
        total = 0
        for k in result[attr]:
            total += len(result[attr][k])
        total += len(illegal)
        # print(total, len(attr_value_space))
        assert total == len(attr_value_space)
        # check end

        return result

    def find_prefix(self, attr, a, b):
        # find possible c that exist a rule satisified (a, b, c)
        possible = []
        # print(self.projected_value_space[attr])
        for rule_key in self.projected_value_space[attr]:
            for item in self.projected_value_space[attr][rule_key]:
                # print(item)
                if item[0] == a and item[1] == b:
                    possible_c = item[2]
                    possible_rule = deserialize_rule_key(rule_key)
                    possible.append((possible_c, possible_rule))
                    break

        return possible

    def generate_rule_candidate(self, row_3, answer):
        tmp = []
        prefix = list(zip(*row_3))
        # 4 dim attr
        for (attr, (a, b)) in zip(self.config[self.task]['attrs'], prefix):
            possible = self.find_prefix(attr, a, b)
            tmp.append(possible)

        possible_space = list(product(*tmp))
        possible_space_wo_correct = []
        for item in possible_space:
            possible_answer = tuple([i[0] for i in item])
            if possible_answer != answer:
                possible_space_wo_correct.append(item)

        assert len(possible_space) == len(possible_space_wo_correct) + 1

        if len(possible_space_wo_correct) > 7:
            possible_space_wo_correct = random.sample(
                possible_space_wo_correct, 7)

        result = []
        for item in possible_space_wo_correct:
            result.append((tuple([i[0] for i in item]), [i[1] for i in item]))
        return result

    def solve(self, row_3, rule_group, candidates):
        pred_answer = rule_group.apply(*row_3)
        pred = -1
        num = 0
        for i, c in enumerate(candidates):
            if c == pred_answer:
                pred = i
                num += 1
        assert num == 1 and pred != -1  # only one answer
        return pred

    def random_modify_attr(self, panel, attr_index):
        # panel: tuple
        result = list(panel)
        attr = self.core_config[self.task]['attrs'][attr_index]
        value_range = list(
            range(1, 1 + len(self.config[self.task]['value'][attr])))
        new_value = random.choice(value_range)
        while new_value == panel[attr_index]:
            new_value = random.choice(value_range)
        result[attr_index] = new_value

        return tuple(result)

    def generate_attr_candidate(self, answer):
        cur_list = [answer]
        levels = 3  # 2^3 = 8
        modified_attrs_indices = random.sample(
            list(range(len(self.core_config[self.task]['attrs']))), k=levels)
        for l, attr_index in enumerate(modified_attrs_indices):
            tmp_list = []
            for item in cur_list:
                # to left
                tmp_list.append(item)
                # to right
                tmp_list.append(self.random_modify_attr(item, attr_index))
            cur_list = copy.deepcopy(tmp_list)
        assert answer == cur_list[0]
        return cur_list

    def generate_symbol_wo_rule_based(self, rule_group=None):
        # first sample rule
        if rule_group is None:
            rule_group = []
            for attr in self.config[self.task]['attrs']:
                rule_space = self.config[self.task]['rule'][attr]
                rule_group.append((attr, ) + random.choice(rule_space))

        rule_group = RuleGroup(rule_group)
        data = {}
        # sample value space
        for rule in rule_group:
            attr = rule.attr
            rule_key = rule.to_dict_key()
            data[attr] = []
            for _ in range(3):
                data[attr] += list(random.choice(self.projected_value_space[attr][rule_key]))

        # generate symbol
        symbol = list(zip(*[v for _, v in data.items()]))

        row_12 = symbol[0:6]
        row_3 = symbol[6:8]
        answer = symbol[8]

        # generate candidates
        context = row_12 + row_3

        tmp_candidates = self.generate_attr_candidate(answer)

        label = 0

        shuffle_indices = list(range(len(tmp_candidates)))
        random.shuffle(shuffle_indices)

        candidates = []
        for j in shuffle_indices:
            candidates.append(tmp_candidates[j])

        label = shuffle_indices.index(label)

        pred = self.solve(row_3, rule_group, candidates)

        return {
            "context": context,
            "candidates": candidates,
            "label": label,
            "pred": pred,
            "acc": 1 if pred == label else 0,
            'rules': rule_group.to_tuple()
        }

    def generate_symbol(self, rule_group=None):
        # first sample rule
        if rule_group is None:
            rule_group = []
            for attr in self.config[self.task]['attrs']:
                rule_space = self.config[self.task]['rule'][attr]
                rule_group.append((attr, ) + random.choice(rule_space))

        rule_group = RuleGroup(rule_group)
        data = {}
        # sample value space
        for rule in rule_group:
            attr = rule.attr
            rule_key = rule.to_dict_key()
            data[attr] = []
            for _ in range(3):
                data[attr] += list(random.choice(self.projected_value_space[attr][rule_key]))

        # generate symbol
        symbol = list(zip(*[v for _, v in data.items()]))

        row_12 = symbol[0:6]
        row_3 = symbol[6:8]
        answer = symbol[8]

        # generate candidates
        context = row_12 + row_3

        tmp_candidates = self.generate_rule_candidate(row_3, answer)

        tmp_candidates = tmp_candidates + [(answer, rule_group.to_tuple())]

        label = len(tmp_candidates) - 1

        shuffle_indices = list(range(len(tmp_candidates)))
        random.shuffle(shuffle_indices)

        candidates = []
        candidate_rules = []
        for j in shuffle_indices:
            candidates.append(tmp_candidates[j][0])
            candidate_rules.append(tmp_candidates[j][1])

        label = shuffle_indices.index(label)

        pred = self.solve(row_3, rule_group, candidates)

        return {
            "context": context,
            "flag": 1 if len(candidates) == 8 else 0,
            "candidates": candidates,
            "candidate_rules": candidate_rules,
            "label": label,
            "pred": pred,
            "acc": 1 if pred == label else 0,
        }

    def generate_warmup(self):
        print("Generating warm up data...")

        os.makedirs(self.data_dir, exist_ok=True)
        for mode in ["train", "test", "validation"]:
            filename = "%s_visual.pkl" % mode
            data = []
            factor = 0.6 if mode == "train" else 0.2
            print("%s set: %d samples" % (mode, int(self.samples * factor)))
            while len(data) < int(self.samples * factor):
                d = self.generate_symbol()
                data.append({
                    "label": d["label"],
                    "symbol": np.expand_dims(np.array(d["context"] + d["candidates"] + (8 - len(d["candidates"])) * [(0, 0, 0, 0)]), 1),
                    "rules": d["candidate_rules"][d["label"]],
                    "candidate_rules": d["candidate_rules"]
                })

            with open(os.path.join(self.data_dir, filename), "wb") as f:
                pickle.dump(data, f)
        print("Warm up data generation is done!\n")

    def hash_data_sample(self, d):
        context = d['context']
        answer = d['candidates'][d['label']]
        merge = context + [answer]
        h = str(merge)
        return h

    def generate_rule_vs_attr(self):
        print("Generating ablation data rule_vs_attr with (train_rule, valid_rule) and (train_attr, valid_attr)...")

        for t in ["rule", "attr"]:
            os.makedirs(os.path.join(self.data_dir, t), exist_ok=True)
            d = self.generate_symbol()
            rule_group_space = list(product(*[[(attr, ) + item for item in self.core_config[self.task]
                                    ['rule'][attr]] for attr in self.core_config[self.task]['attrs']]))

            data_set = set()
            data = []
            for rg in rule_group_space:
                cnt = 0
                while cnt < self.samples_per_rule:
                    if t == "rule":
                        d = self.generate_symbol(rule_group=rg)
                        hash_d = self.hash_data_sample(d)
                        if d["flag"] == 1 and hash_d not in data_set:
                            data.append({
                                "label": d["label"],
                                "symbol": np.expand_dims(np.array(d["context"] + d["candidates"]), 1),
                                "rules": d["candidate_rules"][d["label"]],
                                "candidate_rules": d["candidate_rules"]
                            })
                            cnt += 1
                            data_set.add(hash_d)
                    else:
                        d = self.generate_symbol_wo_rule_based(rule_group=rg)
                        hash_d = self.hash_data_sample(d)
                        if hash_d not in data_set:
                            data.append({
                                "label": d["label"],
                                "symbol": np.expand_dims(np.array(d["context"] + d["candidates"]), 1),
                                "rules": d["rules"],
                            })
                            cnt += 1
                            data_set.add(hash_d)
            assert len(data) == self.samples_per_rule * len(rule_group_space)
            # mean style
            #
            train_num = self.samples_per_rule - self.test_samples_per_rule
            train_set = []
            valid_set = []
            for i in range(len(rule_group_space)):
                train_set += data[i * self.samples_per_rule: i *
                                self.samples_per_rule + train_num]
                valid_set += data[i * self.samples_per_rule +
                                train_num: (i + 1) * self.samples_per_rule]
            random.shuffle(train_set)
            random.shuffle(valid_set)
            data_split = {
                'train': train_set,
                'validation': valid_set
            }
            for mode in ["train", "validation"]:
                filename = "%s_visual.pkl" % mode
                print("%s set: %d samples" % (mode, len(data_split[mode])))
                with open(os.path.join(self.data_dir, t, filename), "wb") as f:
                    pickle.dump(data_split[mode], f)
            for mode in ["test"]:
                filename = "%s_visual.pkl" % mode
                print("%s set: %d samples" % (mode, len(valid_set)))
                with open(os.path.join(self.data_dir, t, filename), "wb") as f:
                    pickle.dump(valid_set, f)

        print("Ablation data rule_vs_attr generation is done!\n")

    def generate_iid(self):
        print("Generating IID data with train, valid(test)...")

        os.makedirs(self.data_dir, exist_ok=True)
        d = self.generate_symbol()
        rule_group_space = list(product(*[[(attr, ) + item for item in self.core_config[self.task]
                                ['rule'][attr]] for attr in self.core_config[self.task]['attrs']]))

        data_set = set()
        data = []
        for rg in rule_group_space:
            cnt = 0
            while cnt < self.samples_per_rule:
                d = self.generate_symbol(rule_group=rg)
                hash_d = self.hash_data_sample(d)
                if d["flag"] == 1 and hash_d not in data_set:
                    data.append({
                        "label": d["label"],
                        "symbol": np.expand_dims(np.array(d["context"] + d["candidates"]), 1),
                        "rules": d["candidate_rules"][d["label"]],
                        "candidate_rules": d["candidate_rules"]
                    })
                    cnt += 1
                    data_set.add(hash_d)
        assert len(data) == self.samples_per_rule * len(rule_group_space)
        # mean style
        #
        train_num = self.samples_per_rule - self.test_samples_per_rule
        train_set = []
        valid_set = []
        for i in range(len(rule_group_space)):
            train_set += data[i * self.samples_per_rule: i *
                              self.samples_per_rule + train_num]
            valid_set += data[i * self.samples_per_rule +
                              train_num: (i + 1) * self.samples_per_rule]
        random.shuffle(train_set)
        random.shuffle(valid_set)
        data_split = {
            'train': train_set,
            'validation': valid_set
        }
        for mode in ["train", "validation"]:
            filename = "%s_visual.pkl" % mode
            print("%s set: %d samples" % (mode, len(data_split[mode])))
            with open(os.path.join(self.data_dir, filename), "wb") as f:
                pickle.dump(data_split[mode], f)

        print("IID data generation is done!\n")

    def generate_iid_inpo(self):
        print("Generating IID with interpolate data, train, valid, and test.")
        INPO_RULES = 300

        os.makedirs(self.data_dir, exist_ok=True)
        d = self.generate_symbol()
        rule_group_space = list(product(*[[(attr, ) + item for item in self.core_config[self.task]
                                ['rule'][attr]] for attr in self.core_config[self.task]['attrs']]))
        random.shuffle(rule_group_space)

        inpo_rule_group_space = random.sample(rule_group_space, k=INPO_RULES)
        rule_group_space = list(
            set(rule_group_space) - set(inpo_rule_group_space))

        data_set = set()
        data = []
        ##### iid: train and validation
        for rg in rule_group_space:
            cnt = 0
            while cnt < self.samples_per_rule:
                d = self.generate_symbol(rule_group=rg)
                hash_d = self.hash_data_sample(d)
                if d["flag"] == 1 and hash_d not in data_set:
                    data.append({
                        "label": d["label"],
                        "symbol": np.expand_dims(np.array(d["context"] + d["candidates"]), 1),
                        "rules": d["candidate_rules"][d["label"]],
                        "candidate_rules": d["candidate_rules"]
                    })
                    cnt += 1
                    data_set.add(hash_d)

        assert len(data) == self.samples_per_rule * len(rule_group_space)
        # mean style
        #
        train_num = self.samples_per_rule - self.test_samples_per_rule
        train = []
        valid = []
        for i in range(len(rule_group_space)):
            train += data[i * self.samples_per_rule: i *
                          self.samples_per_rule + train_num]
            valid += data[i * self.samples_per_rule +
                          train_num: (i + 1) * self.samples_per_rule]
        random.shuffle(train)
        random.shuffle(valid)
        ##### iid: train and validation

        ##### ood_inpo: test
        test = []
        for rg in inpo_rule_group_space:
            cnt = 0
            while cnt < self.test_samples_per_rule:
                d = self.generate_symbol(rule_group=rg)
                hash_d = self.hash_data_sample(d)
                if d["flag"] == 1 and hash_d not in data_set:
                    test.append({
                        "label": d["label"],
                        "symbol": np.expand_dims(np.array(d["context"] + d["candidates"]), 1),
                        "rules": d["candidate_rules"][d["label"]],
                        "candidate_rules": d["candidate_rules"]
                    })
                    cnt += 1
                    data_set.add(hash_d)

        assert len(test) == self.test_samples_per_rule * \
            len(inpo_rule_group_space)
        random.shuffle(test)
        ##### ood_inpo: test

        data_split = {
            'train': train,
            'validation': valid,
            'test': test
        }
        for mode in ["train", "validation", "test"]:
            filename = "%s_visual.pkl" % mode
            print("%s set: %d samples" % (mode, len(data_split[mode])))
            with open(os.path.join(self.data_dir, filename), "wb") as f:
                pickle.dump(data_split[mode], f)
        print("IID and OOD interpolation data generation is done!\n")

    def generate_ood_expo_l1(self):
        print("Generating OOD Extrapolation Level 1 data...")

        os.makedirs(self.data_dir, exist_ok=True)
        d = self.generate_symbol()
        rule_group_space = []
        unique_rules = [[(attr, ) + r for r in self.core_config[self.task]['rule_unique'][attr]]
                        for attr in self.core_config[self.task]['attrs']]
        full_rules = [[(attr, ) + r for r in self.core_config[self.task]['rule_full'][attr]]
                      for attr in self.core_config[self.task]['attrs']]

        part_rules = [list(set(y) - set(x))
                      for x, y in zip(unique_rules, full_rules)]
        rule_group_space = list(
            set(product(*full_rules)) - set(product(*part_rules)))

        data_set = set()
        data = []
        for rg in rule_group_space:
            cnt = 0
            while cnt < self.test_samples_per_rule:
                d = self.generate_symbol(rule_group=rg)
                hash_d = self.hash_data_sample(d)
                if d["flag"] == 1 and hash_d not in data_set:
                    data.append({
                        "label": d["label"],
                        "symbol": np.expand_dims(np.array(d["context"] + d["candidates"]), 1),
                        "rules": d["candidate_rules"][d["label"]],
                        "candidate_rules": d["candidate_rules"]
                    })
                    cnt += 1
                    data_set.add(hash_d)
        assert len(data) == self.test_samples_per_rule * len(rule_group_space)
        random.shuffle(data)
        data_split = {
            'test': data
        }
        for mode in ["test"]:
            filename = "%s_visual.pkl" % mode
            print("%s set: %d samples" % (mode, len(data_split[mode])))
            with open(os.path.join(self.data_dir, filename), "wb") as f:
                pickle.dump(data_split[mode], f)
        print("OOD Extrapolation level 1 data generation is done!\n")

    def generate_ood_expo_l2(self):
        print("Generating OOD Extrapolation Level 2 data...")

        os.makedirs(self.data_dir, exist_ok=True)
        d = self.generate_symbol()
        rule_group_space = list(product(*[[(attr, ) + item for item in self.core_config[self.task]
                                ['rule'][attr]] for attr in self.core_config[self.task]['attrs']]))

        data_set = set()
        data = []
        for rg in rule_group_space:
            cnt = 0
            while cnt < self.test_samples_per_rule:
                d = self.generate_symbol(rule_group=rg)
                hash_d = self.hash_data_sample(d)
                if d["flag"] == 1 and hash_d not in data_set:
                    data.append({
                        "label": d["label"],
                        "symbol": np.expand_dims(np.array(d["context"] + d["candidates"]), 1),
                        "rules": d["candidate_rules"][d["label"]],
                        "candidate_rules": d["candidate_rules"]
                    })
                    cnt += 1
                    data_set.add(hash_d)
        assert len(data) == self.test_samples_per_rule * len(rule_group_space)
        random.shuffle(data)
        data_split = {
            'test': data
        }
        for mode in ["test"]:
            filename = "%s_visual.pkl" % mode
            print("%s set: %d samples" % (mode, len(data_split[mode])))
            with open(os.path.join(self.data_dir, filename), "wb") as f:
                pickle.dump(data_split[mode], f)
        print("OOD extrapolation level 2 data generation is done!\n")

    def generate_pkl(self):
        if self.mode == 'warmup':
            self.generate_warmup()
        elif self.mode == 'iid':
            self.generate_iid()
        elif self.mode == 'iid_inpo':
            self.generate_iid_inpo()
        elif self.mode == 'ood_expo_l2':
            self.generate_ood_expo_l2()
        elif self.mode == 'ood_expo_l1':
            self.generate_ood_expo_l1()
        elif self.mode == 'rule_vs_attr':
            # ablation
            self.generate_rule_vs_attr()


if __name__ == "__main__":
    import const_expo_l1_20 as const
    a = {
        'mode': 'iid',
        'config': const.config_iid,
        'core_config': const.config_iid_core,
        'samples_per_rule': 20,
        'test_samples_per_rule': 10,
        'data_dir': os.path.join("./data", "iid")
    }
    t = Task('center_single', **a)
    res = t.generate_attr_candidate((2, 3, 4, 5))
    print(res)
    pass
