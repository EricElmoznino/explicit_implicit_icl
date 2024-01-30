config = {
    'center_single': {
        'attrs': ['number', 'type', 'size', 'color'],
        # 'value': {
        #     'number': [1],
        #     'type': [1, 2, 3, 4, 5],
        #     'size': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        #     'color': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        # },
        'value': {
            'number': list(range(1, 20)),
            'type': list(range(1, 20)),
            'size': list(range(1, 20)),
            'color': list(range(1, 20)),
        },
        'rule': {
            'number': [
                ('constant', 0), # constant
                ('progression', -2), # -2, -2
                ('progression', -1), # -1, -1
                ('progression', 1), # +1, +1
                ('progression', 2), # +2, +2
                ('arithmetic', -1), # plus
                ('arithmetic', 1), # minus
                ('comparison', -1), # min
                ('comparison', 1), # max
                ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
            'type': [
                ('constant', 0), 
                ('progression', -2), 
                ('progression', -1), 
                ('progression', 1), 
                ('progression', 2),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
            'size': [
                ('constant', 0), 
                ('progression', -2), 
                ('progression', -1), 
                ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
            'color': [
                ('constant', 0), 
                ('progression', -2), 
                ('progression', -1), 
                ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
        }
    }
}

config_warmup = {
    'center_single': {
        'attrs': ['number', 'type', 'size', 'color'],
        # 'value': {
        #     'number': [1],
        #     'type': [1, 2, 3, 4, 5],
        #     'size': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        #     'color': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        # },
        'value': {
            'number': list(range(1, 20)),
            'type': list(range(1, 20)),
            'size': list(range(1, 20)),
            'color': list(range(1, 20)),
        },
        'rule': {
            'number': [
                ('progression', 1),
                ('progression', -1),
                ('varprogression', 1),
                ('progression', -2),
                ('varprogression', -2),
            ],
            'type': [
                ('progression', 1),
                ('progression', -1),
                ('varprogression', 1),
                ('progression', -2),
                ('varprogression', -2),
            ],
            'size': [
                ('progression', 1),
                ('progression', -1),
                ('varprogression', 1),
                ('progression', -2),
                ('varprogression', -2),
            ],
            'color': [
                ('progression', 1),
                ('progression', -1),
                ('varprogression', 1),
                ('progression', -2),
                ('varprogression', -2),
            ],
        }
    }
}

config_iid_core = {
    'center_single': {
        'attrs': ['number', 'type', 'size', 'color'],
        # 'value': {
        #     'number': [1],
        #     'type': [1, 2, 3, 4, 5],
        #     'size': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        #     'color': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        # },
        'value': {
            'number': list(range(1, 20)),
            'type': list(range(1, 20)),
            'size': list(range(1, 20)),
            'color': list(range(1, 20)),
        },
        'rule': {
            'number': [
                ('constant', 0), 
                # ('progression', -2), 
                # ('progression', -1), 
                # ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                # ('varprogression', 1), # +1, +2
                # ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                # ('varprogression', -2), # -2, -1
            ],
            'type': [
                ('constant', 0), 
                # ('progression', -2), 
                # ('progression', -1), 
                # ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                # ('varprogression', 1), # +1, +2
                # ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                # ('varprogression', -2), # -2, -1
            ],
            'size': [
                ('constant', 0), 
                # ('progression', -2), 
                # ('progression', -1), 
                # ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                # ('varprogression', 1), # +1, +2
                # ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                # ('varprogression', -2), # -2, -1
            ],
            'color': [
                ('constant', 0), 
                # ('progression', -2), 
                # ('progression', -1), 
                # ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                # ('varprogression', 1), # +1, +2
                # ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                # ('varprogression', -2), # -2, -1
            ],
        }
    }
}

config_iid = {
    'center_single': {
        'attrs': ['number', 'type', 'size', 'color'],
        # 'value': {
        #     'number': [1],
        #     'type': [1, 2, 3, 4, 5],
        #     'size': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        #     'color': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        # },
        'value': {
            'number': list(range(1, 20)),
            'type': list(range(1, 20)),
            'size': list(range(1, 20)),
            'color': list(range(1, 20)),
        },
        'rule': {
            'number': [
                ('constant', 0), 
                ('progression', -2), 
                ('progression', -1), 
                ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
            'type': [
                ('constant', 0), 
                ('progression', -2), 
                ('progression', -1), 
                ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
            'size': [
                ('constant', 0), 
                ('progression', -2), 
                ('progression', -1), 
                ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
            'color': [
                ('constant', 0), 
                ('progression', -2), 
                ('progression', -1), 
                ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
        }
    }
}

config_iid_inpo_core = {
    'center_single': {
        'attrs': ['number', 'type', 'size', 'color'],
        # 'value': {
        #     'number': [1],
        #     'type': [1, 2, 3, 4, 5],
        #     'size': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        #     'color': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        # },
        'value': {
            'number': list(range(1, 20)),
            'type': list(range(1, 20)),
            'size': list(range(1, 20)),
            'color': list(range(1, 20)),
        },
        'rule': {
            'number': [
                ('constant', 0), 
                # ('progression', -2), 
                # ('progression', -1), 
                # ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                # ('varprogression', 1), # +1, +2
                # ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                # ('varprogression', -2), # -2, -1
            ],
            'type': [
                ('constant', 0), 
                # ('progression', -2), 
                # ('progression', -1), 
                # ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                # ('varprogression', 1), # +1, +2
                # ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                # ('varprogression', -2), # -2, -1
            ],
            'size': [
                ('constant', 0), 
                # ('progression', -2), 
                # ('progression', -1), 
                # ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                # ('varprogression', 1), # +1, +2
                # ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                # ('varprogression', -2), # -2, -1
            ],
            'color': [
                ('constant', 0), 
                # ('progression', -2), 
                # ('progression', -1), 
                # ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                # ('varprogression', 1), # +1, +2
                # ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                # ('varprogression', -2), # -2, -1
            ],
        }
    }
}

config_iid_inpo = {
    'center_single': {
        'attrs': ['number', 'type', 'size', 'color'],
        # 'value': {
        #     'number': [1],
        #     'type': [1, 2, 3, 4, 5],
        #     'size': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        #     'color': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        # },
        'value': {
            'number': list(range(1, 20)),
            'type': list(range(1, 20)),
            'size': list(range(1, 20)),
            'color': list(range(1, 20)),
        },
        'rule': {
            'number': [
                ('constant', 0), 
                ('progression', -2), 
                ('progression', -1), 
                ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
            'type': [
                ('constant', 0), 
                ('progression', -2), 
                ('progression', -1), 
                ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
            'size': [
                ('constant', 0), 
                ('progression', -2), 
                ('progression', -1), 
                ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
            'color': [
                ('constant', 0), 
                ('progression', -2), 
                ('progression', -1), 
                ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
        }
    }
}

config_ood_expo_l2_core = {
    'center_single': {
        'attrs': ['number', 'type', 'size', 'color'],
        # 'value': {
        #     'number': [1],
        #     'type': [1, 2, 3, 4, 5],
        #     'size': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        #     'color': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        # },
        'value': {
            'number': list(range(1, 20)),
            'type': list(range(1, 20)),
            'size': list(range(1, 20)),
            'color': list(range(1, 20)),
        },
        'rule': {
            'number': [
                ('constant', 0), 
                # ('progression', -2), 
                # ('progression', -1), 
                # ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                # ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                # ('varprogression', -2), # -2, -1
            ],
            'type': [
                # ('constant', 0), 
                ('progression', -2), 
                # ('progression', -1), 
                # ('progression', 1), 
                # ('progression', 2),
                # ('arithmetic', -1),
                # ('arithmetic', 1),
                # ('comparison', -1), # MIN
                # ('comparison', 1), # MAX
                # ('varprogression', 1), # +1, +2
                # ('varprogression', 2), # +2, +1
                # ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
            'size': [
                # ('constant', 0), 
                ('progression', -2), 
                # ('progression', -1), 
                # ('progression', 1), 
                # ('progression', 2),
                # ('arithmetic', -1),
                # ('arithmetic', 1),
                # ('comparison', -1), # MIN
                # ('comparison', 1), # MAX
                # ('varprogression', 1), # +1, +2
                # ('varprogression', 2), # +2, +1
                # ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
            'color': [
                # ('constant', 0), 
                ('progression', -2), 
                # ('progression', -1), 
                # ('progression', 1), 
                # ('progression', 2),
                # ('arithmetic', -1),
                # ('arithmetic', 1),
                # ('comparison', -1), # MIN
                # ('comparison', 1), # MAX
                # ('varprogression', 1), # +1, +2
                # ('varprogression', 2), # +2, +1
                # ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
        }
    }
}

config_ood_expo_l2 = {
    'center_single': {
        'attrs': ['number', 'type', 'size', 'color'],
        # 'value': {
        #     'number': [1],
        #     'type': [1, 2, 3, 4, 5],
        #     'size': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        #     'color': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        # },
        'value': {
            'number': list(range(1, 20)),
            'type': list(range(1, 20)),
            'size': list(range(1, 20)),
            'color': list(range(1, 20)),
        },
        'rule': {
            'number': [
                ('constant', 0), 
                ('progression', -2), 
                ('progression', -1), 
                ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
            'type': [
                ('constant', 0), 
                ('progression', -2), 
                ('progression', -1), 
                ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
            'size': [
                ('constant', 0), 
                ('progression', -2), 
                ('progression', -1), 
                ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
            'color': [
                ('constant', 0), 
                ('progression', -2), 
                ('progression', -1), 
                ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
        }
    }
}

config_ood_expo_l1_core = {
    'center_single': {
        'attrs': ['number', 'type', 'size', 'color'],
        # 'value': {
        #     'number': [1],
        #     'type': [1, 2, 3, 4, 5],
        #     'size': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        #     'color': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        # },
        'value': {
            'number': list(range(1, 20)),
            'type': list(range(1, 20)),
            'size': list(range(1, 20)),
            'color': list(range(1, 20)),
        },
        'rule_unique': {
            'number': [
                # ('constant', 0), 
                # ('progression', -2), 
                # ('progression', -1), 
                # ('progression', 1), 
                # ('progression', 2),
                # ('arithmetic', -1),
                # ('arithmetic', 1),
                # ('comparison', -1), # MIN
                # ('comparison', 1), # MAX
                # ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                # ('varprogression', -1), # -1, -2
                # ('varprogression', -2), # -2, -1
            ],
            'type': [
                # ('constant', 0), 
                # ('progression', -2), 
                # ('progression', -1), 
                # ('progression', 1), 
                # ('progression', 2),
                # ('arithmetic', -1),
                # ('arithmetic', 1),
                # ('comparison', -1), # MIN
                # ('comparison', 1), # MAX
                # ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                # ('varprogression', -1), # -1, -2
                # ('varprogression', -2), # -2, -1
            ],
            'size': [
                # ('constant', 0), 
                # ('progression', -2), 
                # ('progression', -1), 
                # ('progression', 1), 
                # ('progression', 2),
                # ('arithmetic', -1),
                # ('arithmetic', 1),
                # ('comparison', -1), # MIN
                # ('comparison', 1), # MAX
                # ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                # ('varprogression', -1), # -1, -2
                # ('varprogression', -2), # -2, -1
            ],
            'color': [
                # ('constant', 0), 
                # ('progression', -2), 
                # ('progression', -1), 
                # ('progression', 1), 
                # ('progression', 2),
                # ('arithmetic', -1),
                # ('arithmetic', 1),
                # ('comparison', -1), # MIN
                # ('comparison', 1), # MAX
                # ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                # ('varprogression', -1), # -1, -2
                # ('varprogression', -2), # -2, -1
            ],
        },
        'rule_full': {
            'number': [
                ('constant', 0), 
                # ('progression', -2), 
                # ('progression', -1), 
                # ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                # ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                # ('varprogression', -2), # -2, -1
            ],
            'type': [
                ('constant', 0), 
                # ('progression', -2), 
                # ('progression', -1), 
                # ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                # ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                # ('varprogression', -2), # -2, -1
            ],
            'size': [
                ('constant', 0), 
                # ('progression', -2), 
                # ('progression', -1), 
                # ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                # ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                # ('varprogression', -2), # -2, -1
            ],
            'color': [
                ('constant', 0), 
                # ('progression', -2), 
                # ('progression', -1), 
                # ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                # ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                # ('varprogression', -2), # -2, -1
            ],
        }
    }
}

config_ood_expo_l1 = {
    'center_single': {
        'attrs': ['number', 'type', 'size', 'color'],
        # 'value': {
        #     'number': [1],
        #     'type': [1, 2, 3, 4, 5],
        #     'size': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        #     'color': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        # },
        'value': {
            'number': list(range(1, 20)),
            'type': list(range(1, 20)),
            'size': list(range(1, 20)),
            'color': list(range(1, 20)),
        },
        'rule': {
            'number': [
                ('constant', 0), 
                ('progression', -2), 
                ('progression', -1), 
                ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
            'type': [
                ('constant', 0), 
                ('progression', -2), 
                ('progression', -1), 
                ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
            'size': [
                ('constant', 0), 
                ('progression', -2), 
                ('progression', -1), 
                ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
            'color': [
                ('constant', 0), 
                ('progression', -2), 
                ('progression', -1), 
                ('progression', 1), 
                ('progression', 2),
                ('arithmetic', -1),
                ('arithmetic', 1),
                ('comparison', -1), # MIN
                ('comparison', 1), # MAX
                ('varprogression', 1), # +1, +2
                ('varprogression', 2), # +2, +1
                ('varprogression', -1), # -1, -2
                ('varprogression', -2), # -2, -1
            ],
        }
    }
}