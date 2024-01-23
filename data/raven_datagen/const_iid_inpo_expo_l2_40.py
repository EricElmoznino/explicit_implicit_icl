config_iid_inpo_core = {
    "center_single": {
        "attrs": ["number", "type", "size", "color"],
        "value": {
            "number": list(range(1, 40)),
            "type": list(range(1, 40)),
            "size": list(range(1, 40)),
            "color": list(range(1, 40)),
        },
        "rule": {
            "number": [
                ("constant", 0),
                # ('progression', -2),
                # ('progression', -1),
                # ('progression', 1),
                ("progression", 2),
                ("arithmetic", -1),
                ("arithmetic", 1),
                ("comparison", -1),  # MIN
                ("comparison", 1),  # MAX
                # ('varprogression', 1), # +1, +2
                # ('varprogression', 2), # +2, +1
                ("varprogression", -1),  # -1, -2
                # ('varprogression', -2), # -2, -1
            ],
            "type": [
                ("constant", 0),
                # ('progression', -2),
                # ('progression', -1),
                # ('progression', 1),
                ("progression", 2),
                ("arithmetic", -1),
                ("arithmetic", 1),
                ("comparison", -1),  # MIN
                ("comparison", 1),  # MAX
                # ('varprogression', 1), # +1, +2
                # ('varprogression', 2), # +2, +1
                ("varprogression", -1),  # -1, -2
                # ('varprogression', -2), # -2, -1
            ],
            "size": [
                ("constant", 0),
                # ('progression', -2),
                # ('progression', -1),
                # ('progression', 1),
                ("progression", 2),
                ("arithmetic", -1),
                ("arithmetic", 1),
                ("comparison", -1),  # MIN
                ("comparison", 1),  # MAX
                # ('varprogression', 1), # +1, +2
                # ('varprogression', 2), # +2, +1
                ("varprogression", -1),  # -1, -2
                # ('varprogression', -2), # -2, -1
            ],
            "color": [
                ("constant", 0),
                # ('progression', -2),
                # ('progression', -1),
                # ('progression', 1),
                ("progression", 2),
                ("arithmetic", -1),
                ("arithmetic", 1),
                ("comparison", -1),  # MIN
                ("comparison", 1),  # MAX
                # ('varprogression', 1), # +1, +2
                # ('varprogression', 2), # +2, +1
                ("varprogression", -1),  # -1, -2
                # ('varprogression', -2), # -2, -1
            ],
        },
    }
}

config_iid_inpo = {
    "center_single": {
        "attrs": ["number", "type", "size", "color"],
        "value": {
            "number": list(range(1, 40)),
            "type": list(range(1, 40)),
            "size": list(range(1, 40)),
            "color": list(range(1, 40)),
        },
        "rule": {
            "number": [
                ("constant", 0),
                ("progression", -2),
                ("progression", -1),
                ("progression", 1),
                ("progression", 2),
                ("arithmetic", -1),
                ("arithmetic", 1),
                ("comparison", -1),  # MIN
                ("comparison", 1),  # MAX
                ("varprogression", 1),  # +1, +2
                ("varprogression", 2),  # +2, +1
                ("varprogression", -1),  # -1, -2
                ("varprogression", -2),  # -2, -1
            ],
            "type": [
                ("constant", 0),
                ("progression", -2),
                ("progression", -1),
                ("progression", 1),
                ("progression", 2),
                ("arithmetic", -1),
                ("arithmetic", 1),
                ("comparison", -1),  # MIN
                ("comparison", 1),  # MAX
                ("varprogression", 1),  # +1, +2
                ("varprogression", 2),  # +2, +1
                ("varprogression", -1),  # -1, -2
                ("varprogression", -2),  # -2, -1
            ],
            "size": [
                ("constant", 0),
                ("progression", -2),
                ("progression", -1),
                ("progression", 1),
                ("progression", 2),
                ("arithmetic", -1),
                ("arithmetic", 1),
                ("comparison", -1),  # MIN
                ("comparison", 1),  # MAX
                ("varprogression", 1),  # +1, +2
                ("varprogression", 2),  # +2, +1
                ("varprogression", -1),  # -1, -2
                ("varprogression", -2),  # -2, -1
            ],
            "color": [
                ("constant", 0),
                ("progression", -2),
                ("progression", -1),
                ("progression", 1),
                ("progression", 2),
                ("arithmetic", -1),
                ("arithmetic", 1),
                ("comparison", -1),  # MIN
                ("comparison", 1),  # MAX
                ("varprogression", 1),  # +1, +2
                ("varprogression", 2),  # +2, +1
                ("varprogression", -1),  # -1, -2
                ("varprogression", -2),  # -2, -1
            ],
        },
    }
}
