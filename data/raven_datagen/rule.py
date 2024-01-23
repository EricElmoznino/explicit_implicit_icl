def create_rule(attr, name, value):
    if name == "constant":
        return Constant(attr, name, value)
    elif name == "progression":
        return Progression(attr, name, value)
    elif name == "arithmetic":
        return Arithmetic(attr, name, value)
    elif name == "comparison":
        return Comparison(attr, name, value)
    elif name == "varprogression":
        return VarProgression(attr, name, value)
    else:
        raise ValueError


def deserialize_rule_key(key):
    result = key.split("_")
    result[2] = int(result[2])
    return tuple(result)


def check_constant(a, b, c):
    return a == b == c


def check_progression(value, a, b, c):
    return (b - a) == value and (c - b) == value


def check_arithmetic(value, a, b, c):
    if value == 1:
        return a + b == c
    else:
        return a - b == c


def check_comparison(value, a, b, c):
    if value == 1:
        return max(a, b) == c
    else:
        return min(a, b) == c


def check_varprogression(value, a, b, c):
    if value == 1:
        return (b - a) == 1 and (c - b) == 2
    elif value == 2:
        return (b - a) == 2 and (c - b) == 1
    elif value == -1:
        return (b - a) == -1 and (c - b) == -2
    elif value == -2:
        return (b - a) == -2 and (c - b) == -1


class Rule(object):
    def __init__(self, attr, name, value) -> None:
        self.attr = attr
        self.name = name
        self.value = value

    def to_tuple(self):
        return (self.attr, self.name, self.value)

    def to_dict_key(self):
        return "_".join([str(i) for i in self.to_tuple()])


class Constant(Rule):
    def __init__(self, name, attr, value) -> None:
        super(Constant, self).__init__(name, attr, value)

    def check(self, a, b, c):
        return check_constant(a, b, c)

    def apply(self, a, b):
        return a


class Progression(Rule):
    def __init__(self, name, attr, value) -> None:
        super(Progression, self).__init__(name, attr, value)

    def check(self, a, b, c):
        flag = False
        flag = flag or check_constant(a, b, c)
        for v in [-1, 1]:
            flag = flag or check_arithmetic(v, a, b, c)
        for v in [-1, 1]:
            flag = flag or check_comparison(v, a, b, c)
        for v in [-2, -1, 1, 2]:
            flag = flag or check_varprogression(v, a, b, c)

        if flag:
            # if is other rule
            return False
        else:
            return check_progression(self.value, a, b, c)

    def apply(self, a, b):
        return 2 * b - a


class Arithmetic(Rule):
    def __init__(self, name, attr, value) -> None:
        super(Arithmetic, self).__init__(name, attr, value)

    def check(self, a, b, c):
        flag = False
        flag = flag or check_constant(a, b, c)
        for v in [-2, -1, 1, 2]:
            flag = flag or check_progression(v, a, b, c)
        for v in [-1, 1]:
            flag = flag or check_comparison(v, a, b, c)
        for v in [-2, -1, 1, 2]:
            flag = flag or check_varprogression(v, a, b, c)

        if flag:
            # if is other rule
            return False
        else:
            # filter out zero, a + 0 = a
            if a == 0 or b == 0:
                return False
            else:
                return check_arithmetic(self.value, a, b, c)

    def apply(self, a, b):
        return (a + b) if self.value == 1 else (a - b)


class Comparison(Rule):
    """
    max for 1 and min for -1
    c = max or min of a, b

    filter out ambiguious samples
    """

    def __init__(self, name, attr, value) -> None:
        super(Comparison, self).__init__(name, attr, value)

    def check(self, a, b, c):
        flag = False
        flag = flag or check_constant(a, b, c)
        for v in [-2, -1, 1, 2]:
            flag = flag or check_progression(v, a, b, c)
        for v in [-1, 1]:
            flag = flag or check_arithmetic(v, a, b, c)
        for v in [-2, -1, 1, 2]:
            flag = flag or check_varprogression(v, a, b, c)

        if flag:
            # if is other rule
            return False
        else:
            if a == b:
                return False
            else:
                return check_comparison(self.value, a, b, c)

    def apply(self, a, b):
        return max(a, b) if self.value == 1 else min(a, b)


class VarProgression(Rule):
    """
    1: +1, +2
    2: +2, +1
    -1: -1, -2
    -2: -2, -1
    filter out ambiguious samples
    """

    def __init__(self, name, attr, value) -> None:
        super(VarProgression, self).__init__(name, attr, value)

    def check(self, a, b, c):
        flag = False
        flag = flag or check_constant(a, b, c)
        for v in [-2, -1, 1, 2]:
            flag = flag or check_progression(v, a, b, c)
        for v in [-1, 1]:
            flag = flag or check_arithmetic(v, a, b, c)
        for v in [-1, 1]:
            flag = flag or check_comparison(v, a, b, c)

        if flag:
            # if is other rule
            return False
        else:
            return check_varprogression(self.value, a, b, c)

    def apply(self, a, b):
        if self.value == 1:
            return b + 2
        elif self.value == 2:
            return b + 1
        elif self.value == -1:
            return b - 2
        elif self.value == -2:
            return b - 1


class RuleGroup(object):
    def __init__(self, rules):
        # rules = [('constant', None)]
        self.rules = [create_rule(*r) for r in rules]

    def to_tuple(self):
        return [r.to_tuple() for r in self.rules]

    def __getitem__(self, i):
        return self.rules[i]

    def apply(self, row_3_1, row_3_2):
        result = []
        for a, b, r in zip(row_3_1, row_3_2, self.rules):
            result.append(r.apply(a, b))
        return tuple(result)
