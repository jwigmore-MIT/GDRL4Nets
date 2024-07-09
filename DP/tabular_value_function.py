from collections import defaultdict

from DP.value_function import ValueFunction


class TabularValueFunction(ValueFunction):
    def __init__(self, q_max = 100, penalty = -100, default=0):
        super().__init__(q_max, penalty)
        self.value_table = defaultdict(lambda: default)


    def update(self, state, value):
        if not isinstance(state, tuple):
            state = tuple(state)
        self.value_table[state] = value

    def merge(self, value_table):
        for state in value_table.value_table.keys():
            self.update(state, value_table.get_value(state))

    def get_value(self, state):
        if not isinstance(state, tuple):
            state = tuple(state)

        return self.value_table[state]

