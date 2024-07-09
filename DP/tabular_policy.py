from collections import defaultdict

from DP.policy import DeterministicPolicy


def max_weight_policy(q_state, y_state):
    """

    :param q_state:
    :param y_state:
    :return:
    """
    action = []

class TabularPolicy(DeterministicPolicy):
    def __init__(self, policy_table = None, default_action=None):
        if policy_table is None:
            self.policy_table = defaultdict(lambda: default_action)
        else:
            self.policy_table = policy_table

    def select_action(self, state):
        if not isinstance(state, tuple):
            state = tuple(state)
        return self.policy_table[state]

    def update(self, state, action):
        if not isinstance(state, tuple):
            state = tuple(state)
        self.policy_table[state] = action
