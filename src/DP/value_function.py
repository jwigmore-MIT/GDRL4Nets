from DP.tabular_policy import TabularPolicy

class ValueFunction():

    def __init__(self, q_max = 100, penalty = -100):
        self.q_max = q_max
        self.limit = int(q_max+1)
        self.penalty = penalty
    def update(self, state, value):
        pass

    def merge(self, value_table):
        pass

    def get_value(self, state):
        pass

    """ Return the Q-value of action in state """
    def get_q_value(self, mdp, state, action):
        "Need to modify such that if state is not in the value table, we return self.get_value(state)-100"
        q_value = 0.0
        for (new_state, probability) in mdp.get_transitions(state, action):
            reward = mdp.get_reward(state, action, new_state)
            if self.limit in new_state:
                q_value += probability * (reward + (mdp.get_discount_factor() * (self.get_value(state)+self.penalty)))
            else:
                q_value += probability * (
                    reward
                    + (mdp.get_discount_factor() * self.get_value(new_state))
                )

        return q_value

    """ Return a policy from this value function """

    def extract_policy(self, mdp):
        policy = TabularPolicy()
        for state in mdp.get_states():
            # if not isinstance(state, tuple):
            #     state = tuple(state)
            max_q = float("-inf")
            for action in mdp.get_actions(state):
                q_value = self.get_q_value(mdp, state, action)

                # If this is the maximum Q-value so far,
                # set the policy for this state
                if q_value > max_q:
                    policy.update(state, action)
                    max_q = q_value

        return policy
