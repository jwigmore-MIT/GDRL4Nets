from tqdm import tqdm
from DP.qtable import *
from DP.tabular_value_function import *
import numpy as np

class ValueIterationMinus:
    def __init__(self, mdp, values):
        self.mdp = mdp
        self.values = values

    def value_iteration(self, max_iterations=100, theta=0.1):
        pbar = tqdm(range(int(max_iterations)))
        for i in pbar:
            delta = 0.0
            new_values = TabularValueFunction()
            for state in self.mdp.get_states():
                # create an indicator vector for if s[0] and s[1] are equal to q_max
                vec = np.array([s_i == self.mdp.q_max for s_i in state], dtype = int)
                if vec.sum() < 1:
                    qtable = QTable()

                    for action in self.mdp.get_actions(state):
                        # Calculate the value of Q(s,a)
                        new_value = 0.0
                        for (new_state, probability) in self.mdp.get_transitions(
                            state, action
                        ):
                            reward = self.mdp.get_reward(state, action, new_state)
                            new_value += probability * (
                                reward
                                + (
                                    self.mdp.get_discount_factor()
                                    * self.values.get_value(new_state)
                                )
                            )


                        qtable.update(state, action, new_value)

                    # V(s) = max_a Q(sa)
                    (_, max_q) = qtable.get_max_q(state, self.mdp.get_actions(state))
                    delta = max(delta, abs(self.values.get_value(state) - max_q))
                    new_values.update(state, max_q)
                else:
                    closest_state = state - vec
                    est_state_value = new_values.get_value(closest_state) #+ self.mdp.get_discount_factor()*self.mdp.get_reward(state, 0, closest_state)
                    new_values.update(state, est_state_value)

            self.values.merge(new_values)

            pbar.set_description(f"Delta: {delta}")

            # Terminate if the value function has converged
            if delta < theta:
                return i

