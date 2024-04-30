from DP.tabular_value_function import TabularValueFunction
from DP.qtable import QTable
from tqdm import tqdm

class PolicyIteration:
    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy

    def policy_evaluation(self, policy, values, theta=0.001, max_iterations=100):
        pbar = tqdm(range(int(max_iterations)))
        for i in pbar:
            delta = 0.0
            new_values = TabularValueFunction()
            for state in self.mdp.get_states():
                # Calculate the value of V(s)
                old_value = values.get_value(state)
                new_value = values.get_q_value(self.mdp, state, policy.select_action(state))
                values.update(state, new_value)
                delta = max(delta, abs(old_value - new_value))
                pbar.update_description(f"Policy Evaluation Phase: Delta: {delta}")

            # terminate if the value function has converged
            if delta < theta:
                break

        return values, delta

    """ Implmentation of policy iteration iteration. Returns the number of iterations exected """

    def policy_iteration(self, max_iterations=100, theta=0.001):

        # create a value function to hold details
        values = TabularValueFunction()
        pbar = tqdm(range(int(max_iterations)))

        for i in pbar:
            policy_changed = False
            values, delta = self.policy_evaluation(self.policy, values, theta)
            for state in self.mdp.get_states():
                old_action = self.policy.select_action(state)

                q_values = QTable()
                for action in self.mdp.get_actions(state):
                    # Calculate the value of Q(s,a)
                    new_value = values.get_q_value(self.mdp, state, action)
                    q_values.update(state, action, new_value)

                # V(s) = argmax_a Q(s,a)
                (new_action, _) = q_values.get_max_q(state, self.mdp.get_actions(state))
                self.policy.update(state, new_action)

                policy_changed = (
                    True if new_action is not old_action else policy_changed
                )
            pbar.set_description(f"Delta: {delta}")

            if not policy_changed:
                return i

        return max_iterations
