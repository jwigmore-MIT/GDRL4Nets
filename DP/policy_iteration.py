from DP.tabular_value_function import TabularValueFunction
from DP.qtable import QTable
from tqdm import tqdm
import numpy as np

class PolicyIteration:
    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy

    def policy_evaluation(self, policy, values, theta=0.1, max_iterations=1000):
        pbar = tqdm(range(int(max_iterations)))
        for i in pbar:
            delta = 0.0
            new_values = TabularValueFunction(default=-10_000)
            for state in self.mdp.get_states():
                # Calculate the value of V(s)
                old_value = values.get_value(state)
                new_value = values.get_q_value(self.mdp, state, policy.select_action(state))
                values.update(state, new_value)
                delta = max(delta, abs(old_value - new_value))
            pbar.set_description(f"Policy Evaluation Phase: Delta: {delta}")

            # terminate if the value function has converged
            if delta < theta:
                break

        return values, delta

    """ Implmentation of policy iteration iteration. Returns the number of iterations exected """

    def policy_iteration(self, max_iterations=100, theta=0.001):

        # create a value function to hold details
        values = TabularValueFunction(q_max=self.mdp.q_max, penalty=-500)
        pbar = tqdm(range(int(max_iterations)))

        plot_policy_table(self.policy.policy_table, lim = self.mdp.q_max)
        for i in pbar:
            policy_changed = False
            policy_changes = []
            theta = theta/2
            values, delta = self.policy_evaluation(self.policy, values, theta)
            for state in self.mdp.get_states():
                old_action = self.policy.select_action(state)


                q_values = QTable()
                for action in self.mdp.get_actions(state):
                    # Calculate the value of Q(s,a)
                    new_value = values.get_q_value(self.mdp, state, action)
                    q_values.update(state, action, new_value)

                # V(s) = argmax_a Q(s,a)
                if self.mdp.q_max in state and (np.array(state) > 0).all():
                    # return the action with the max queue size
                    new_action = np.array([False, True, False]) if state[0] > state[1] else np.array([False, False, True])
                else:
                    (new_action, _) = q_values.get_max_q(state, self.mdp.get_actions(state))
                self.policy.update(state, new_action)
                state_policy_change = sum(new_action*old_action)
                if state_policy_change == 0:
                    policy_changes.append((state, old_action, new_action))
                    policy_changed = True
            pbar.set_description(f"Delta: {delta}")
            plot_policy_table(self.policy.policy_table, iteration=i, lim=self.mdp.q_max)

            if not policy_changed:
                return i
            else:
                pass
        return max_iterations


def plot_policy_table(policy_table, iteration = None, lim = 30):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from experiments.experiment18.analysis_functions import plot_state_action_map
    """
    Plot the policy table for the states Y1=1 and Y2=1
    policy_table has keys ["Q1", ..., "QK", "Y1", ..., "YK"] and value "action"
    First create a dataframe from the policy table with columns Q1,..., QK, Y1, ..., YK action
    Then filter the dataframe to only include rows where Y1=1 and Y2=1
    Then plot the action column as a heatmap
    """
    df = pd.DataFrame(policy_table.keys(), columns = ["Q1", "Q2", "Y1", "Y2"])
    df["Action"] = list(policy_table.values())
    df["Action"] = df["Action"].apply(lambda x: np.argmax(x))
    df = df[(df["Y1"] == 1) & (df["Y2"] == 1)]
    fig, ax = plt.subplots(1,1, figsize = (10,10))
    plot_state_action_map(df, [("Y1", 1), ("Y2", 1)], ax = ax, axis_keys = ["Q1", "Q2"], policy_type = "PI", plot_type = "Action", collected_frames = iteration, lim = lim)
    plt.show()
    return fig, ax


def plot_q_table(mdp, values):

    # For each state with Y1=1 and Y2=1, plot the Q-values for actions [False, True, False] and [False, False, True]
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from experiments.experiment18.analysis_functions import plot_state_action_map

    df = pd.DataFrame(mdp.get_states(), columns = ["Q1", "Q2", "Y1", "Y2"])
    df = df[(df["Y1"] == 1) & (df["Y2"] == 1)]
    df = df[(df["Q1"] >= 1) & (df["Q2"] >= 1)]
    actions  = [(False, True, False), (False, False, True)]
    df["Q_diff"] = df.apply(lambda x: values.get_q_value(mdp, tuple(x[["Q1", "Q2", "Y1", "Y2"]]), actions[0]) - values.get_q_value(mdp, tuple(x[["Q1", "Q2", "Y1", "Y2"]]), actions[1]), axis = 1)


    return df

