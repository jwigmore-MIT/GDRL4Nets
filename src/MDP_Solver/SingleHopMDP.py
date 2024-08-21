from DP.mdp import MDP
import numpy as np
from copy import deepcopy
import pickle
import os.path as path
import tensordict
import torch
#from DP.policy_iteration import PolicyIteration
from DP.value_iteration import ValueIteration
from DP.value_iteration_plus import ValueIterationPlus
from DP.value_iteration_minus import ValueIterationMinus
from DP.tabular_value_function import TabularValueFunction
import os
from DP.tabular_policy import TabularPolicy
from DP.policy_iteration import PolicyIteration
from collections import defaultdict

class SingleHopMDP(MDP):

    def __init__(self, env, name = "SingleHopMDP", q_max = 10, discount = 0.99, value_iterator = "normal"):
        self.actions = np.eye(env.action_spec.n).astype(bool) # each, column is a one-hot boolean vector
        self.n_queues = env.N
        self.state_list = self._get_state_list(env, q_max) # truncated state-space as list
        self.tx_matrix= None
        self.q_max = q_max
        self.discount = discount
        self.name = f"{name}_qmax{q_max}_discount{discount}"
        self.env = env
        self.value_iterator = value_iterator


    class TX_Matrix:

        def __init__(self, tx_matrix = None, n_tx_samples = None, num_s_a_pairs = None):
            self.tx_matrix = tx_matrix
            self.n_tx_samples = n_tx_samples
            self.num_s_a_pairs = num_s_a_pairs

        def __call__(self, *args, **kwargs):
            return self.tx_matrix

        def __as_dict__(self):
            return {"tx_matrix": self.tx_matrix, "n_tx_samples": self.n_tx_samples, "num_s_a_pairs": self.num_s_a_pairs}

    def estimate_tx_matrix(self,
                           max_samples=1000,
                           min_samples=100,
                           theta=0.001,
                           save_path = None):
        env = self.env
        tx_matrix, n_tx_samples = form_transition_matrix(env, self.state_list, self.actions, max_samples,
                                                         min_samples, theta)
        num_s_a_pairs = len(tx_matrix.keys())
        num_samples = np.array(list(n_tx_samples.values()))
        print("Transition Matrix Estimated")
        print("Mean number of samples per state-action pair: ", np.mean(num_samples))
        print("Number of state-action pairs: ", np.sum(num_s_a_pairs))
        self.tx_matrix = self.TX_Matrix(tx_matrix, n_tx_samples, num_s_a_pairs)

        if save_path is not None:
            name = f"{self.name}_max_samples-{max_samples}_tx_matrix.pkl"
            self.save_tx_matrix(save_path, name)
        return self.tx_matrix

    def compute_tx_matrix(self,
                           save_path = None):
        env = self.env
        tx_matrix = form_transition_matrix2(env, self.state_list, self.actions, q_max = self.q_max)
        num_s_a_pairs = len(tx_matrix.keys())
        print("Transition Matrix Estimated")
        print("Number of state-action pairs: ", np.sum(num_s_a_pairs))
        self.Tx_matrix = self.TX_Matrix(tx_matrix, num_s_a_pairs=num_s_a_pairs)
        self.tx_matrix = self.Tx_matrix.tx_matrix
        if save_path is not None:
            name = f"{self.name}_computed_tx_matrix.pkl"
            self.save_tx_matrix(save_path, name)
        return self.tx_matrix

    def save_tx_matrix(self, save_path, name = "tx_matrix.pkl"):
        # if the save_path does not exist, create it
        if not path.exists(save_path):
            os.makedirs(save_path)
        pickle.dump(self.Tx_matrix.__as_dict__(), open(path.join(save_path, name), "wb"))
    def save_mdp(self, save_path, name = None):
        self.env = None
        if name is None:
            name = f"{self.name}_MDP.pkl"
        if not path.exists(save_path):
            os.makedirs(save_path)
        pickle.dump(self, open(path.join(save_path, name), "wb"))


    def load_tx_matrix(self, path):
        tx_matrix_dict = pickle.load(open(path, "rb"))
        self.TX_Matrix = self.TX_Matrix(**tx_matrix_dict)
        # self.n_tx_samples = {key: len(self.tx_matrix[key]) for key in self.tx_matrix.keys()}
        self.tx_matrix = self.TX_Matrix.tx_matrix

    def get_states(self):
        # return all possible states from 0 to s_max for each server
        return self.state_list

    def get_transitions(self, state, action):
        if hasattr(action, "tolist"):
            action = action.tolist()
        key = state + action
        transitions = list(self.tx_matrix[tuple(key)].items())
        # key = deepcopy(state)
        # key.extend(action.tolist())
        # tx_dict = self.tx_matrix[tuple(key)]
        # transitions = list(zip(list(tx_dict.keys()), list(tx_dict.values())))

        return transitions

    def get_reward(self, state, action, next_state):
        next_buffers = next_state[:self.n_queues]
        if np.any(np.array(next_buffers) >= self.q_max):
            return -np.sum(next_buffers)
        else:
            return -np.sum(next_buffers)

    def get_actions(self, state):
        self.env.base_env.set_state(state)
        mask = self.env.get_mask()
        return self.actions[mask]
        # # get all valid actions given the state
        # self.env.base_env.set_state(state)
        # mask = selfv.get_mask()
        #         # # get index of True indices
        #         # valid_action_index = np.where(mask == True)[0]
        #         # valid_actions = self.actions[valid_action_index]
        #         # return valid_actions.en

    def get_mask(self, state):
        self.env.base_env.set_state(state)
        return self.env.get_mask()


    def get_initial_state(self):
        return np.zeros(self.n_queues * 2)

    def is_terminal(self, state):
        return False

    def get_discount_factor(self):
        return self.discount

    def get_goal_states(self):
        return None

    def do_PI(self, default_policy = None, max_iterations=100, theta=0.1):
        self.pi_policy = TabularPolicy(policy_table=default_policy)
        # Need to to initialize the policy with a default action that is valid
        # for each state in the transition matrix
        if default_policy is None:
            for state in self.state_list:
                valid_actions = self.get_actions(state)
                self.pi_policy.update(state, valid_actions[0])
        policy_iteration = PolicyIteration(self, self.pi_policy)
        policy_iteration.policy_iteration(max_iterations, theta)
        print("Policy Iteration Complete, updated mdp.pi_policy")

    def save_pi_policy(self, path):
        if self.pi_policy is None:
            raise ValueError("Policy must be estimated before saving it")
        with open(path, 'wb') as f:
            pickle.dump(self.pi_policy.policy_table, f)

    def load_pi_policy(self, path):
        with open(path, 'rb') as f:
            policy_table = pickle.load(f)
        self.pi_policy = TabularPolicy(default_action=self.actions[1])
        self.pi_policy.policy_table = policy_table

    def get_pi_policy_table(self):
        if self.pi_policy is None:
            raise ValueError("Policy must be estimated before getting it")
        return self.pi_policy.policy_table



    def do_VI(self, max_iterations=100, theta=0.1):
        if self.tx_matrix is None:
            raise ValueError("Transition Matrix must be estimated before running VI")
        value_table = TabularValueFunction()
        if hasattr(self, "value_table"):
            value_table.value_table = self.value_table
        if self.value_iterator == "plus":
            self.value_iterator = ValueIterationPlus(self, value_table)
        elif self.value_iterator == "minus":
            self.value_iterator = ValueIterationMinus(self, value_table)
        else:
            self.value_iterator = ValueIteration(self, value_table)

        self.value_iterator.value_iteration(max_iterations, theta)


        policy = value_table.extract_policy(self)
        policy_table = dict(policy.policy_table)
        value_table = dict(value_table.value_table)
        self.vi_policy = policy_table
        self.value_table = value_table

    def save_VI(self, path):
        """
        Saves self.vi_policy and self.value_table to the specified path as a dictionary
        """
        if self.vi_policy is None or self.value_table is None:
            raise ValueError("VI must be run before saving the policy and value table")
        save_dict = {"vi_policy": self.vi_policy, "value_table": self.value_table}
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

    def load_VI(self, path):
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        self.vi_policy = save_dict["vi_policy"]
        self.value_table = save_dict["value_table"]



    def get_VI_policy(self):
        if self.value_table is None:
            raise ValueError("Value Table must be estimated before getting policy")
        policy = self.value_table.extract_policy(self)
        policy_table = dict(policy.policy_table)
        self.vi_policy = policy_table

    def save_MDP(self, path):
        self.env = None
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def use_vi_policy(self, state):
        if self.vi_policy is None:
            raise ValueError("Policy must be estimated before using it")
        # check if the state is in the policy table
        if tuple(state) in self.vi_policy.keys():
            return self.vi_policy[tuple(state)]
        else: # return the closest match
            # Get the MaxWeight action
            # q = np.array(state[:self.n_queues])
            # y = np.array(state[self.n_queues:])
            # mw_action = np.argmax(q*y) + 1
            # # return the mw_action as a one-hot vector of length q.shape[0] + 1
            # action = np.zeros(q.shape[0] + 1)
            # action[mw_action] = 1
            # return action

            # get the closest state in the policy table
            #print("State not in policy table")
            closest_state = min(self.vi_policy.keys(), key=lambda x: np.linalg.norm(np.array(x) - np.array(state), ord = 2))
            return self.vi_policy[closest_state]

    def use_pi_policy(self, state):
        if self.pi_policy is None:
            raise ValueError("Policy must be estimated before using it")
        # check if the state is in the policy table
        if tuple(state) in self.pi_policy.policy_table.keys():
            return self.pi_policy.policy_table[tuple(state)]
        else:
            # return the closest match
            print("State not in policy table")
            closest_state = min(self.pi_policy.policy_table.keys(), key=lambda x: np.linalg.norm(np.array(x) - np.array(state)))
            return self.pi_policy.policy_table[closest_state]

    def _get_state_list(self, env, q_max):
        """
        Get a list of all possible states in the MDP
        """

        def create_state_map(low, high):
            state_elements = []
            for l, h in zip(low, high):
                state_elements.append(list(np.arange(l, h + 1)))
            state_combos = np.array(np.meshgrid(*state_elements)).T.reshape(-1, len(state_elements))
            # convert all arrays in state_combos to lists
            state_combos = [list(state) for state in state_combos]
            return state_combos
        # Max Queue size for each queue
        # if q_max is an integer or float, we assume that all queues have the same max size
        if isinstance(q_max, (int, float)):
            high_queues = np.ones(self.n_queues) * q_max
        elif isinstance(q_max, np.ndarray):
            high_queues = q_max
        else:
            high_queues  = np.array(q_max)

        # Max Link State for each link
        high_links = np.array([rv.max() for rv in env.Y_gen[1:]])
        high = np.concatenate((high_queues, high_links))
        low = np.zeros_like(high)
        state_list = create_state_map(low, high)
        return state_list


def form_transition_matrix(env, state_list, action_list, max_samples = 1000, min_samples = 100, theta = 0.001):
    from tqdm import tqdm
    """
    state_list: list of states as lists
    action_list: list of integers
    
    
    
    """
    tx_matrix = {} # keys will be tuples of the form (state, action)
    n_samples = {}
    num_sa_pairs = len(state_list)*len(action_list)
    pbar = tqdm(total = num_sa_pairs, desc = "Estimating Transition Matrix")
    n=0
    for state in state_list:
        # get valid actions
        env.base_env.set_state(state)
        mask = env.get_mask()
        #action_list = np.where(mask == False)[0]
        for action in action_list:
            n+=1
            # update progress bar to reflect the state-action pair being processed
            pbar.update(1)
            pbar.set_description(f"Estimating Transition Matrix: State {state}, Action {action}")
            # create tuple key for the transition matrix
            key = deepcopy(state)
            key.extend(action)
            # Apply and function to the mask and action
            and_action = np.logical_and(mask, action)
            if np.any(and_action):
                tx_matrix[tuple(key)], n_samples[tuple(key)] = estimate_transitions(env, state, action, max_samples, min_samples, theta)
            else: # if the action is invalid don't add it to the transition matrix
                pass
    return tx_matrix, n_samples


def form_transition_matrix2(env, state_list, action_list, q_max = 30):
    from tqdm import tqdm
    """
    state_list: list of states as lists
    action_list: list of integers

    For each state = [Q0, Q1, Q2, ..., Qn, Y0, Y1, Y2, ..., Yn] and action = a_i, the transition matrix is a dictionary
    There are only a couple of reachable states for each state action pair
    a_i is a one-hot vector
    
    Let Q' be the Q vector after applying action i to state Q
    Q' = Q-a_i*Y
    
    The next possible Q states depend on all possible arrival vectors X
    X has a limited alphabet and we can compute the probability of each x in X
    P(X = x) = P(X0 = x0, X1 = x1, ..., Xn = xn) = P(X0 = x0)P(X1 = x1)...P(Xn = xn)
    
    
    # Should also recompute transition matrix such that for all states where q_i> q_max, we set q_i = q_max
    

    """
    tx_matrix = {}  # keys will be tuples of the form (state, action)
    n_samples = {}
    num_sa_pairs = len(state_list) * len(action_list)
    pbar = tqdm(total=num_sa_pairs, desc="Estimating Transition Matrix")
    n = 0

    all_x_dist = [X.dist() for X in env.X_gen[1:]]
    all_y_dist = [Y.dist() for Y in env.Y_gen[1:]]
    X_dist = MultiDiscreteDistribution(all_x_dist)
    Y_dist = MultiDiscreteDistribution(all_y_dist)
    XY_dist = CombinedMultiDiscreteDistribution(X_dist.dist_dict, Y_dist.dist_dict)


    for state in state_list:
        # get valid actions
        env.base_env.set_state(state)
        mask = env.get_mask()
        # action_list = np.where(mask == False)[0]
        for action in action_list:
            n += 1
            # update progress bar to reflect the state-action pair being processed
            pbar.update(1)
            pbar.set_description(f"Estimating Transition Matrix: State {state}, Action {action}")
            # create tuple key for the transition matrix
            and_action = np.logical_and(mask, action)
            if np.any(and_action):
                key = deepcopy(state)
                key.extend(action)
                Q = np.array(state[:env.N])
                Y = np.array(state[env.N:])
                if action[0]:
                    Q_prime = Q # meaning we idle
                else:
                    Q_prime = np.max([Q - np.multiply(Y,action[1:]), np.zeros_like(Q)], axis = 0)
                    # Q_prime =np.max([Q - np.multiply(Y,action[1:]), np.zeros_like(Q)], axis = 0), q_max*np.ones_like(Q) # restrict Q to be less than q_max
                temp = deepcopy(XY_dist.dist_dict)
                new_dist = defaultdict(lambda : 0)
                for x_key, x_value in temp.items():
                    state_prime = np.concatenate([Q_prime, np.zeros_like(Q_prime)])
                    #new_key = np.min([np.array(list(x_key)) + state_prime, q_max*np.ones_like(x_key)], axis = 0)
                    new_key = np.array(list(x_key)) + state_prime
                    new_dist[tuple(new_key)] += x_value
                    # check that the new dist sums to 1
                if np.abs(np.sum(list(new_dist.values())) - 1) > 0.01:
                    raise ValueError("Transition Probabilities do not sum to one")
                # convert new_dist to a normal dictionary
                tx_matrix[tuple(key)] = dict(new_dist)
            else:
                pass


    return tx_matrix


def estimate_transitions(env, state, action, max_samples = 1000, min_samples = 100, theta = 0.001):
    """
    Estimate the transition probabilities for a given state-action pair
    """

    C_sas = {} # counter for transitions (s,a,s')
    P_sas = {} # probabilities for transitions (s,a,s')
    diff = {}
    action = torch.Tensor(action)
    for n in range(1, max_samples+1):
        env.base_env.set_state(state)
        action_td = tensordict.TensorDict({"action": action,
                                                  "step_count": torch.Tensor([n])}
                                          , batch_size=[])

        td = env.step(action_td)
        next_state = torch.concatenate([td["next","Q"], td["next","Y"]]).numpy()
        if tuple(next_state) in C_sas.keys():
            C_sas[tuple(next_state)] += 1# increment the counter for the visits to next state
            p_sas = C_sas[tuple(next_state)]/n # calculate the probability of the transition
            diff[tuple(next_state)] = np.abs(P_sas[tuple(next_state)] - p_sas)
            P_sas[tuple(next_state)] = p_sas
        else:
            C_sas[tuple(next_state)] = 1
            P_sas[tuple(next_state)] = 1

        # check for convergence:
        if n > min_samples and np.all(list(diff.values())) < theta:
            break
    P_sas = {key: value/n for key, value in C_sas.items()}
    if np.abs(np.sum(list(P_sas.values())) - 1) > 0.001:
        raise ValueError("Transition Probabilities do not sum to one")

    # convert to probabilities

    return P_sas, n


class MultiDiscreteDistribution:

    def __init__(self, dists: dict):
        "Each dist in dists is a tuple of lists"
        "The first list in each tuple specifies the values"
        "The second list in each tuple specifies the probabilities"
        self.marginal_dists = dists
        self.dist_dict = {}
        self.create_dist_dict()

    def create_dist_dict(self):
        "I need to make a dictionary of all possible value vectors and their probabilities"
        "Start by getting all combinations of values from the marginal distributions"
        X_i_keys = [list(dist.keys()) for dist in self.marginal_dists]
        X_i_values = [list(dist.values()) for dist in self.marginal_dists]
        X_keys = np.array(np.meshgrid(*X_i_keys)).T.reshape(-1, len(X_i_keys))
        X_values = np.array(np.meshgrid(*X_i_values)).T.reshape(-1, len(X_i_values)).prod(axis = 1)

        for key, value in zip(X_keys, X_values):
            self.dist_dict[tuple(key)] = value

class CombinedMultiDiscreteDistribution:

    def __init__(self, dist1: dict, dist2: dict):
        self.dist1 = dist1
        self.dist2 = dist2
        self.dist_dict = {}
        self.create_dist_dict()

    def create_dist_dict(self):
        for key1, value1 in self.dist1.items():
            for key2, value2 in self.dist2.items():
                # create a tuple that combines all elements in key1 and key2
                key = list(deepcopy(key1))
                key.extend(key2)
                self.dist_dict[tuple(key)] = value1*value2
        # check to make sure that all probabilities sum to 1
        if np.abs(np.sum(list(self.dist_dict.values())) - 1) > 0.01:
            raise ValueError("Transition Probabilities do not sum to one")

if __name__ == "__main__":
    from torchrl_development.envs.env_generators import make_env, parse_env_json
    import numpy as np
    env_params = parse_env_json("SH1.json")
    env = make_env(env_params,observation_keys=["Q", "Y"])
    mdp = SingleHopMDP(env, name = "SH1", q_max = 20)
    # mdp.estimate_tx_matrix(max_samples = 1000, min_samples=500, save_path = "")
    mdp.compute_tx_matrix(save_path = "")
    #mdp.load_tx_matrix("SH1_qmax5_discount0.99_max_samples-100_tx_matrix.pkl")

    all_x_dist = [X.dist() for X in env.X_gen[1:]]
    all_y_dist = [Y.dist() for Y in env.Y_gen[1:]]
    X_dist = MultiDiscreteDistribution(all_x_dist)
    Y_dist = MultiDiscreteDistribution(all_y_dist)
    XY_dist = CombinedMultiDiscreteDistribution(X_dist.dist_dict, Y_dist.dist_dict)

    # all_x_dist is a multivariate discrete distribution where all_x_dist[i] is the distribution of X_i

