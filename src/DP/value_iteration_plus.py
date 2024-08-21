from tqdm import tqdm
from DP.qtable import *
from DP.tabular_value_function import *
import torch
import torch.nn as nn
"""
We want to train a critic network to imitate the value table of the MDP.  
Let V_\phi(s) be the value of state s according to the critic network.  We want to minimize the loss function:
     1/2 (V_\phi(s) - V(s))^2
where V(s) is the value of state s according to the value table.
"""

class Critic(nn.Module):

    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

class ValueIterationPlus:
    def __init__(self, mdp, values):
        self.mdp = mdp
        self.values = values
        self.critic = Critic(mdp.n_queues*2)
        # Find the index of the state that is [self.mdp.q_max, self.mdp.q_max, 0, 0]
        self.q_max_index = self.mdp.get_states().index([self.mdp.q_max-10, self.mdp.q_max-10, 0, 0])
    def value_iteration(self, max_iterations=100, theta=0.1):
        pbar = tqdm(range(int(max_iterations)))
        critic_loss = nn.MSELoss()
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.01)

        for i in pbar:
            delta = 0.0
            new_values = TabularValueFunction()
            for state in self.mdp.get_states():
                qtable = QTable()
                # check if state contains mdp.q_max
                if self.mdp.q_max not in state:
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
                    # Use the critic network to predict the value of the state
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    value = self.critic(state_tensor).item()
                    delta = max(delta, abs(self.values.get_value(state) - value))
                    new_values.update(state, value)
            self.values.merge(new_values)


            states = torch.tensor(list(self.values.value_table.keys()), dtype=torch.float32)
            targets = torch.tensor(list(self.values.value_table.values()), dtype=torch.float32)
            # split states, targets into batches
            batch_size = 32
            n_batches = len(states) // batch_size
            critic_losses = []
            for j in range(n_batches):
                critic_optimizer.zero_grad()
                batch_states = states[j*batch_size:(j+1)*batch_size]
                batch_targets = targets[j*batch_size:(j+1)*batch_size]
                critic_values = self.critic(batch_states)
                loss = critic_loss(critic_values, batch_targets)
                loss.backward()
                critic_optimizer.step()
                critic_losses.append(loss.item())

            total_loss = sum(critic_losses)/len(critic_losses)

            pbar.set_description(f"Delta: {delta}, Critic Loss: {loss.detach().item()/total_loss}")

            # Terminate if the value function has converged
            if delta < theta:
                return i

