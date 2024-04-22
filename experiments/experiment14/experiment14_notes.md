# Purpose
The goal of experiment 14 is to determine if the MDP and MWN models can learn the optimal policy 

# Main Question: Can an MWN agent learn the optimal policy?
To verify this we want to:
1. Create a network instance with a small effective state-action space under the MaxWeight and optimal policy
2. Solve for the optimal policy using the Value Iteration Algorithm
Check if 
3. The MLP network can imitate the optimal policy – and if not, can the MLP network learn a policy that is close to optimal 
4. The MWN network can imitate the optimal policy – and if not, can the MWN network learn a policy that is close to optimal

## Step 1: Create a network instance with a small effective state-action space under the MaxWeight and optimal policy
Note: We also want there to be a gap between the performance of the MaxWeight and Optimal policy. 
### File: SH1B_Context_Enumeration.py
The goal of this file is to create a network instance with a small effective state-action space under the MaxWeight and optimal policy.
It is a manual script that will import the topology from the SH1B.json environment file, and allow me to change the 
arrival rates and service rates of the network.  The SH1B network is a 2 queue network with Bernoulli arrivals of rate
$\lambda_i$ and Bernoulli capacities with rate $\mu_i$. 

## Step 2: Solve for the optimal policy using the Value Iteration Algorithm
This was done in the file SH1B_Context_Enumeration.py. The optimal policy improves upon MaxWeight run on SH1B.json by about 37%


## Step 3: See if the MLP network can imitate the optimal policy
Found that the MLP policy can learn the VI policy.  

