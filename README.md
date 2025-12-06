# Reinforcement Learning Demos

A personal portfolio project where I assess and implement different reinforcement learning (RL) algorithms across classic control environments. The goal is to build clear, reproducible baselines starting from tabular Q-learning and moving towards function-approximation methods like DQN, policy gradients, and actor–critic.

## Introduction to Reinforcement Learning
- **Environment**: The world the agent interacts with. It defines state transitions and reward signals.
- **Agent**: The learner/decision-maker that selects actions to maximize cumulative reward.
- **States**: Observations that describe the environment at a given time.
- **Actions**: Choices the agent can take from a state.
- **Rewards**: Scalar feedback indicating the immediate desirability of the last action.
- **State Value Function** `V(s)`: Expected return starting from state `s` under a policy.
- **State–Action Value Function** `Q(s, a)`: Expected return from state `s` taking action `a` and following a policy thereafter.

## Tabular Q-Learning
Tabular Q-learning is a value-based, off-policy algorithm that learns the optimal `Q(s, a)` by iteratively updating a table. It uses the Bellman optimality equation with temporal-difference (TD) targets:

`Q(s, a) ← Q(s, a) + α [ r + γ max_{a'} Q(s', a') − Q(s, a) ]`

Key properties:
- **Off-policy**: Learns the greedy policy while exploring with ε-greedy.
- **Requires discrete state space**: Each row of the Q-table corresponds to a unique state index.
- **Converges with sufficient exploration and decaying learning rate** in stationary environments.

In this repo we apply tabular Q-learning to two discrete environments where a Q-table is feasible: **Taxi** and **FrozenLake**.

## Environments and Gymnasium
We use the `gymnasium` library to create and interact with environments.

- **Taxi-v3**: A grid-world with discrete states and actions; the agent picks up and drops off a passenger. Deterministic transitions, sparse rewards, clear terminal conditions.
- **FrozenLake-v1**: A grid of safe tiles and holes. With `is_slippery=True`, transitions are stochastic, making learning less stable.

You can quickly get a feel for each environment by playing with the provided scripts.

### Play the Games
- Taxi: `demos/Taxi/play.py`
- FrozenLake: `demos/FrozenLake/play.py`

Run with:

```bash
# Activate environment (example: conda base)
source /opt/conda/bin/activate base

# Install dependencies
pip install -r requirements.txt

# Play Taxi
python demos/Taxi/play.py

# Play FrozenLake
python demos/FrozenLake/play.py
```

## Training: Tabular Q-Learning
Current implemented agent: `rl_agents/tabular_q_learning.py` with demos in:
- Taxi training: `demos/Taxi/q_learning.py`
- FrozenLake training: `demos/FrozenLake/q_learning.py`

Run training:

```bash
# Taxi training
python demos/Taxi/q_learning.py

# FrozenLake training
python demos/FrozenLake/q_learning.py
```

### Taxi Results and Conclusions
- **Learning behavior**: Rewards per episode improve as ε decays and the Q-table values stabilize.
- **Convergence**: The agent reliably completes the task (pickup → dropoff) with high success rate after sufficient episodes.
- **Takeaways**: Taxi’s discrete, deterministic dynamics are well-suited for tabular methods. Hyperparameters (α, γ, ε schedule) primarily affect speed of convergence.

### FrozenLake Notes
- **Stochasticity**: With `is_slippery=True`, actions may not lead to intended moves, increasing variance in episode returns.
- **Metrics**: Success rate and average return are not perfect, but the agent achieves a performance considered “solved” for the chosen map and configuration.
- **Implication**: Even with tabular methods, careful tuning and sufficient exploration can handle moderate stochasticity; however, function approximation often scales better for larger or more complex maps.

## Repository Structure
- `demos/Taxi/`: Play and tabular Q-learning training scripts for Taxi.
- `demos/FrozenLake/`: Play and tabular Q-learning training scripts for FrozenLake.
- `rl_agents/`: Python package for agents.
	- `tabular_q_learning.py`: Current implementation of a tabular Q-learning agent.
	- `dqn.py`: Placeholder for a Deep Q-Network agent.
	- `agent.py`: Base interfaces/utilities.

## Roadmap
- **DQN**: Implement Deep Q-Network with replay buffer, target network, and ε-greedy exploration; apply to `LunarLander-v2`.
- **Policy Gradient (TBD)**: Implement a baseline REINFORCE or similar method for a suitable environment.
- **Actor–Critic (TBD)**: Implement an advantage actor–critic variant for improved sample efficiency.
- **Experiment tracking**: Add logging and plots for returns, success rates, and Q-value convergence.
