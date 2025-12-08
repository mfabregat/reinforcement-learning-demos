# Reinforcement Learning Tutorial

This repository aims to provide a clear and concise introduction to fundamental Reinforcement Learning (RL) concepts and algorithms through custom implementations, popular library integrations, and practical demos.

## Introduction to Reinforcement Learning

Reinforcement Learning is a subfield of machine learning where an agent (i.e., the model) learns to make decisions by performing actions in an environment to maximize cumulative rewards. Unlike supervised learning (e.g., image recognition), RL does not rely on labeled datasets. Instead, it focuses on learning optimal policies through interaction with the environment (trial and error).


<img src="assets/rl_loop.png" width=50% alt="Reinforcement Learning Loop"/>


The figure above illustrates the core components of an RL system:
1. **Agent**: The learner or decision-maker that selects actions based on a policy π.
2. **Environment**: The external system with which the agent interacts (e.g., a game, a financial market).
3. **State**: A representation of the current situation of the agent within the environment (e.g., the position of a game character, the current stock prices).
4. **Action**: A set of possible moves the agent can make (e.g., move left, buy stock).
5. **Reward**: A scalar feedback signal indicating the immediate benefit of an action taken (e.g., points scored, profit gained).

Agents seek to learn a policy π that maximizes the expected cumulative reward over time, often formalized through Markov Decision Processes (MDPs).
A policy π is a mapping from states to actions. That is, given a state ``s`, the policy π(s) defines the action `a` the agent should take to maximize future rewards.
Different RL algorithms, such as the ones introduced in this repository, provide various approaches to learning optimal policies.

## Q Learning

Besides using the reward `r` as a heuristic to guide policy search, it can also be used to directly learn the value of taking a specific action `a` in a given state `s`. This is known as the Q-value, denoted as `Q(s, a)`. The Q-value represents the expected cumulative reward the agent can obtain by taking action `a` in state `s` and following the optimal policy thereafter:
$$ Q(s, a) = \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t r_{t} \mid s_0 = s, a_0 = a \right] $$

Note that the successive rewards are discounted by a factor `γ` (0 ≤ γ < 1) to prioritize immediate rewards over distant future rewards. This component determines the trade-off between exploration (seeking new knowledge, when `γ` is close to 0) and exploitation (leveraging known information, when `γ` is close to 1).

The Q-learning algorithm iteratively updates the Q-values based on the agent's experiences using the Bellman equation:
$$ Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right] $$
where:
- `α` is the learning rate (0 < α ≤ 1), determining how much new information overrides old information.
- `s'` is the next state after taking action `a` in state `s`.
- `max_{a'} Q(s', a')` represents the maximum expected future reward achievable from state `s'`.    

