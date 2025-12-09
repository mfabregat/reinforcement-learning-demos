# Reinforcement Learning: From Theory to Practice

Welcome! This repository bridges the gap between Reinforcement Learning (RL) theory and code. We will explore fundamental concepts by implementing agents that learn to solve increasingly complex tasks.

## 1. Introduction to Reinforcement Learning

At its core, RL is about learning from **interaction**. An **Agent** takes actions in an **Environment**, which responds with a **State** update and a **Reward**.

<img src="assets/rl_loop.png" width=50% alt="Reinforcement Learning Loop"/>

- **Agent**: The learner (e.g., our code).
- **Environment**: The world (e.g., the game).
- **State ($s$)**: The current situation.
- **Action ($a$)**: The move the agent makes.
- **Reward ($r$)**: Feedback on how good the action was.

The goal is to find a **Policy ($\pi$)**—a strategy mapping states to actions—that maximizes the total expected reward over time.

---

## 2. Tabular Q-Learning (The Basics)

### The Theory: Q-Values & The Bellman Equation
How does the agent know which action is best? It learns a **Action-Value Function**, $Q(s, a)$, which estimates the total future reward of taking action $a$ in state $s$.

For simple low-dimensional environments like Taxi and FrozenLake, we can store these values in a table (a Q-Table). The agent updates this table based on its experience using the **Bellman Equation**:

$$ Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)] $$

- $\alpha$ (Learning Rate): How much we trust new information.
- $\gamma$ (Discount Factor): How much we value future rewards vs. immediate ones.
- $\max_{a'} Q(s', a')$: The best possible future value from the next state.

> **_Note:_** When we refer to _environments_ like Taxi and FrozenLake, we are referring to those provided by the [Gymnasium library](https://gymnasium.farama.org/). This library is the standard for prototyping RL environments in Python.

### The Practice: Learning to Drive a Taxi
The first environment is `Taxi-v3`. The agent must pick up and drop off passengers at the right locations on a grid.
You can play the game manually to get a feel for it:

```bash
python demos/Taxi/human_play.py
```

Now, let's train a Q-Learning agent to master the Taxi environment:

```bash
python demos/Taxi/train.py
```

Several figures will be generated that will tell us how well the agent learnt:

TO ADD: Metrics figure Taxi

Finally, we can watch the trained agent in action:

```bash
python demos/Taxi/agent_play.py
```
TO ADD: Video of trained Taxi agent

Note that this environment is _deterministic_; the same action in the same state will always yield the same result. This makes it easier for the agent to learn.

### The Practice: Navigating FrozenLake
Next, we tackle `FrozenLake-v1`, where the agent must navigate a slippery frozen lake to reach a goal without falling into holes.
Every time an action is taken, there's a chance the agent will slip and end up in a different state than intended. This stochasticity makes learning more challenging.

Once again, you can try playing it yourself:

```bash
python demos/FrozenLake/human_play.py
```

Now, let's train a Q-Learning agent for FrozenLake:

```bash
python demos/FrozenLake/train.py
```

TO ADD : Metrics figure FrozenLake

Note how the learning curve is less smooth than in Taxi, due to the environment's randomness.

Finally, watch the trained agent navigate the frozen lake:

```bash
python demos/FrozenLake/agent_play.py
```
TO ADD: Video of trained FrozenLake agent

---

## 3. Deep Q-Networks (Going Deep)

### The Theory: From Tables to Networks
What happens when there are too many possible states to store in a table? If we tried to use a Q-table for a high-dimensional environment, the memory requirements would quickly become infeasible.
Moreover, environments are not always discrete; they can have continuous state spaces (like positions and velocities). We could discretize them, but that leads to loss of information and poor performance.
In complex environments (like video games or robotics), the number of possible states is too huge for a table. We can't store a Q-value for every single pixel configuration!

Instead, we use a **Neural Network** to *approximate* the Q-function. This is called a **Deep Q-Network (DQN)**.
- Input: The state (e.g., coordinates, velocity).
- Output: Q-values for every possible action (i.e., how good each action is for the input state).

This introduces new challenges, like instability, which we solve with techniques like **Experience Replay** (learning from past memories) and **Target Networks** (keeping the learning target stable).

### The Practice: Learning to Land a Spacecraft on the Moon
The `LunarLander-v2` is a classic control problem where the agent must land a spacecraft safely on the moon's surface. You can try it out yourself:
```bash
python demos/LunarLander/human_play.py
```

Now, let's train a DQN agent to master the Lunar Lander:

```bash
python demos/LunarLander/train.py
```

TO ADD: Metrics figure LunarLander

Finally, watch the trained agent land the spacecraft:

```bash
python demos/LunarLander/agent_play.py
```
TO ADD: Video of trained LunarLander agent

---

## 4. Policy Gradients (Directly Learning Policies)

TO DO: Disadvantages of value-based methods

### The Theory: Policy Optimization
**Policy Gradient** methods directly optimize the policy $\pi(a|s)$ instead of estimating value functions. This is especially useful for environments with continuous action spaces (like robots) where selecting discrete actions isn't feasible.
The key idea is to adjust the policy parameters $\theta$ in the direction that maximizes expected reward. The update rule is based on the **Policy Gradient Theorem**:
$$ \nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(a|s) Q^\pi(s, a)] $$

Where:
- $J(\theta)$: The expected return under policy $\pi_\theta$.
- $d^\pi$: The state distribution under policy $\pi$.

There exist several algorithms based on this idea, but one of the most popular is **Proximal Policy Optimization (PPO)**, which balances exploration and exploitation while ensuring stable updates.

### The Practice: Learning to Run
Now we are getting into more complex environments. The `HalfCheetah-v4` environment requires the agent to learn to run as fast as possible using a simulated two-legged robot.

This time, the action space is continuous (e.g., how much to move each joint) and 6-dimensional, so there's no easy way to test it manually.
Instead, let's directly train a PPO agent to master the HalfCheetah:

```bash
python demos/HalfCheetah/train.py
```

This time, rather than coding our own PPO implementation, which would be less efficient, we leverage the powerful [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/) library.
However, the REINFORCE algorithm has been implemented from scratch in `rl_agents/reinforce.py` for educational purposes.

Let's see how well our agent learned to run:

```bash
python demos/HalfCheetah/agent_play.py
```
TO ADD: Video of trained HalfCheetah agent

---


## 5. Actor-Critic methods (Mastering Control)
### The Theory: Combining Value and Policy Learning
**Actor-Critic** methods combine the strengths of value-based and policy-based approaches.
They maintain two models:
- The **Actor** learns the policy $\pi(a|s)$, deciding which action to take (like in policy gradients).
- The **Critic** estimates the value function $V(s)$ or $Q(s, a)$, providing feedback (thus _critic_) on how good the action was (like in value-based methods).
The Actor updates its policy based on the feedback from the Critic, allowing for more stable and efficient learning.

In this tutorial, we will use the **Soft Actor-Critic (SAC)** algorithm implemented in Stable Baselines3, which is particularly effective for continuous action spaces.

### The Practice: Learning to Walk with a Humanoid Robot
The `Humanoid-v5` environment challenges the agent to control a humanoid robot to stand and walk upright.
Let's train a SAC agent to master the Humanoid:

```bash
python demos/Humanoid/train.py
```
This environment is quite complex, so training will take a significant amount of time (several hours).

Once trained, we can watch the humanoid agent in action:

```bash
python demos/Humanoid/agent_play.py
```
TO ADD: Video of trained Humanoid agent

Indeed, the gait learnt by the humanoid is quite strange, but it manages to stay upright and walk forward! The training could be further improved by tuning hyperparameters and training for longer, but this serves as a solid introduction to Actor-Critic methods.
You can check repository TBD for a realistic humanoid simulation and an actually good policy to walk.