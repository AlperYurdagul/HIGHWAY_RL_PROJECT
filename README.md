# HIGHWAY_RL_PROJECT
# Autonomous Highway Driving Agent (DQN)

*Authors:*
*   *Belamir Korkmaz* - [2101600]
*   *Eren Yiğit Ateş* - [Student ID]
*   *Tarık Tanırcan* - [Student ID]

*Frameworks:* Gymnasium, Highway-Env, PyTorch

---

## Team Contribution Statement

All group members contributed equally to every stage of this project. We conducted regular synchronous working sessions to:

1.  Pair-program the codebase and debug implementation issues.
2.  Monitor and tune the training processes (optimizing for 750k steps and fine-tuning hyperparameters).
3.  Co-author this technical report and analyze the resulting metrics.

---

## 1. Evolution of the Agent

Evolution video of the Agent starting from untrained, followed by half-trained and fully trained versions.

![Evolution Video](videos/dqn_highway_final/evolution.mp4)

---

## 2. Methodology

### The Goal

The objective was to train an agent in the ⁠ highway-fast-v0 ⁠ environment to maximize speed while avoiding collisions. The agent controls a vehicle in a 4-lane highway with dense traffic.

### The Model Architecture

We utilized a *Deep Q-Network (DQN)* because it yielded better results for this specific discrete action space.

*   *Algorithm:* DQN (Deep Q-Network)
*   *Policy:* MLP (Multi-Layer Perceptron)
*   *Network Architecture:* [256, 256] fully connected layers
*   *Optimizer:* Adam (learning_rate=5e-4)
*   *Buffer Size:* 20,000 transitions
*   *Exploration:* Epsilon-greedy starting at 100% and decaying to 5%.

*Hyperparameter Justification:*

*   *Learning Rate (5e-4):* A lower learning rate was chosen to ensure stable convergence. Standard rates (e.g., 1e-3) often caused the loss to oscillate.
*   *Exploration (Decay 30%):* We allow a long exploration phase (30% of total steps) to ensure the agent encounters rare crash scenarios before converging.
*   *Target Update (500):* Frequent target updates provided more stable learning targets in this highly dynamic environment.

### The Mathematical Reward Function

To solve the "passive driver" problem, we engineered a custom reward function. The total reward $R_t$ at step $t$ is calculated as:

$$ R_t = (w_{speed} \cdot R_{speed}) + P_{Collision} + P_{LaneChange} $$

Where:
*   *Speed Reward ($R_{speed}$):* Linearly mapped from the range $[35, 50]$ km/h. Driving below 35 km/h yields 0 reward, forcing the agent to speed up.
*   *Collision Penalty ($P_{Collision} = -100$):* A severe penalty to strictly forbid collisions.
*   *Lane Change Penalty ($P_{LaneChange} = -0.75$):* A minor fee to prevent unnecessary zigzagging.

---

## 3. Training Analysis

The agent was trained for *750,000 timesteps*. Below is the performance analysis based on the cumulative reward per episode.

![Graph](training_graph.png)

### Commentary

The learning curve demonstrates a distinct three-phase progression:

1.  *Initial Instability (0 - 200k):* The graph begins with low rewards. The agent acts randomly (high epsilon), resulting in frequent crashes.
2.  *Exploration & Learning (200k - 500k):* The reward trend begins to rise. The agent learns the physics of the car and understands that crashing is the ultimate failure state. It starts to stay in lanes.
3.  *Convergence & Mastery (500k+):* A dramatic shift occurs. The agent "cracks" the code—realizing that high-speed weaving is the only way to maximize the function. The reward stabilizes at a high level.

---

## 4. Challenges & Failures

Our road to a working model was not a straight line. We iterated through several distinct major issues:

### 1. Stuck at Low Speeds
*Problem:* The agent was not speeding up properly. It was sitting at a comfortable low speed, never attempting to overtake.
*Fix:* We altered the speed reward logic. We mapped the reward to be positive ONLY if speed > 35 km/h. This made high speeds mathematically irresistible.

### 2. The "Profitable Crash" Issue
*Problem:* The agent was crashing even at the fully trained state because a short burst of speed was worth more than the crash penalty.
*Fix:* We raised the collision penalty to ⁠ -100 ⁠, making survival the absolute priority.

### 3. The "Zig-Zag" Effect
*Problem:* The agent started prioritizing lane changes over speed, resulting in constant "zig-zag" driving.
*Fix:* We adjusted the lane change reward to be a penalty (⁠ -0.75 ⁠). Just enough to discourage it, but allowed if necessary for overtaking.

### 4. Lack of Curiosity
*Problem:* The agent found a "comfort zone" and stopped trying new strategies.
*Fix:* We extended the Epsilon Decay period to cover 30% of the training, forcing the agent to explore random actions for much longer.

### 5. Catastrophic Forgetting
*Problem:* As the agent improved, it stopped crashing. Consequently, the Replay Buffer filled up exclusively with "safe driving" data, pushing out the early "crash experiences." The agent eventually forgot that crashing was bad.
*Fix:* We increased the batch size to ⁠ 128 ⁠ and kept the replay buffer diverse to prevent overfitting to safe data.

### 6. Observation Space: Greyscale vs. Kinematics
*Problem:* Our initial approach was to use ⁠ GrayscaleImage ⁠ (pixel-based) observations to simulate a realistic autonomous driving experience using computer vision. We trained this model for an extensive *1.2 million steps*, expecting it to learn spatial awareness from raw pixels. However, despite the significant computational resource and time investment, the agent struggled to extract optimal features and failed to converge to a robust driving policy.
*Decision:* We decided to pivot to ⁠ Kinematics ⁠ (mathematical vectors) to represent the environment. This change allowed the agent to perceive the exact positions and velocities of surrounding vehicles directly. As a result, we achieved significantly *more optimal results* with *fewer training steps*, proving that for this specific environment, a vector-based observation space is far more efficient than a pixel-based one.

---

## 5. Results & Evaluation

We compared the agent's performance halfway through training versus the final result.

### Comparison Table

| Metric | Half-Trained | Fully Trained | Improvement |
| :--- | :--- | :--- | :--- |
| *Crash Rate* | High (~80%) | Low (~10%) | *Safety Improved* |
| *Average Speed* | ~20 km/h | ~48 km/h | *+140%* |
| *Episode Length* | Short (Crashes) | Max Duration | *+100%* |
| *Behavior* | Wobbly / Passive | Aggressive / Precise | *Mastery* |

### Conclusion
The data confirms that 750,000 timesteps was necessary. The agent keeps the high speed (~48 km/h) it learned early on, but it is now smart enough to avoid collisions.

---

## Resources
*   *Repository:* [GitHub Link]
*   *Library:* [Highway-Env Documentation](https://highway-env.farama.org/)
