# HIGHWAY_RL_PROJECT
# Autonomous Highway Driving Agent (DQN)

*Authors:*
* *Belamir Korkmaz* - [2101600]
* *Eren Yiğit Ateş* - [2202893]
* *Tarık Tanırcan* - [Student ID]

*Frameworks:* Gymnasium, Highway-Env, PyTorch

---

## Team Contribution Statement

All group members contributed equally to every stage of this project. We conducted regular synchronous working sessions to:

1.  Pair-program the codebase and debug implementation issues.
2.  Monitor and tune the training processes (optimizing for **500,000 steps** and fine-tuning hyperparameters).
3.  Co-author this technical report and analyze the resulting metrics.

---

## 1. 🎥 The Evolution (Visual Proof)

Evolution video of the Agent starting from untrained, followed by half-trained and fully trained versions.

![Evolution Video](https://github.com/user-attachments/assets/f069c6c7-7323-44d6-a5bb-7b904334765b)


---

## 2. Methodology

### The Goal
The objective was to train an agent in the `highway-fast-v0` environment to maximize speed while avoiding collisions. The agent controls a vehicle in a 4-lane highway with dense traffic consisting of **30 vehicles**.

### The Model Architecture
We utilized a *Deep Q-Network (DQN)* because it yielded better results for this specific discrete action space.

* *Algorithm:* DQN (Deep Q-Network)
* *Policy:* MLP (Multi-Layer Perceptron)
* *Network Architecture:* **[256, 256]** fully connected layers
* *Optimizer:* Adam (learning_rate=**5e-4**)
* *Buffer Size:* **50,000** transitions
* *Exploration:* Epsilon-greedy starting at 100% and decaying to **5%**.

*Hyperparameter Justification:*
* *Learning Rate (5e-4):* A lower learning rate was chosen to ensure stable convergence.
* *Exploration (Epsilon Decay Steps):* We allow an exploration phase covering **40% of total steps** (**200,000 steps**) to ensure the agent encounters rare scenarios.
* *Target Update (1000):* Target network updates are performed every **1000 steps** to provide stable learning targets.

### The Mathematical Reward Function
The total reward $R_t$ at step $t$ is calculated based on our configuration:

$$R_t = R_{speed} + P_{Collision} + P_{LaneChange}$$

Where:
* *Speed Reward ($R_{speed}$):* Linearly mapped from the range **$[35, 50]$** km/h. Driving below 35 km/h yields 0 reward.
* *Collision Penalty ($P_{Collision} = -125.0$):* A severe penalty to strictly forbid collisions.
* *Lane Change Penalty ($P_{LaneChange} = -1.2$):* A fee to prevent unnecessary zigzagging.

---

## 3. Training Analysis

The agent was trained for **500,000 timesteps** using **multi-core (asynchronous) vectorization**.

![Graph](<img width="1000" height="500" alt="training_graph" src="https://github.com/user-attachments/assets/8c494bdb-9aad-4636-8c59-c79636c9433c" />)


### Commentary
The learning curve demonstrates a distinct three-phase progression:
1.  *Initial Instability (0 - 200k):* The agent acts randomly due to high epsilon, resulting in frequent collisions.
2.  *Exploration & Learning (200k - 350k):* As epsilon decays toward 5%, the agent starts to prioritize staying in lanes and avoiding the **-125.0** penalty.
3.  *Convergence & Mastery (350k+):* The agent "cracks" the code—realizing that maintaining speed between **35-50 km/h** is the only way to maximize the function.

---

## 4. Challenges & Failures

### 1. Stuck at Low Speeds
*Problem:* The agent was sitting at low speeds to avoid risk.
*Fix:* We set `reward_speed_range` to **[35, 50]**. This made high speeds mathematically necessary.

### 2. The "Profitable Crash" Issue
*Problem:* A short burst of speed was sometimes worth more than a minor penalty.
*Fix:* We raised the collision penalty to **-125.0**, making survival the absolute priority.

### 3. The "Zig-Zag" Effect
*Problem:* Constant lane changes.
*Fix:* We set `lane_change_reward` to **-1.2**. This discourages it unless necessary for overtaking.

### 4. Lack of Curiosity
*Fix:* We extended the Epsilon Decay period to **200,000 steps** (40% of training).

### 5. Catastrophic Forgetting
*Fix:* We increased the batch size to **256** and memory capacity to **50,000** to keep the replay buffer diverse.

### 6. Observation Space: Greyscale vs. Kinematics
*Decision:* We used **Kinematics** (mathematical vectors). This allows the agent to perceive surroundings directly, achieving optimal results with only **500k steps**.

---

## 5. Results & Evaluation

Comparison based on training logs and `simulasyon.py` results:

| Metric | Half-Trained (250k) | Fully Trained (500k) | Improvement |
| :--- | :--- | :--- | :--- |
| *Crash Rate* | High | Low (~10%) | *Safety Improved* |
| *Average Speed* | Low | ~48 km/h | *High Efficiency* |
| *Behavior* | Learning Lanes | Aggressive / Precise | *Mastery* |

### Conclusion
The data confirms that **500,000 timesteps** with a **256-unit hidden layer** was the optimal setup.

---

## Resources
* *Library:* [Highway-Env Documentation](https://highway-env.farama.org/)
---

