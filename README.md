# HIGHWAY_RL_PROJECT

# Autonomous Highway Driving Agent (DQN)

*Authors:*
* *Belamir Korkmaz* - [2101600]
* *Eren Yiğit Ateş* - [2202893]
* *Tarık Tanırcan* - [2103149]

*Frameworks:* Gymnasium, Highway-Env, PyTorch

---

## Team Contribution Statement

All group members contributed equally to every stage of this project. We conducted regular synchronous working sessions to:

1. Pair-program the codebase and debug implementation issues.
2. Monitor and tune the training processes (optimizing for **500,000 steps** and fine-tuning hyperparameters).
3. Co-author this technical report and analyze the resulting metrics.

---

## 1. 🎥 The Evolution (Visual Proof)

Evolution video of the Agent starting from untrained, followed by half-trained and fully trained versions.

https://github.com/user-attachments/assets/f069c6c7-7323-44d6-a5bb-7b904334765b

---

## 2. 📁 Repository Structure

To maintain high **Repo Hygiene**, our project is organized as follows:

* **`src/`**: Contains core logic, including `agent.py` for DQN architecture and `config.py` for centralized hyperparameters.
* **`models/`**: Stores serialized `.pth` files for the untrained, half-trained, and final models.
* **`videos/`**: Contains recorded evolution videos and simulation clips.
* **`train.py`**: Entry point for starting the multi-core (asynchronous) training process.
* **`simulasyon.py`**: Script to visualize and record the agent's performance.
* **`requirements.txt`**: List of required Python libraries for reproducibility.

---

## 3. 🧠 Methodology

### The Goal
The objective was to train an agent in the `highway-fast-v0` environment to maximize speed while avoiding collisions. The agent controls a vehicle in a 4-lane highway with dense traffic consisting of **30 vehicles**.

### 🚗 Agent Capabilities (Actions & States)

**Observation Space (What the Agent Sees):**
We use the **Kinematics** observation type. The agent perceives a matrix representing:
* **Presence**: Slot availability of nearby vehicles.
* **Coordinates**: Relative $x$ and $y$ positions of surrounding cars.
* **Velocities**: The speed of neighbors in both $x$ and $y$ directions.

**Action Space (What the Agent Does):**
The agent operates with a **DiscreteMetaAction** space:

| Action ID | Action Name | Description |
| :--- | :--- | :--- |
| **0** | **LANE_LEFT** | Switches to the lane on the left if safe. |
| **1** | **IDLE** | Maintains the current lane and speed. |
| **2** | **LANE_RIGHT** | Switches to the lane on the right if safe. |
| **3** | **FASTER** | Increases speed to reach the $[35, 50]$ km/h target. |
| **4** | **SLOWER** | Decreases speed to avoid obstacles. |

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

## 4. 📈 Training Analysis

The agent was trained for **500,000 timesteps** using **multi-core (asynchronous) vectorization**.

<img width="100%" alt="training_graph" src="https://github.com/user-attachments/assets/86693dd3-3cb3-4090-9403-ebc5bf0dbba3" />

### Commentary
The learning curve demonstrates a distinct three-phase progression:
1. *Initial Instability (0 - 200k):* The agent acts randomly due to high epsilon, resulting in frequent collisions.
2. *Exploration & Learning (200k - 350k):* As epsilon decays toward 5%, the agent starts to prioritize staying in lanes and avoiding the **-125.0** penalty.
3. *Convergence & Mastery (350k+):* The agent "cracks" the code—realizing that maintaining speed between **35-50 km/h** is the only way to maximize the function.

---

## 5. 🛑 Challenges & Failures

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

## 6. 🛠️ Installation & Usage

To ensure the environment is set up correctly (as required by the **Setup** criteria):

1. **Clone the Repository**: Ensure all source files and the `models/` folder are present.
2. **Setup & Execution**:
   Use the commands below to install dependencies and run the project:
   ```bash
   # 1. Install Dependencies
   pip install -r requirements.txt

   # 2. Run Training (Starts training the agent from scratch)
   python train.py

   # 3. Run Simulation (Watch the fully trained agent drive)
   python simulasyon.py

## 7. Results & Evaluation

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

