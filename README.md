# 🤖 AutoRL: Autonomous Reinforcement Learning Research

## 🎯 The Vision

Most time in Robotics RL is wasted "reward shaping"—manually tweaking floating-point numbers in a Python file to make a robot walk instead of backflip.

AutoRobot-RL implements Andrei Karpathy's "AutoResearch" philosophy. Instead of a human engineer sitting at the desk, an LLM Researcher acts as the lead scientist. It autonomously proposes hypotheses, modifies the training script, executes RL runs in simulation, and analyzes the telemetry to converge on an optimal policy.

## 🛠️ How It Works

**Setup & Installation:**
Ensure you initialize the git submodules containing the MuJoCo assets when cloning:
```bash
git clone --recursive https://github.com/Om-Kulkarni/AutoRL.git
# Or if already cloned:
# git submodule update --init --recursive
```

The system operates in a closed-loop "Scientific Method" cycle:

1.  **Hypothesize:** The `researcher.py` (GPT-4/Claude) reads the output/logs from the previous run.
2.  **Edit:** It identifies bottlenecks (e.g., "The quadruped is moving forward but shaking violently") and modifies `scripts/train_mutable.py` to add a torque smoothness penalty.
3.  **Execute:** `main.py` triggers a simulation run in the `PandaEnv` (e.g., 50,000 steps in MuJoCo).
4.  **Observe:** The system captures the Success Rate and Reward Curve.
5.  **Archive:** The specific version of the code and its results are saved in `experiments/run_N`.

## 🚀 Key Features

*   **Mutable Training Script:** A designated sandbox file (`train_mutable.py`) where the LLM has full permission to rewrite Reward Functions, Hyperparameters (LR, Gamma, Batch Size), and Network Topology.
*   **Static Physics Anchor:** The physics environment (e.g., `env/panda_env.py` which wraps `base_sim.py`) remains untouched to ensure scientific validity across experiments. The MuJoCo simulation includes a 7-DoF Franka Panda arm and parallel gripper utilizing DeepMind's `mujoco_menagerie` models.
*   **Self-Correction:** If the LLM writes code that throws a `NameError` or NaN loss, the researcher agent receives the traceback and must fix its own "bug" in the next iteration.

## 📊 Success Metrics

*   **Sample Efficiency:** Reaching target velocity in fewer simulation steps.
*   **Policy Smoothness:** Minimizing jitter and energy consumption.
*   **Sim-to-Real Score:** Performance in "noisy" environments designed to mimic the real world.
