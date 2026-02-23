# Self-Play DDPG Tennis

**Author**: Tim Wilcoxson

A self-play DDPG agent that learns to play Tennis in a Unity ML-Agents environment, where two agents cooperate to rally a ball over a net.

## Project Details

In this environment, two agents control rackets to bounce a ball over a net.

- **State space**: The observation space consists of 24 variables per agent, corresponding to the position and velocity of the ball and racket (3 stacked frames × 8 variables each).
- **Action space**: Each agent has 2 continuous actions available — movement toward or away from the net, and jumping — each clipped to [-1, 1].
- **Agents**: 2 agents sharing one actor, one critic, and one replay buffer (self-play).
- **Reward**: An agent receives +0.1 for hitting the ball over the net, and -0.01 if the ball hits the ground or goes out of bounds.
- **Solving condition**: The environment is considered solved when the average score reaches +0.5 over 100 consecutive episodes, where each episode's score is the maximum over both agents.

## Getting Started

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Dependencies

1. Clone this repository:
   ```bash
   git clone https://github.com/trwilcoxson/self-play-ddpg-tennis.git
   cd self-play-ddpg-tennis
   ```

2. Create and activate the conda environment:
   ```bash
   conda create -n drlnd-nav python=3.10 -y
   conda activate drlnd-nav
   ```

3. Install dependencies (includes PyTorch, NumPy, Jupyter, protobuf, and all other required packages):
   ```bash
   cd python
   pip install .
   cd ..
   ```

4. Install the Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name drlnd-nav --display-name "Python (drlnd-nav)"
   ```

### Download the Unity Environment

Download the Tennis environment for your OS, unzip it, and place it in the project root:

- [macOS](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- [Windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
- [Windows 32-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)

**macOS users**: After unzipping, remove the quarantine attribute so the app can launch:
```bash
xattr -cr Tennis.app
```

**Linux/Windows users**: After unzipping, update the `file_name` path in the notebook's environment initialization cell to match your extracted binary (e.g., `Tennis_Linux/Tennis.x86_64` or `Tennis_Windows_x86_64/Tennis.exe`).

## Instructions

To train the agent from scratch:

```bash
conda activate drlnd-nav
jupyter notebook Tennis.ipynb
```

Select the **"Python (drlnd-nav)"** kernel and run all cells. The notebook will:
1. Initialize the Tennis environment
2. Train a self-play DDPG agent (typically solves in 1000–2000 episodes)
3. Save model weights to `checkpoint_actor.pth` and `checkpoint_critic.pth`
4. Plot the training scores
5. Run a 100-episode greedy evaluation with the trained weights

To watch the trained agent play without retraining, skip the training cell and load the pre-trained `.pth` weights directly.

Alternatively, run the standalone training script:
```bash
conda activate drlnd-nav
python -u train.py
```

## Project Structure

| File | Description |
|---|---|
| `Tennis.ipynb` | Main training notebook with results and report |
| `model.py` | Actor and Critic network architectures (128-unit layers, no BatchNorm) |
| `maddpg_agent.py` | Self-play DDPG agent with exploration noise decay, OU noise, replay buffer |
| `train.py` | Standalone training script |
| `Report.md` | Detailed report: algorithm, architecture, hyperparameters, plot, future work |
| `checkpoint_actor.pth` | Trained actor weights (saved at solve point) |
| `checkpoint_critic.pth` | Trained critic weights (saved at solve point) |
| `scores.npy` | Raw per-episode scores |
| `scores_plot.png` | Training rewards plot |
| `python/` | Bundled Unity ML-Agents Python package (v0.4) |

## Results

See [Report.md](Report.md) for the full learning algorithm description, architecture details, training plot, and ideas for future work.
