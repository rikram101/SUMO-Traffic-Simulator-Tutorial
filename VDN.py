import os
import sys
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import traci
import matplotlib.pyplot as plt
# -------------------------
# Hyperparameters
# -------------------------
TOTAL_STEPS = 6000
BATCH_SIZE = 64
GAMMA = 0.9
LR = 1e-3
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = TOTAL_STEPS
MIN_GREEN_STEPS = 100

# -------------------------
# SUMO Setup
# -------------------------
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

Sumo_config = [
    'sumo-gui', '-c', 'FourWay.sumocfg', '--step-length', '0.10', '--delay', '1000', '--lateral-resolution', '0'
]
traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

# -------------------------
# Neural Network for Each Agent
# -------------------------
class AgentNet(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),      nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def forward(self, x):
        return self.net(x)

# -------------------------
# Replay Buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def add(self, obs1, obs2, state, actions, reward, next_obs1, next_obs2, next_state):
        self.buffer.append((obs1, obs2, state, actions, reward, next_obs1, next_obs2, next_state))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))
    def __len__(self):
        return len(self.buffer)

# -------------------------
# Utility Functions
# -------------------------
def get_state_J1():
    obs = []
    # Existing detectors for East Bound (EB) and South Bound (SB)
    for det in ["Node1_EB_0", "Node1_EB_1", "Node1_EB_2",
                "Node1_SB_0", "Node1_SB_1", "Node1_SB_2"]:
        obs.append(traci.lanearea.getLastStepVehicleNumber(det))
    
    # New detectors for North Bound (NB)
    for det in ["Node1_NB_0", "Node1_NB_1", "Node1_NB_2"]:
        obs.append(traci.lanearea.getLastStepVehicleNumber(det))
    
    # New detectors for West Bound (WB)
    for det in ["Node1_WB_0", "Node1_WB_1", "Node1_WB_2"]:
        obs.append(traci.lanearea.getLastStepVehicleNumber(det))
    
    # Add the current phase of the traffic light
    obs.append(traci.trafficlight.getPhase("J1"))
    return np.array(obs, dtype=np.float32)

def get_state_J2():
    obs = []
    # Existing detectors for East Bound (EB) and South Bound (SB)
    for det in ["Node2_EB_0", "Node2_EB_1", "Node2_EB_2",
                "Node2_SB_0", "Node2_SB_1", "Node2_SB_2"]:
        obs.append(traci.lanearea.getLastStepVehicleNumber(det))
    
    # New detectors for North Bound (NB)
    for det in ["Node2_NB_0", "Node2_NB_1", "Node2_NB_2"]:
        obs.append(traci.lanearea.getLastStepVehicleNumber(det))
    
    # New detectors for West Bound (WB)
    for det in ["Node2_WB_0", "Node2_WB_1", "Node2_WB_2"]:
        obs.append(traci.lanearea.getLastStepVehicleNumber(det))
    
    # Add the current phase of the traffic light
    obs.append(traci.trafficlight.getPhase("J2"))
    return np.array(obs, dtype=np.float32)

def get_global_state(s1, s2):
    return np.concatenate([s1, s2]).astype(np.float32)

# -------------------------
# Apply Actions with MIN_GREEN_STEPS Logic
# -------------------------
# Initialize step counters for each traffic light
last_switch_step_J1 = -MIN_GREEN_STEPS
last_switch_step_J2 = -MIN_GREEN_STEPS

def apply_action_J1(action):
    global last_switch_step_J1
    current_step = step  # Use the current step from the main loop

    if action == 1 and (current_step - last_switch_step_J1 >= MIN_GREEN_STEPS):
        current_phase = traci.trafficlight.getPhase("J1")
        num_phases = len(traci.trafficlight.getAllProgramLogics("J1")[0].phases)
        next_phase = (current_phase + 1) % num_phases
        traci.trafficlight.setPhase("J1", next_phase)
        last_switch_step_J1 = current_step  # Update the last switch step

def apply_action_J2(action):
    global last_switch_step_J2
    current_step = step  # Use the current step from the main loop

    if action == 1 and (current_step - last_switch_step_J2 >= MIN_GREEN_STEPS):
        current_phase = traci.trafficlight.getPhase("J2")
        num_phases = len(traci.trafficlight.getAllProgramLogics("J2")[0].phases)
        next_phase = (current_phase + 1) % num_phases
        traci.trafficlight.setPhase("J2", next_phase)
        last_switch_step_J2 = current_step  # Update the last switch step

# -------------------------
# Update Observation Dimension
# -------------------------
obs_dim = 13  # Updated from 7 (6 new detectors added for each agent)

# -------------------------
# Instantiate Networks, Buffer, Optimizers
# -------------------------
n_actions = 2
agent1_net = AgentNet(obs_dim, n_actions)
agent2_net = AgentNet(obs_dim, n_actions)
optimizer = optim.Adam(
    list(agent1_net.parameters()) + list(agent2_net.parameters()), lr=LR
)
replay_buffer = ReplayBuffer()

dt = lambda step: EPS_END + (EPS_START - EPS_END) * np.exp(-1. * step / EPS_DECAY)

# -------------------------
# Main Loop: Collect & Train
# -------------------------
cumulative_reward = 0.0
reward_per_step = []  # Track rewards for each step

for step in range(TOTAL_STEPS):
    eps = dt(step)
    # ---- Get Observations ----
    s1 = get_state_J1()
    s2 = get_state_J2()
    gs = get_global_state(s1, s2)

    # ---- Select Actions ----
    if random.random() < eps:
        a1, a2 = random.randrange(n_actions), random.randrange(n_actions)
    else:
        with torch.no_grad():
            q1 = agent1_net(torch.tensor(s1))
            q2 = agent2_net(torch.tensor(s2))
        a1 = int(q1.argmax().item())
        a2 = int(q2.argmax().item())

    # ---- Apply Actions ----
    apply_action_J1(a1)  # Apply action for J1
    apply_action_J2(a2)  # Apply action for J2
    traci.simulationStep()

    # ---- Observe Next ----
    ns1 = get_state_J1()
    ns2 = get_state_J2()
    ngs = get_global_state(ns1, ns2)
    reward = -(ns1[:-1].sum() + ns2[:-1].sum())
    cumulative_reward += reward

    # ---- Track Reward ----
    reward_per_step.append(reward)  # Track reward for this step

    # ---- Store ----
    replay_buffer.add(s1, s2, gs, (a1, a2), reward, ns1, ns2, ngs)

    # ---- Training ----
    if len(replay_buffer) >= BATCH_SIZE:
        b_s1, b_s2, b_gs, b_a, b_r, b_ns1, b_ns2, b_ngs = replay_buffer.sample(BATCH_SIZE)

        # Convert to tensors
        bs1 = torch.tensor(b_s1)
        bs2 = torch.tensor(b_s2)
        br = torch.tensor(b_r).unsqueeze(1)
        ba1 = torch.tensor(b_a[:, 0]).unsqueeze(1)
        ba2 = torch.tensor(b_a[:, 1]).unsqueeze(1)
        bns1 = torch.tensor(b_ns1)
        bns2 = torch.tensor(b_ns2)

        # Q-values for current states
        q1_vals = agent1_net(bs1).gather(1, ba1)
        q2_vals = agent2_net(bs2).gather(1, ba2)
        q_tot = q1_vals + q2_vals

        # Target Q for next states (VDN uses sum of max next-agent-Qs)
        with torch.no_grad():
            next_q1 = agent1_net(bns1).max(1)[0].unsqueeze(1)
            next_q2 = agent2_net(bns2).max(1)[0].unsqueeze(1)
            target = br + GAMMA * (next_q1 + next_q2)

        # Loss and optimize
        loss = nn.MSELoss()(q_tot, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# -------------------------
# Cleanup
# -------------------------
traci.close()
print(f"Training done. Cumulative reward: {cumulative_reward:.2f}")

# Calculate average reward per step
avg_reward_per_step = cumulative_reward / TOTAL_STEPS
print(f"Average reward per step: {avg_reward_per_step:.2f}")

# Plot the reward per step
plt.figure(figsize=(10, 6))
plt.plot(reward_per_step, label="Reward per Step")
plt.xlabel("Simulation Step")
plt.ylabel("Reward")
plt.title("Reward per Step Over Time")
plt.grid(True)
plt.legend()
plt.show()

# Smooth the reward curve using a moving average
window = 100  # Smoothing window
rewards_smooth = np.convolve(reward_per_step, np.ones(window) / window, mode='valid')

# Plot the smoothed reward curve
plt.figure(figsize=(10, 6))
plt.plot(rewards_smooth, label=f"{window}-Step Moving Average")
plt.xlabel("Simulation Step")
plt.ylabel("Smoothed Reward")
plt.title(f"{window}-Step Moving Average of Reward")
plt.grid(True)
plt.legend()
plt.show()
