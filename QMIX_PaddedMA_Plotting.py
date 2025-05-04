import os
import sys
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import traci
import matplotlib.pyplot as plt

# -------------------------
# Hyperparameters
# -------------------------
TOTAL_STEPS = 80000
BATCH_SIZE = 64
GAMMA = 0.9
LR = 1e-3
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = TOTAL_STEPS
MIN_GREEN_STEPS = 100
TARGET_UPDATE = 200  # steps between target network sync

# -------------------------
# SUMO Setup
# -------------------------
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

Sumo_config = [
    'sumo-gui', '-c', 'FourWay.sumocfg',
    '--step-length', '0.10', '--delay', '1', '--lateral-resolution', '0'
]
traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

# -------------------------
# Device & Seeds
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# -------------------------
# Neural Networks
# -------------------------
class AgentNet(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def forward(self, x):
        return self.net(x)

class Mixer(nn.Module):
    def __init__(self, n_agents, state_dim, embed_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        # hypernetworks produce positive weights
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, 1))
    def forward(self, agent_qs, states):
        # agent_qs: [batch, n_agents], states: [batch, state_dim]
        bs = agent_qs.size(0)
        # first layer
        w1 = torch.abs(self.hyper_w1(states))
        w1 = w1.view(bs, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(bs, 1, self.embed_dim)
        q_vals = agent_qs.view(bs, 1, self.n_agents)
        hidden = torch.relu(torch.bmm(q_vals, w1) + b1)  # [bs,1,embed]
        # second layer
        w2 = torch.abs(self.hyper_w2(states)).view(bs, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(bs, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2  # [bs,1,1]
        return q_tot.view(bs, 1)

# -------------------------
# Replay Buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def add(self, s1, s2, gs, actions, reward, ns1, ns2, ngs):
        self.buffer.append((s1, s2, gs, actions, reward, ns1, ns2, ngs))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))
    def __len__(self):
        return len(self.buffer)

# -------------------------
# Utility: States & Actions
# -------------------------
def get_state_J1():
    obs = []
    for det in ["Node1_EB_0","Node1_EB_1","Node1_EB_2",
                "Node1_SB_0","Node1_SB_1","Node1_SB_2",
                "Node1_NB_0","Node1_NB_1","Node1_NB_2",
                "Node1_WB_0","Node1_WB_1","Node1_WB_2"]:
        obs.append(traci.lanearea.getLastStepVehicleNumber(det))
    obs.append(traci.trafficlight.getPhase("J1"))
    return np.array(obs, dtype=np.float32)

def get_state_J2():
    obs = []
    for det in ["Node2_EB_0","Node2_EB_1","Node2_EB_2",
                "Node2_SB_0","Node2_SB_1","Node2_SB_2",
                "Node2_NB_0","Node2_NB_1","Node2_NB_2",
                "Node2_WB_0","Node2_WB_1","Node2_WB_2"]:
        obs.append(traci.lanearea.getLastStepVehicleNumber(det))
    obs.append(traci.trafficlight.getPhase("J2"))
    return np.array(obs, dtype=np.float32)

def get_global_state(s1, s2):
    return np.concatenate([s1, s2]).astype(np.float32)

# -------------------------
# Action Application
# -------------------------
last_switch_step_J1 = -MIN_GREEN_STEPS
last_switch_step_J2 = -MIN_GREEN_STEPS

def apply_action_J1(action, step):
    global last_switch_step_J1
    if action == 1 and (step - last_switch_step_J1 >= MIN_GREEN_STEPS):
        phase = traci.trafficlight.getPhase("J1")
        num = len(traci.trafficlight.getAllProgramLogics("J1")[0].phases)
        traci.trafficlight.setPhase("J1", (phase+1)%num)
        last_switch_step_J1 = step

def apply_action_J2(action, step):
    global last_switch_step_J2
    if action == 1 and (step - last_switch_step_J2 >= MIN_GREEN_STEPS):
        phase = traci.trafficlight.getPhase("J2")
        num = len(traci.trafficlight.getAllProgramLogics("J2")[0].phases)
        traci.trafficlight.setPhase("J2", (phase+1)%num)
        last_switch_step_J2 = step

# -------------------------
# Setup Models & Buffer
# -------------------------
obs_dim = 13
state_dim = obs_dim * 2
n_actions = 2
n_agents = 2

agent1 = AgentNet(obs_dim, n_actions).to(device)
agent2 = AgentNet(obs_dim, n_actions).to(device)
mixer = Mixer(n_agents, state_dim).to(device)

agent1_target = copy.deepcopy(agent1)
agent2_target = copy.deepcopy(agent2)
mixer_target = copy.deepcopy(mixer)
for p in mixer_target.parameters(): p.requires_grad = False

optimizer = optim.Adam(
    list(agent1.parameters()) + list(agent2.parameters()) + list(mixer.parameters()), lr=LR
)
replay = ReplayBuffer()

dt = lambda t: EPS_END + (EPS_START-EPS_END)*np.exp(-1. * t / EPS_DECAY)

# -------------------------
# Training Loop
# -------------------------
cumulative_reward = 0.0
reward_per_step = []

for step in range(TOTAL_STEPS):
    eps = dt(step)
    s1 = get_state_J1(); s2 = get_state_J2()
    gs = get_global_state(s1, s2)

    # select actions
    if random.random() < eps:
        a1, a2 = random.randrange(n_actions), random.randrange(n_actions)
    else:
        with torch.no_grad():
            q1 = agent1(torch.from_numpy(s1).to(device))
            q2 = agent2(torch.from_numpy(s2).to(device))
        a1, a2 = int(q1.argmax()), int(q2.argmax())

    # apply and step
    apply_action_J1(a1, step)
    apply_action_J2(a2, step)
    traci.simulationStep()

    # observe
    ns1 = get_state_J1(); ns2 = get_state_J2()
    ngs = get_global_state(ns1, ns2)
    reward = -(ns1[:-1].sum() + ns2[:-1].sum())
    cumulative_reward += reward
    reward_per_step.append(reward)

    # store
    replay.add(s1, s2, gs, (a1,a2), reward, ns1, ns2, ngs)

    # train
    if len(replay) >= BATCH_SIZE:
        b_s1, b_s2, b_gs, b_a, b_r, b_ns1, b_ns2, b_ngs = replay.sample(BATCH_SIZE)
        bs1 = torch.from_numpy(b_s1).float().to(device)
        bs2 = torch.from_numpy(b_s2).float().to(device)
        bgs = torch.from_numpy(b_gs).float().to(device)
        br = torch.from_numpy(b_r).float().unsqueeze(1).to(device)
        ba1 = torch.from_numpy(b_a[:,0]).long().unsqueeze(1).to(device)
        ba2 = torch.from_numpy(b_a[:,1]).long().unsqueeze(1).to(device)
        bns1 = torch.from_numpy(b_ns1).float().to(device)
        bns2 = torch.from_numpy(b_ns2).float().to(device)
        bngs = torch.from_numpy(b_ngs).float().to(device)

        # current Q vals
        q1_vals = agent1(bs1).gather(1, ba1)
        q2_vals = agent2(bs2).gather(1, ba2)
        q_vals = torch.cat([q1_vals, q2_vals], dim=1)  # [B,2]
        q_tot = mixer(q_vals, bgs)

        # target Q
        with torch.no_grad():
            next_q1 = agent1_target(bns1).max(1)[0].unsqueeze(1)
            next_q2 = agent2_target(bns2).max(1)[0].unsqueeze(1)
            next_qs = torch.cat([next_q1, next_q2], dim=1)
            target = br + GAMMA * mixer_target(next_qs, bngs)

        loss = nn.MSELoss()(q_tot, target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    # update targets
    if step % TARGET_UPDATE == 0:
        agent1_target.load_state_dict(agent1.state_dict())
        agent2_target.load_state_dict(agent2.state_dict())
        mixer_target.load_state_dict(mixer.state_dict())

# cleanup
traci.close()

# -------------------------
# Final Plotting (with padded moving averages)
# -------------------------
def padded_moving_average(data, window):
    pad_width = window // 2
    padded = np.pad(data, (pad_width, pad_width), mode='edge')
    return np.convolve(padded, np.ones(window)/window, mode='valid')

print(f"Training done. Cumulative reward: {cumulative_reward:.2f}")
print(f"Average reward per step: {cumulative_reward / TOTAL_STEPS:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(reward_per_step, label="Reward per Step")
plt.xlabel("Simulation Step")
plt.ylabel("Reward")
plt.title("QMIX Reward per Step")
plt.grid(True)
plt.legend()
plt.show()

# Moving averages and trendlines
window = 100
trend_window_1 = 1000
trend_window_2 = 10000

smoothed = padded_moving_average(reward_per_step, window)
trendline_1 = padded_moving_average(smoothed, trend_window_1)
trendline_2 = padded_moving_average(smoothed, trend_window_2)

plt.figure(figsize=(10, 6))
plt.plot(smoothed, label=f"{window}-Step Moving Average")
plt.plot(trendline_1, 'r--', label=f"{trend_window_1}-Step Trendline")
plt.plot(trendline_2, 'g--', label=f"{trend_window_2}-Step Trendline")
plt.xlabel("Simulation Step")
plt.ylabel("Smoothed Reward")
plt.title("QMIX: Smoothed Reward with Trendlines")
plt.grid(True)
plt.legend()
plt.show()
