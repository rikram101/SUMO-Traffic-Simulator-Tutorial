
import os
import sys
import numpy as np
import traci
import matplotlib.pyplot as plt

TOTAL_STEPS = 20000
MIN_GREEN_STEPS = 200

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

Sumo_config = [
    "sumo-gui", "-c", "FourWay.sumocfg", "--step-length", "0.10", "--delay", "1"
]
traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

# -------------------------
# Fixed-time Logic
# -------------------------
last_switch_J1 = -MIN_GREEN_STEPS
last_switch_J2 = -MIN_GREEN_STEPS

def get_state(detectors):
    return sum(traci.lanearea.getLastStepVehicleNumber(det) for det in detectors)

def switch_if_ready(junction_id, last_switch_step):
    current_phase = traci.trafficlight.getPhase(junction_id)
    num_phases = len(traci.trafficlight.getAllProgramLogics(junction_id)[0].phases)
    if (step - last_switch_step) >= MIN_GREEN_STEPS:
        traci.trafficlight.setPhase(junction_id, (current_phase + 1) % num_phases)
        return step
    return last_switch_step

# Detector IDs
detectors_J1 = [f"Node1_{d}_{i}" for d in ["EB", "NB", "SB", "WB"] for i in range(3)]
detectors_J2 = [f"Node2_{d}_{i}" for d in ["EB", "NB", "SB", "WB"] for i in range(3)]

reward_per_step = []
cumulative_reward = 0

# -------------------------
# Main Simulation Loop
# -------------------------
for step in range(TOTAL_STEPS):
    total_queue = get_state(detectors_J1) + get_state(detectors_J2)
    reward = -total_queue
    reward_per_step.append(reward)
    cumulative_reward += reward

    last_switch_J1 = switch_if_ready("J1", last_switch_J1)
    last_switch_J2 = switch_if_ready("J2", last_switch_J2)

    traci.simulationStep()

traci.close()
print(f"Cumulative reward: {cumulative_reward}")
print(f"Average reward per step: {cumulative_reward / TOTAL_STEPS:.2f}")

# -------------------------
# Plotting
# -------------------------
plt.figure(figsize=(10, 6))
plt.plot(reward_per_step, label="Reward per Step")
plt.xlabel("Simulation Step")
plt.ylabel("Reward")
plt.title("Fixed-Time Controller: Reward per Step")
plt.grid(True)
plt.legend()
plt.show()

# Smoothed
window = 100
smoothed = np.convolve(reward_per_step, np.ones(window)/window, mode='valid')
plt.figure(figsize=(10, 6))
plt.plot(smoothed, label=f"{window}-Step Moving Average")
plt.xlabel("Simulation Step")
plt.ylabel("Smoothed Reward")
plt.title(f"Fixed-Time: {window}-Step Moving Average of Reward")
plt.grid(True)
plt.legend()
plt.show()
