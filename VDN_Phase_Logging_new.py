
import os
import sys
import random
import numpy as np
import traci
import matplotlib.pyplot as plt

TOTAL_STEPS = 5000
MIN_GREEN_STEPS = 100

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
# Detector Definitions
# -------------------------
detectors_J1 = [f"Node1_{d}_{i}" for d in ["EB", "NB", "SB", "WB"] for i in range(3)]
detectors_J2 = [f"Node2_{d}_{i}" for d in ["EB", "NB", "SB", "WB"] for i in range(3)]

def get_state(detectors):
    return {
        "vehicle_count": sum(traci.lanearea.getLastStepVehicleNumber(d) for d in detectors),
        "halting_count": sum(traci.lanearea.getLastStepHaltingNumber(d) for d in detectors)
    }

reward_per_step = []
reward_J1_only = []
reward_J2_only = []
cumulative_reward = 0

last_switch_J1 = -MIN_GREEN_STEPS
last_switch_J2 = -MIN_GREEN_STEPS

def switch_if_ready(junction_id, last_switch_step):
    current_phase = traci.trafficlight.getPhase(junction_id)
    num_phases = len(traci.trafficlight.getAllProgramLogics(junction_id)[0].phases)
    if (step - last_switch_step) >= MIN_GREEN_STEPS:
        traci.trafficlight.setPhase(junction_id, (current_phase + 1) % num_phases)
        return step
    return last_switch_step

# -------------------------
# Main Simulation Loop
# -------------------------
for step in range(TOTAL_STEPS):
    state_J1 = get_state(detectors_J1)
    state_J2 = get_state(detectors_J2)

    reward_J1 = -(state_J1["vehicle_count"] + state_J1["halting_count"])
    reward_J2 = -(state_J2["vehicle_count"] + state_J2["halting_count"])
    reward = reward_J1 + reward_J2

    reward_J1_only.append(reward_J1)
    reward_J2_only.append(reward_J2)
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
window = 100
def smooth(y, w=window):
    return np.convolve(y, np.ones(w)/w, mode='valid')

plt.figure(figsize=(10, 6))
plt.plot(smooth(reward_per_step), label="Total Reward")
plt.plot(smooth(reward_J1_only), label="J1 Reward")
plt.plot(smooth(reward_J2_only), label="J2 Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Smoothed Reward")
plt.title(f"{window}-Step Moving Average of Reward by Junction")
plt.grid(True)
plt.legend()
plt.show()
