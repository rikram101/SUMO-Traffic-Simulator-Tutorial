
import os
import sys
import numpy as np
import traci
import matplotlib.pyplot as plt

TOTAL_STEPS = 1000

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
# Detector Setup
# -------------------------
def get_state(detectors):
    return sum(traci.lanearea.getLastStepVehicleNumber(det) for det in detectors)

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

    # Let SUMO handle signal timing (no phase switching here)
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
plt.title("SUMO-Timed Controller: Reward per Step")
plt.grid(True)
plt.legend()
plt.show()

# Smoothed and Moving Average Trendline
window = 100
smoothed = np.convolve(reward_per_step, np.ones(window)/window, mode='valid')

# Apply a second, longer moving average for trendline
trend_window = 1000
trendline = np.convolve(smoothed, np.ones(trend_window)/trend_window, mode='valid')

plt.figure(figsize=(10, 6))
plt.plot(smoothed, label=f"{window}-Step Moving Average")
plt.plot(np.arange(len(trendline)), trendline, 'r--', label=f"{trend_window}-Step Trendline")
plt.xlabel("Simulation Step")
plt.ylabel("Smoothed Reward")
plt.title(f"SUMO-Timed: {window}-Step Smoothed Reward with {trend_window}-Step Trendline")
plt.grid(True)
plt.legend()
plt.show()
