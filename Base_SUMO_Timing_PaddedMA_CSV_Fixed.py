
import os
import sys
import numpy as np
import traci
import matplotlib.pyplot as plt

TOTAL_STEPS = 10000

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

# Padded moving average
def padded_moving_average(data, window):
    pad_width = window // 2
    padded = np.pad(data, (pad_width, pad_width), mode='edge')
    return np.convolve(padded, np.ones(window)/window, mode='valid')

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

# Smoothed and padded trendlines
window = 100
trend_window_1 = 1000
trend_window_2 = 10000


# Compute moving averages and trendlines
smoothed = padded_moving_average(reward_per_step, window)
trendline_1 = padded_moving_average(smoothed, trend_window_1)
trendline_2 = padded_moving_average(smoothed, trend_window_2)

# Export data to CSV
import csv
with open("reward_graph_data.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Step", "Reward", "Smoothed", "Trendline_1000", "Trendline_10000"])
    for i in range(len(smoothed)):
        row = [
            i,
            reward_per_step[i] if i < len(reward_per_step) else "",
            smoothed[i],
            trendline_1[i] if i < len(trendline_1) else "",
            trendline_2[i] if i < len(trendline_2) else ""
        ]
        writer.writerow(row)

