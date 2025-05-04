
import os
import sys
import numpy as np
import traci
import matplotlib.pyplot as plt

TOTAL_STEPS = 20000

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

# Smoothed and Moving Average Trendlines
window = 100
trend_window_1 = 1000
trend_window_2 = 10000

smoothed = np.convolve(reward_per_step, np.ones(window)/window, mode='same')
trendline_1 = np.convolve(smoothed, np.ones(trend_window_1)/trend_window_1, mode='same') if len(smoothed) >= trend_window_1 else []
trendline_2 = np.convolve(smoothed, np.ones(trend_window_2)/trend_window_2, mode='same') if len(smoothed) >= trend_window_2 else []

plt.figure(figsize=(10, 6))
plt.plot(smoothed, label=f"{window}-Step Moving Average")
if len(trendline_1): plt.plot(np.arange(len(trendline_1)), trendline_1, 'r--', label=f"{trend_window_1}-Step Trendline")
if len(trendline_2): plt.plot(np.arange(len(trendline_2)), trendline_2, 'g--', label=f"{trend_window_2}-Step Trendline")
plt.xlabel("Simulation Step")
plt.ylabel("Smoothed Reward")
plt.title(f"SUMO-Timed: {window}-Step Smoothed Reward with Trendlines")
plt.grid(True)
plt.legend()
plt.show()
