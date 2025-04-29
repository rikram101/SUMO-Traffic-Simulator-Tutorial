# Step 1: Add modules to provide access to specific libraries and functions
import os  # Module provides functions to handle file paths, directories, environment variables
import sys  # Module provides access to Python-specific system parameters and functions
import random
import numpy as np
import matplotlib.pyplot as plt  # Visualization

# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module to provide access to specific libraries and functions
import traci  # Static network information (such as reading and analyzing network files)

# Step 4: Define Sumo configuration
Sumo_config = [
    'sumo-gui',
    '-c', 'FourWay.sumocfg',
    '--step-length', '0.10',
    '--delay', '1000',
    '--lateral-resolution', '0'
]

# Step 5: Open connection between SUMO and Traci
traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

# -------------------------
# Step 6: Define Variables
# -------------------------

# Variables for RL State (queue lengths from detectors and current phase)
q1_EB_0 = 0
q1_EB_1 = 0
q1_EB_2 = 0
q1_SB_0 = 0
q1_SB_1 = 0
q1_SB_2 = 0
q1_NB_0 = 0
q1_NB_1 = 0
q1_NB_2 = 0
q1_WB_0 = 0
q1_WB_1 = 0
q1_WB_2 = 0
current_phase1 = 0  # Updated from current_phase

# Variables for RL State (queue lengths from detectors and current phase) for J2
q2_EB_0 = 0
q2_EB_1 = 0
q2_EB_2 = 0
q2_SB_0 = 0
q2_SB_1 = 0
q2_SB_2 = 0
q2_NB_0 = 0
q2_NB_1 = 0
q2_NB_2 = 0
q2_WB_0 = 0
q2_WB_1 = 0
q2_WB_2 = 0
current_phase2 = 0  # Updated from current_phase_cluster

# ---- Reinforcement Learning Hyperparameters ----
TOTAL_STEPS = 10000   # The total number of simulation steps for continuous (online) training.

ALPHA = 0.1            # Learning rate (α) between[0, 1]    #If α = 1, you fully replace the old Q-value with the newly computed estimate.
                                                            #If α = 0, you ignore the new estimate and never update the Q-value.
GAMMA = 0.9            # Discount factor (γ) between[0, 1]  #If γ = 0, the agent only cares about the reward at the current step (no future rewards).
                                                            #If γ = 1, the agent cares equally about current and future rewards, looking at long-term gains.
EPSILON = 0.1          # Exploration rate (ε) between[0, 1] #If ε = 0 means very greedy, if=1 means very random

ACTIONS = [0, 1]       # The discrete action space (0 = keep phase, 1 = switch phase)

# Q-table dictionary: key = state tuple, value = numpy array of Q-values for each action
Q_table = {}

# Q-table for clusterJ10_J7_J8
Q_table_cluster = {}

# ---- Additional Stability Parameters ----
MIN_GREEN_STEPS = 100

# Separate last switch timestamps for each traffic light
last_switch_step_J1 = -MIN_GREEN_STEPS
last_switch_step_J2 = -MIN_GREEN_STEPS

# -------------------------
# Step 7: Define Functions
# -------------------------

def get_max_Q1_value_of_state(s):  # Updated from get_max_Q_value_of_state
    if s not in Q_table:
        Q_table[s] = np.zeros(len(ACTIONS))
    return np.max(Q_table[s])

def get_max_Q2_value_of_state(s):  # Updated from get_max_Q_value_of_state_cluster
    if s not in Q_table_cluster:
        Q_table_cluster[s] = np.zeros(len(ACTIONS))
    return np.max(Q_table_cluster[s])

def get_reward(state):
    """
    Simple reward function:
    Negative of total queue length to encourage shorter queues.
    """
    total_queue = sum(state[:-1])  # Exclude the current_phase element
    reward = -float(total_queue)
    return reward

def agent1_get_state():
    global q1_EB_0, q1_EB_1, q1_EB_2, q1_SB_0, q1_SB_1, q1_SB_2, q1_NB_0, q1_NB_1, q1_NB_2, q1_WB_0, q1_WB_1, q1_WB_2, current_phase1

    # Detector IDs for Node1-2-EB
    detector_Node1_SB_0 = "Node1_SB_0"
    detector_Node1_SB_1 = "Node1_SB_1"
    detector_Node1_SB_2 = "Node1_SB_2"

    detector_Node1_EB_0 = "Node1_EB_0"
    detector_Node1_EB_1 = "Node1_EB_1"
    detector_Node1_EB_2 = "Node1_EB_2"

    # Detector IDs for Node1-NB
    detector_Node1_NB_0 = "Node1_NB_0"
    detector_Node1_NB_1 = "Node1_NB_1"
    detector_Node1_NB_2 = "Node1_NB_2"

    # Detector IDs for Node1-WB
    detector_Node1_WB_0 = "Node1_WB_0"
    detector_Node1_WB_1 = "Node1_WB_1"
    detector_Node1_WB_2 = "Node1_WB_2"

    # Traffic light ID
    traffic_light_id = "J1"

    # Get queue lengths from each detector
    q1_SB_0 = get_queue_length(detector_Node1_SB_0)
    q1_SB_1 = get_queue_length(detector_Node1_SB_1)
    q1_SB_2 = get_queue_length(detector_Node1_SB_2)

    q1_EB_0 = get_queue_length(detector_Node1_EB_0)
    q1_EB_1 = get_queue_length(detector_Node1_EB_1)
    q1_EB_2 = get_queue_length(detector_Node1_EB_2)

    q1_NB_0 = get_queue_length(detector_Node1_NB_0)
    q1_NB_1 = get_queue_length(detector_Node1_NB_1)
    q1_NB_2 = get_queue_length(detector_Node1_NB_2)

    q1_WB_0 = get_queue_length(detector_Node1_WB_0)
    q1_WB_1 = get_queue_length(detector_Node1_WB_1)
    q1_WB_2 = get_queue_length(detector_Node1_WB_2)

    # Get current phase index
    current_phase1 = get_current_phase(traffic_light_id)

    return (q1_EB_0, q1_EB_1, q1_EB_2, q1_SB_0, q1_SB_1, q1_SB_2,
            q1_NB_0, q1_NB_1, q1_NB_2, q1_WB_0, q1_WB_1, q1_WB_2, current_phase1)

def agent2_get_state():  # Previously get_state_cluster()
    global q2_SB_0, q2_SB_1, q2_SB_2, q2_EB_0, q2_EB_1, q2_EB_2, q2_NB_0, q2_NB_1, q2_NB_2, q2_WB_0, q2_WB_1, q2_WB_2, current_phase2

    # Detector IDs for Node2-EB
    detector_Node2_EB_0 = "Node2_EB_0"
    detector_Node2_EB_1 = "Node2_EB_1"
    detector_Node2_EB_2 = "Node2_EB_2"

    # Detector IDs for Node2-SB
    detector_Node2_SB_0 = "Node2_SB_0"
    detector_Node2_SB_1 = "Node2_SB_1"
    detector_Node2_SB_2 = "Node2_SB_2"

    # Detector IDs for Node2-NB
    detector_Node2_NB_0 = "Node2_NB_0"
    detector_Node2_NB_1 = "Node2_NB_1"
    detector_Node2_NB_2 = "Node2_NB_2"

    # Detector IDs for Node2-WB
    detector_Node2_WB_0 = "Node2_WB_0"
    detector_Node2_WB_1 = "Node2_WB_1"
    detector_Node2_WB_2 = "Node2_WB_2"

    # Traffic light ID
    traffic_light_id_cluster = "J2"

    # Get queue lengths from each detector
    q2_SB_0 = get_queue_length(detector_Node2_SB_0)
    q2_SB_1 = get_queue_length(detector_Node2_SB_1)
    q2_SB_2 = get_queue_length(detector_Node2_SB_2)

    q2_EB_0 = get_queue_length(detector_Node2_EB_0)
    q2_EB_1 = get_queue_length(detector_Node2_EB_1)
    q2_EB_2 = get_queue_length(detector_Node2_EB_2)

    q2_NB_0 = get_queue_length(detector_Node2_NB_0)
    q2_NB_1 = get_queue_length(detector_Node2_NB_1)
    q2_NB_2 = get_queue_length(detector_Node2_NB_2)

    q2_WB_0 = get_queue_length(detector_Node2_WB_0)
    q2_WB_1 = get_queue_length(detector_Node2_WB_1)
    q2_WB_2 = get_queue_length(detector_Node2_WB_2)

    # Get current phase index
    current_phase2 = get_current_phase(traffic_light_id_cluster)

    return (q2_SB_0, q2_SB_1, q2_SB_2, q2_EB_0, q2_EB_1, q2_EB_2,
            q2_NB_0, q2_NB_1, q2_NB_2, q2_WB_0, q2_WB_1, q2_WB_2, current_phase2)

def apply_action_J1(action, tls_id="J1"):  # Updated from apply_action()
    global last_switch_step_J1

    if action == 0:
        # Do nothing (keep current phase)
        return

    elif action == 1:
        # Check if minimum green time has passed before switching
        if current_simulation_step - last_switch_step_J1 >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases = len(program.phases)
            next_phase = (get_current_phase(tls_id) + 1) % num_phases
            traci.trafficlight.setPhase(tls_id, next_phase)
            last_switch_step_J1 = current_simulation_step

def apply_action_J2(action, tls_id="J2"):  # Updated from apply_action_cluster()
    global last_switch_step_J2

    if action == 0:
        # Do nothing (keep current phase)
        return

    elif action == 1:
        # Check if minimum green time has passed before switching
        if current_simulation_step - last_switch_step_J2 >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases = len(program.phases)
            next_phase = (get_current_phase(tls_id) + 1) % num_phases
            traci.trafficlight.setPhase(tls_id, next_phase)
            last_switch_step_J2 = current_simulation_step


def update_Q1_table(old_state, action, reward, new_state):  # Updated function
    if old_state not in Q_table:
        Q_table[old_state] = np.zeros(len(ACTIONS))
    
    # Predict current Q-values from old_state
    old_q = Q_table[old_state][action]
    # Predict Q-values for new_state to get max future Q
    best_future_q = get_max_Q1_value_of_state(new_state)  # Updated function call
    # Update Q-value
    Q_table[old_state][action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)

def update_Q2_table_cluster(old_state, action, reward, new_state):  # Updated function
    if old_state not in Q_table_cluster:
        Q_table_cluster[old_state] = np.zeros(len(ACTIONS))

    # Predict current Q-values from old_state
    old_q = Q_table_cluster[old_state][action]
    # Predict Q-values for new_state to get max future Q
    best_future_q = get_max_Q2_value_of_state(new_state)  # Updated function call
    # Update Q-value
    Q_table_cluster[old_state][action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)


def get_action_from_policy(state, Q_table):  # Added Q_table as a parameter
    if random.random() < EPSILON:
        return random.choice(ACTIONS)  # Exploration: choose a random action
    else:
        if state not in Q_table:
            Q_table[state] = np.zeros(len(ACTIONS))  # Initialize Q-values for unseen state
        return int(np.argmax(Q_table[state]))  # Exploitation: choose the best action


def get_queue_length(detector_id): #8.Constraint 8
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id): #8.Constraint 8
    return traci.trafficlight.getPhase(tls_id)

# -------------------------
# Step 8: Fully Online Continuous Learning Loop
# -------------------------

# Lists to record data for plotting
step_history = []
reward_history = []
queue_history = []

# Lists to record data for the second agent
step_history_cluster = []
reward_history_cluster = []
queue_history_cluster = []

cumulative_reward = 0.0

# Initialize cumulative reward for Agent 2
cumulative_reward_cluster = 0.0

print("\n=== Starting Fully Online Continuous Learning ===")
for step in range(TOTAL_STEPS):
    current_simulation_step = step

    # Agent 1 (J1)
    state = agent1_get_state()
    action = get_action_from_policy(state, Q_table)  # Pass Q_table for Agent 1
    apply_action_J1(action)

    new_state = agent1_get_state()
    reward = get_reward(new_state)
    update_Q1_table(state, action, reward, new_state)

    # Advance simulation
    traci.simulationStep()

    # Agent 2 (J2)
    state_cluster = agent2_get_state()
    action_cluster = get_action_from_policy(state_cluster, Q_table_cluster)  # Pass Q_table_cluster for Agent 2
    apply_action_J2(action_cluster)

    new_state_cluster = agent2_get_state()
    reward_cluster = get_reward(new_state_cluster)
    update_Q2_table_cluster(state_cluster, action_cluster, reward_cluster, new_state_cluster)

    # Incrementally update cumulative rewards
    cumulative_reward += reward
    cumulative_reward_cluster += reward_cluster  # Incremental update for Agent 2

    # Record data every step
    if step % 1 == 0:
        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(sum(new_state[:-1]))  # sum of queue lengths for J11

        step_history_cluster.append(step)
        reward_history_cluster.append(cumulative_reward_cluster)
        queue_history_cluster.append(sum(new_state_cluster[:-1]))  # sum of queue lengths for clusterJ10_J7_J8
     
# -------------------------
# Step 9: Close connection between SUMO and Traci
# -------------------------
traci.close()

# Print final Q-table info for Agent 1 (J1)
print("\nOnline Training completed. Final Q-table size for Agent 1 (J1):", len(Q_table))
for st, actions in Q_table.items():
    print("State (Agent 1):", st, "-> Q-values:", actions)

# Print final Q-table info for Agent 2 (J2)
print("\nOnline Training completed. Final Q-table size for Agent 2 (J2):", len(Q_table_cluster))
for st, actions in Q_table_cluster.items():
    print("State (Agent 2):", st, "-> Q-values:", actions)

# -------------------------
# Visualization of Results
# -------------------------

# Plot Cumulative Reward over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, marker='o', linestyle='-', label="Cumulative Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("RL Training: Cumulative Reward over Steps (Agent 1)")
plt.legend()
plt.grid(True) 
plt.show()

# Plot Total Queue Length over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_history, marker='o', linestyle='-', label="Total Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("RL Training: Queue Length over Steps")
plt.legend()
plt.grid(True)
plt.show()

# Plot Cumulative Reward over Simulation Steps for Agent 2 
plt.figure(figsize=(10, 6))
plt.plot(step_history_cluster, reward_history_cluster, marker='o', linestyle='-', label="Cumulative Reward (Agent 2)")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("RL Training: Cumulative Reward over Steps (Agent 2)")
plt.legend()
plt.grid(True)
plt.show()

# Plot Total Queue Length over Simulation Steps for Agent 2 
plt.figure(figsize=(10, 6))
plt.plot(step_history_cluster, queue_history_cluster, marker='o', linestyle='-', label="Total Queue Length (Agent 2)")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("RL Training: Queue Length over Steps (Agent 2)")
plt.legend()
plt.grid(True)
plt.show()
