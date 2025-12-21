import json
import numpy as np

file_name = "runner.json"

with open(file_name, mode="r") as f:
    data = json.loads(f.read())


# Amount of times drone env simulated
runs= len(data)

# Average computation time for planning
total_computation_time = 0
for run in data:
    total_computation_time += run['time_trajectory_calculation']

avg_computation_time = total_computation_time/runs


# Average Trajectory length
total_trajectory_length = 0
trajectory_runs = 0
for run in data:
    if run['drones_in_endzone'][0]:
        trajectory_runs += 1
        total_trajectory_length += run['trajectory_length']
avg_trajectory_length = total_trajectory_length/trajectory_runs


# Average Time to reach goal
total_time_to_reach_goal = 0
for run in data:
    total_time_to_reach_goal += run['drones_in_endzone_times'][0]

avg_time_to_reach_goal = total_time_to_reach_goal/runs


# Success rate
successes = 0
for thing in data:
    if thing['drones_in_endzone'][0]:
        successes += 1
    # else:
    #     print(thing['seed'])

success_rate = successes / runs


print(f"DATA FILE '{file_name}' METRICS:")
print(f"Average computation time  : {np.round(avg_computation_time,2)}")
print(f"Average trajectory length : {np.round(avg_trajectory_length,2)}")
print(f"Average time to reach_goal: {np.round(avg_time_to_reach_goal,2)}")
print(f"Success rate              : {np.round(success_rate,2)}")

