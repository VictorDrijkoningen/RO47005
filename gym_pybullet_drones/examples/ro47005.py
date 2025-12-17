import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.ProjectEnvironment import ProjectEnvironment
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool


from rrt_star_planner import RRTStarPlanner

DEFAULT_SEED = 155
DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 400
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

# --- RRT / COLLISION HELPER FUNCTIONS ---

def check_line_collision(env, start_pos, end_pos, num_checks=10):
    """
    Checks if the straight line segment between two 3D points
    is collision-free using the environment's 3D occupancy map.
    """
    start_pos = np.array(start_pos)
    end_pos = np.array(end_pos)
    
    for i in range(1, num_checks + 1):
        t = i / num_checks
        point = start_pos + t * (end_pos - start_pos)
        if env.check_collision(point):
            return True 
    return False

def get_furthest_goal(env, z_height):

    """
    Finds a valid, collision-free goal inside the furthest room (ROWS-1, COLS-1).
    """
    # Hardcoded to match ProjectEnvironment settings
    ROWS = 3
    COLS = 3
    ROOM_W = 3.0
    
    # Target the top-right corner room indices
    target_r = ROWS - 1
    target_c = COLS - 1
    
    # Calculate the bounds of that specific room
    # (Remember: room center is at offset + index * width + width/2)
    # But since we centered the grid, we can calculate min/max simply:
    _, x_min, y_min, _, res = env.get_occupancy_map()

    # Calculate the min/max X and Y for just THIS room
    room_x_min = x_min + (target_c * ROOM_W)
    room_x_max = room_x_min + ROOM_W
    room_y_min = y_min + (target_r * ROOM_W)
    room_y_max = room_y_min + ROOM_W

    # Try 100 times to find a safe spot in this room
    for _ in range(100):
        rand_x = np.random.uniform(room_x_min + 0.5, room_x_max - 0.5)
        rand_y = np.random.uniform(room_y_min + 0.5, room_y_max - 0.5)
        
        candidate = np.array([rand_x, rand_y, z_height])
        
        if not env.check_collision(candidate):
            return candidate

    print("[CRITICAL WARNING] Could not find safe goal in furthest room! Using unsafe center.")
    # Fallback to center if the room is completely packed
    center_x = room_x_min + ROOM_W/2
    center_y = room_y_min + ROOM_W/2
    return np.array([center_x, center_y, z_height])

def run(
        seed=DEFAULT_SEED,
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB,
        ):
    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .3
    
    # Spawn drones centered at (0,0) which is now the first room
    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2), H+i*H_STEP] for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])


    path_planner_start_time = time.time()
    PERIOD = 20
    NUM_WP = control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP,3))
    
    # Currently just hovering/circling near start. 
    # YOU WILL REPLACE THIS WITH YOUR RRT PATH LATER
    for i in range(NUM_WP):
        TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 1], 0
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])

    

    #### Create the environment ################################
    env = ProjectEnvironment(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui,
                        seed = seed
                        )
    
    # --- SET CUSTOM CAMERA START POSITION ---
    p.resetDebugVisualizerCamera(
        cameraDistance=6.0,          # How far away the camera is
        cameraYaw=-90,               # Left/Right rotation
        cameraPitch=-40,             # Up/Down rotation (-90 is straight down)
        cameraTargetPosition=[4, 4, 0], # The point the camera looks at
        physicsClientId=env.getPyBulletClient()
    )

    # --- GOAL GENERATION ---
    goal_pos = get_furthest_goal(env, z_height=1.0)
    print(f"Goal set at furthest room: {goal_pos}")
    
    # === RRT* PLANNING ======================
    


    print("\n[RRT*] Starting path planning...")
    planner = RRTStarPlanner(max_iter=5000, goal_sample_rate=0.15, max_step=0.3)
    waypoints, success = planner.plan(env, start=INIT_XYZS[0], goal=goal_pos)

    if success:
        trajectory = planner.get_smooth_trajectory(waypoints, samples_per_segment=50)
        print(f"[RRT*] Trajectory: {len(trajectory)} points")
        print(f"[RRT*] First 3 waypoints:")
        for i in range(min(3, len(waypoints))):
            print(f"      [{i}] {waypoints[i]}")
        print(f"[RRT*] Last 3 waypoints:")
        for i in range(max(0, len(waypoints)-3), len(waypoints)):
            print(f"      [{i}] {waypoints[i]}")
    else:
        print("[RRT*] FAILED - using fallback")
        trajectory = TARGET_POS

    time_trajectory_calculation = time.time() - path_planner_start_time

    # --- VISUALIZE ---
    trajectory_length = -np.inf
    if success:
        # calculate the trajectory length 
        trajectory_length = 0
        for waypoint_idx in range(len(trajectory)-1):
            trajectory_length += np.linalg.norm(trajectory[waypoint_idx] -  trajectory[waypoint_idx+1])
        print("\nGenerating visualization...")
        planner.visualize_tree_and_path(waypoints)

    # Reset counter for trajectory following
    wp_counters = np.zeros(num_drones, dtype=int)
    traj_idx = 0
    # === END RRT* PLANNING ==========================


    # Visualize the goal with a duck
    # SPAWN THE DUCK 
    p.loadURDF("duck_vhacd.urdf", 
               goal_pos, 
               p.getQuaternionFromEuler([np.pi/2, 0, 0]), # Rotated 90 degrees around X-axis
               useFixedBase=True,
               physicsClientId=env.getPyBulletClient())
    
    # Define Endzone Box (Target Room Volume) for success check
    # +/- 1.5m around the goal point
    endzone_box = [
        goal_pos[0] - 1.5, goal_pos[0] + 1.5, 
        goal_pos[1] - 1.5, goal_pos[1] + 1.5, 
        0.1, 2.0
    ]
    # -----------------------

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )
    drones_in_endzone = [False,] * num_drones
    drones_in_endzone_times = [0,] * num_drones

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    action = np.zeros((num_drones, 4))
    START = time.time()
    
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control and Update Waypoints ###################
        for j in range(num_drones):
            
            # --- DRONE 0 (RRT* FOLLOWER) ---
            if j == 0 and success:
                # 1. Get current target
                current_wp_idx = int(wp_counters[j])
                
                if current_wp_idx < len(trajectory):
                    target_pos = trajectory[current_wp_idx]
                    
                    # 2. Calculate PROXIMITY (Fix for Runaway Setpoint)
                    current_pos = obs[j, :3]
                    dist_to_wp = np.linalg.norm(target_pos - current_pos)
                    
                    # 3. Only advance if close enough (e.g., 20cm)
                    if dist_to_wp < 0.2:
                        wp_counters[j] += 1
                        
                    # 4. Orientation logic (Face movement direction)
                    # Look ahead to the NEXT waypoint for smooth yaw
                    next_idx = min(current_wp_idx + 1, len(trajectory) - 1)
                    look_ahead_pos = trajectory[next_idx]
                    
                    direction = look_ahead_pos - current_pos
                    dist_dir = np.linalg.norm(direction)
                    
                    if dist_dir > 0.1: 
                        target_yaw = np.arctan2(direction[1], direction[0])
                        target_rpy = np.array([0.0, 0.0, target_yaw])
                    else:
                        target_rpy = np.array([0.0, 0.0, obs[j, 5]]) # Keep current yaw
                        
                else:
                    # End of path: Hover at last point
                    target_pos = trajectory[-1]
                    target_rpy = np.array([0.0, 0.0, 0.0])

            # --- OTHER DRONES (CIRCLE PATTERN) ---
            else:
                # Simple circular pattern updates every tick (or every N ticks)
                # Since this is a simple animation, unconditional increment is okay
                # IF the circular trajectory is designed for the control frequency.
                idx = wp_counters[j]
                target_pos = TARGET_POS[idx, 0:3]
                target_rpy = INIT_RPYS[j, :]
                
                # Increment circular counter, loop back
                wp_counters[j] = (wp_counters[j] + 1) % NUM_WP

            # --- COMPUTE CONTROL ---
            action[j, :], _, _ = ctrl[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=target_pos,
                target_rpy=target_rpy
            )


        #### Log the simulation ####################################
        for j in range(num_drones):
            # Use modulo to safely index TARGET_POS
            safe_idx = int(wp_counters[j]) % NUM_WP
            
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack([TARGET_POS[safe_idx, 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                       )
            
            # Check if drone is in endzone
            if (obs[j, 0] > endzone_box[0] and obs[j,0] < endzone_box[1] and 
                obs[j, 1] > endzone_box[2] and obs[j,1] < endzone_box[3] and 
                obs[j, 2] > endzone_box[4] and obs[j,2] < endzone_box[5]):
                
                if not drones_in_endzone[j]:
                    print(f"Drone {j} reached the end zone!")
                    drones_in_endzone_times[j] = time.time()-START

                drones_in_endzone[j] = True
                if j == 0 and drones_in_endzone[0]:
                    print(f"✓ TARGET REACHED in {drones_in_endzone_times[0]:.2f} seconds!")
        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

        # Stop simulation if drone 0 reached goal 
        if success and drones_in_endzone[0]:
            print("\nStopping simulation - goal reached.")
            break
    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()
    return {"time_trajectory_calculation" : time_trajectory_calculation, 
            "trajectory_length":trajectory_length,
            "drones_in_endzone_times" : drones_in_endzone_times,
            "drones_in_endzone" : drones_in_endzone}

if __name__ == "__main__":
    output = run(155)

    print(f"Computational time for trajectory planner: {output['time_trajectory_calculation']}")
    print(f"Trajectory length: {output['trajectory_length']}")
    print(f"Drone in endzone took: {output['drones_in_endzone_times']}")
    print(f"Drone success: {output['drones_in_endzone']}")