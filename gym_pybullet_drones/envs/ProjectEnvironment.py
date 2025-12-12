import os
import math
import numpy as np
import pybullet as p
import pkg_resources
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

class ProjectEnvironment(CtrlAviary):
    def _addObstacles(self):
        """
        Builds the environment and generates a 3D Occupancy Grid (Voxel Map).
        """
        np.random.seed(3) 

        # --- Configuration ---
        ROWS, COLS = 3, 3
        ROOM_W = 3.0
        WALL_H = 2.0
        WALL_T = 0.1
        DOOR_W = 1.0
        NUM_DEBRIS = 10
        MIN_WALL_LEN = 0.2
        
        # 3D Grid Settings
        self.RESOLUTION = 0.1  # 1 voxel = 10cm x 10cm x 10cm
        
        # World Bounds
        world_width = COLS * ROOM_W
        world_height = ROWS * ROOM_W
        self.x_min = -ROOM_W / 2
        self.y_min = -ROOM_W / 2
        self.z_min = 0.0
        self.z_max = WALL_H  # Ceiling height
        
        # Initialize 3D Matrix (0 = Free, 1 = Occupied)
        grid_rows = int(math.ceil(world_height / self.RESOLUTION))
        grid_cols = int(math.ceil(world_width / self.RESOLUTION))
        grid_height = int(math.ceil((self.z_max - self.z_min) / self.RESOLUTION))
        
        # Shape: [Y, X, Z]
        self.occupancy_map = np.zeros((grid_rows, grid_cols, grid_height), dtype=int)

        offset_x = -ROOM_W / 2
        offset_y = -ROOM_W / 2

        # --- Helper: Update 3D Matrix ---
        def mark_matrix(lower_bound, upper_bound):
            # Convert world coords to grid indices
            # X / Col
            c_min = int((lower_bound[0] - self.x_min) / self.RESOLUTION)
            c_max = int((upper_bound[0] - self.x_min) / self.RESOLUTION)
            # Y / Row
            r_min = int((lower_bound[1] - self.y_min) / self.RESOLUTION)
            r_max = int((upper_bound[1] - self.y_min) / self.RESOLUTION)
            # Z / Height
            z_min_idx = int((lower_bound[2] - self.z_min) / self.RESOLUTION)
            z_max_idx = int((upper_bound[2] - self.z_min) / self.RESOLUTION)

            # Clip to array bounds
            c_min = max(0, c_min); c_max = min(grid_cols, c_max)
            r_min = max(0, r_min); r_max = min(grid_rows, r_max)
            z_min_idx = max(0, z_min_idx); z_max_idx = min(grid_height, z_max_idx)

            # Mark voxel volume as occupied
            self.occupancy_map[r_min:r_max, c_min:c_max, z_min_idx:z_max_idx] = 1

        # --- Helper: Create Box & Mark ---
        def create_box(pos, dims, color=[0.5, 0.5, 0.5, 1]):
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=dims, rgbaColor=color)
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=dims)
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)
            
            # Calculate AABB for 3D marking
            lower = [pos[0] - dims[0], pos[1] - dims[1], pos[2] - dims[2]]
            upper = [pos[0] + dims[0], pos[1] + dims[1], pos[2] + dims[2]]
            mark_matrix(lower, upper)

        # --- PRE-CALCULATE WALLS (Maze Logic) ---
        h_walls = np.zeros((ROWS + 1, COLS), dtype=int)
        v_walls = np.zeros((COLS + 1, ROWS), dtype=int)

        # Random Fill
        for r in range(ROWS + 1):
            for c in range(COLS):
                if r == 0 or r == ROWS: h_walls[r, c] = 0   
                else: 
                    rand = np.random.random()
                    if rand < 0.2: h_walls[r, c] = 2   
                    elif rand < 0.3: h_walls[r, c] = 1 
                    else: h_walls[r, c] = 0            

        for c in range(COLS + 1):
            for r in range(ROWS):
                if c == 0 or c == COLS: v_walls[c, r] = 0   
                else: 
                    rand = np.random.random()
                    if rand < 0.2: v_walls[c, r] = 2
                    elif rand < 0.3: v_walls[c, r] = 1
                    else: v_walls[c, r] = 0

        # Connectivity Check
        while True:
            reachable = np.zeros((ROWS, COLS), dtype=bool)
            stack = [(0,0)]; reachable[0,0] = True; count = 0
            while stack:
                r, c = stack.pop(); count += 1
                if r < ROWS - 1 and h_walls[r+1, c] != 0 and not reachable[r+1, c]:
                    reachable[r+1, c] = True; stack.append((r+1, c))
                if r > 0 and h_walls[r, c] != 0 and not reachable[r-1, c]:
                    reachable[r-1, c] = True; stack.append((r-1, c))
                if c < COLS - 1 and v_walls[c+1, r] != 0 and not reachable[r, c+1]:
                    reachable[r, c+1] = True; stack.append((r, c+1))
                if c > 0 and v_walls[c, r] != 0 and not reachable[r, c-1]:
                    reachable[r, c-1] = True; stack.append((r, c-1))

            if count == ROWS * COLS: break 

            candidates = []
            for r in range(ROWS):
                for c in range(COLS):
                    if reachable[r, c]:
                        if r < ROWS - 1 and not reachable[r+1, c]: candidates.append(('h', r+1, c))
                        if r > 0 and not reachable[r-1, c]: candidates.append(('h', r, c))
                        if c < COLS - 1 and not reachable[r, c+1]: candidates.append(('v', c+1, r))
                        if c > 0 and not reachable[r, c-1]: candidates.append(('v', c, r))
            if candidates:
                idx = np.random.randint(len(candidates))
                ctype, cr, cc = candidates[idx]
                if ctype == 'h': h_walls[cr, cc] = 1
                else: v_walls[cr, cc] = 1
            else: break

        # --- DRAW WALLS ---
        for r in range(ROWS + 1):
            for c in range(COLS):
                state = h_walls[r, c]
                if state == 2: continue 
                cx = offset_x + c * ROOM_W + ROOM_W / 2
                cy = offset_y + r * ROOM_W
                if state == 1: 
                    len1 = np.random.uniform(MIN_WALL_LEN, ROOM_W - DOOR_W - MIN_WALL_LEN)
                    len2 = ROOM_W - DOOR_W - len1
                    create_box([(cx - ROOM_W/2) + len1/2, cy, WALL_H/2], [len1/2, WALL_T/2, WALL_H/2])
                    create_box([(cx + ROOM_W/2) - len2/2, cy, WALL_H/2], [len2/2, WALL_T/2, WALL_H/2])
                else:
                    create_box([cx, cy, WALL_H/2], [ROOM_W/2, WALL_T/2, WALL_H/2])

        for c in range(COLS + 1):
            for r in range(ROWS):
                state = v_walls[c, r]
                if state == 2: continue 
                cx = offset_x + c * ROOM_W
                cy = offset_y + r * ROOM_W + ROOM_W / 2
                if state == 1: 
                    len1 = np.random.uniform(MIN_WALL_LEN, ROOM_W - DOOR_W - MIN_WALL_LEN)
                    len2 = ROOM_W - DOOR_W - len1
                    create_box([cx, (cy - ROOM_W/2) + len1/2, WALL_H/2], [WALL_T/2, len1/2, WALL_H/2])
                    create_box([cx, (cy + ROOM_W/2) - len2/2, WALL_H/2], [WALL_T/2, len2/2, WALL_H/2])
                else:
                    create_box([cx, cy, WALL_H/2], [WALL_T/2, ROOM_W/2, WALL_H/2])

        # 3. CEILING (Now marked in the matrix!)
        grid_w = COLS * ROOM_W
        grid_h = ROWS * ROOM_W
        center_x = offset_x + grid_w / 2
        center_y = offset_y + grid_h / 2
        create_box([center_x, center_y, WALL_H], [grid_w/2, grid_h/2, 0.05], color=[0.8, 0.8, 0.8, 0.3])

        # 4. FURNITURE
        env_dir = os.path.dirname(os.path.abspath(__file__))
        furniture_dir = os.path.join(env_dir, "../assets/furniture")
        try:
            furniture_files = [f for f in os.listdir(furniture_dir) if f.endswith('.urdf')]
        except FileNotFoundError:
            furniture_files = []

        for _ in range(NUM_DEBRIS):
            r = np.random.randint(0, ROWS)
            c = np.random.randint(0, COLS)
            room_center_x = offset_x + c * ROOM_W + ROOM_W / 2
            room_center_y = offset_y + r * ROOM_W + ROOM_W / 2
            dx = np.random.uniform(-(ROOM_W/2 - 0.6), (ROOM_W/2 - 0.6))
            dy = np.random.uniform(-(ROOM_W/2 - 0.6), (ROOM_W/2 - 0.6))
            
            if furniture_files:
                asset_path = os.path.join(furniture_dir, np.random.choice(furniture_files))
                rand_scale = np.random.uniform(0.8, 1.2)
            else:
                asset_path = os.path.join(env_dir, "../assets/box.urdf")
                rand_scale = np.random.uniform(5.0, 10.0)

            body_id = p.loadURDF(asset_path,
                       [room_center_x + dx, room_center_y + dy, 0.0],
                       p.getQuaternionFromEuler([0, 0, np.random.uniform(0, 3.14)]),
                       useFixedBase=True,
                       globalScaling=rand_scale,
                       physicsClientId=self.CLIENT)
            
            # Use PyBullet AABB to mark the 3D volume of the furniture
            aabb_min, aabb_max = p.getAABB(body_id, physicsClientId=self.CLIENT)
            mark_matrix(aabb_min, aabb_max)

    def get_occupancy_map(self):
        return self.occupancy_map, self.x_min, self.y_min, self.z_min, self.RESOLUTION

    def check_collision(self, position, safety_radius=0.05):
        """
        3D Collision Check with safety margin.
        Checks sphere around position instead of single point.
        """
        x, y, z = position[0], position[1], position[2]
        
        # Check a small radius around the position for safety
        # This prevents getting stuck on voxel boundaries
        check_offsets = [
            (0, 0, 0),  # Center
            (safety_radius, 0, 0), (-safety_radius, 0, 0),
            (0, safety_radius, 0), (0, -safety_radius, 0),
            (0, 0, safety_radius), (0, 0, -safety_radius)
        ]
        
        for dx, dy, dz in check_offsets:
            cx, cy, cz = x + dx, y + dy, z + dz
            
            # Grid Indices
            c = int((cx - self.x_min) / self.RESOLUTION)
            r = int((cy - self.y_min) / self.RESOLUTION)
            k = int((cz - self.z_min) / self.RESOLUTION)
            
            # Bounds Check
            rows, cols, height = self.occupancy_map.shape
            if r < 0 or r >= rows or c < 0 or c >= cols or k < 0 or k >= height:
                return True  # Out of bounds
            
            # Voxel Check
            if self.occupancy_map[r, c, k] == 1:
                return True  # Hit obstacle
        
        return False