import os
import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

class ProjectEnvironment(CtrlAviary):
    def _addObstacles(self):
        """
        Builds a deterministic grid with a GUARANTEE that all rooms are reachable.
        Uses a 'Union-Find' or Flood-Fill approach to merge isolated room clusters.
        """
        # --- 1. Set Fixed Seed ---
        np.random.seed(5) 

        # --- Configuration ---
        ROWS, COLS = 3, 3
        ROOM_W = 3.0
        WALL_H = 2.0
        WALL_T = 0.1
        DOOR_W = 1.0
        NUM_DEBRIS = 10
        MIN_WALL_LEN = 0.2
        
        offset_x = -ROOM_W / 2
        offset_y = -ROOM_W / 2

        # --- Helper: Create Box ---
        def create_box(pos, dims, color=[0.5, 0.5, 0.5, 1]):
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=dims, rgbaColor=color)
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=dims)
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)

        # --- 2. PRE-CALCULATE WALL STATES ---
        h_walls = np.zeros((ROWS + 1, COLS), dtype=int)
        v_walls = np.zeros((COLS + 1, ROWS), dtype=int)

        # A. Initial Random Fill (High Solid Probability)
        for r in range(ROWS + 1):
            for c in range(COLS):
                if r == 0 or r == ROWS: h_walls[r, c] = 0   
                else: 
                    rand = np.random.random()
                    if rand < 0.2: h_walls[r, c] = 2   # Open
                    elif rand < 0.3: h_walls[r, c] = 1 # Door
                    else: h_walls[r, c] = 0            # Solid

        for c in range(COLS + 1):
            for r in range(ROWS):
                if c == 0 or c == COLS: v_walls[c, r] = 0   
                else: 
                    rand = np.random.random()
                    if rand < 0.2: v_walls[c, r] = 2
                    elif rand < 0.3: v_walls[c, r] = 1
                    else: v_walls[c, r] = 0

        # B. GLOBAL CONNECTIVITY CHECK
        # We loop until the 'flood fill' can reach every single room from (0,0)
        while True:
            # 1. Flood Fill to find reachable rooms
            reachable = np.zeros((ROWS, COLS), dtype=bool)
            stack = [(0,0)] # Start at room 0,0
            reachable[0,0] = True
            count = 0
            
            while stack:
                r, c = stack.pop()
                count += 1
                
                # Check North (r+1)
                if r < ROWS - 1:
                    # If wall is NOT solid (it's 1 or 2), we can pass
                    if h_walls[r+1, c] != 0 and not reachable[r+1, c]:
                        reachable[r+1, c] = True
                        stack.append((r+1, c))
                # Check South (r) - Note: h_walls index for south of room r is r
                if r > 0:
                    if h_walls[r, c] != 0 and not reachable[r-1, c]:
                        reachable[r-1, c] = True
                        stack.append((r-1, c))
                # Check East (c+1)
                if c < COLS - 1:
                    if v_walls[c+1, r] != 0 and not reachable[r, c+1]:
                        reachable[r, c+1] = True
                        stack.append((r, c+1))
                # Check West (c)
                if c > 0:
                    if v_walls[c, r] != 0 and not reachable[r, c-1]:
                        reachable[r, c-1] = True
                        stack.append((r, c-1))

            # 2. Check if done
            if count == ROWS * COLS:
                break # All rooms are connected!

            # 3. If not, find a wall that connects a REACHABLE room to an UNREACHABLE room
            candidates = []
            for r in range(ROWS):
                for c in range(COLS):
                    if reachable[r, c]:
                        # Look at neighbors. If neighbor is unreachable AND wall is solid, it's a candidate to open.
                        # North
                        if r < ROWS - 1 and not reachable[r+1, c]:
                            candidates.append(('h', r+1, c))
                        # South
                        if r > 0 and not reachable[r-1, c]:
                            candidates.append(('h', r, c))
                        # East
                        if c < COLS - 1 and not reachable[r, c+1]:
                            candidates.append(('v', c+1, r))
                        # West
                        if c > 0 and not reachable[r, c-1]:
                            candidates.append(('v', c, r))
            
            # 4. Open one random candidate wall
            if candidates:
                # Pick random wall from candidates
                idx = np.random.randint(len(candidates))
                ctype, cr, cc = candidates[idx]
                
                # Force it to be a Door (1)
                if ctype == 'h': h_walls[cr, cc] = 1
                else: v_walls[cr, cc] = 1
            else:
                # Should not happen unless grid size 1x1
                break

        # --- 3. DRAW THE WALLS ---
        
        # Draw Horizontal Walls
        for r in range(ROWS + 1):
            for c in range(COLS):
                state = h_walls[r, c]
                if state == 2: continue 

                cx = offset_x + c * ROOM_W + ROOM_W / 2
                cy = offset_y + r * ROOM_W
                
                if state == 1: # Door
                    total_solid_len = ROOM_W - DOOR_W
                    len1 = np.random.uniform(MIN_WALL_LEN, total_solid_len - MIN_WALL_LEN)
                    len2 = total_solid_len - len1
                    pos1 = (cx - ROOM_W/2) + len1/2
                    pos2 = (cx + ROOM_W/2) - len2/2
                    create_box([pos1, cy, WALL_H/2], [len1/2, WALL_T/2, WALL_H/2])
                    create_box([pos2, cy, WALL_H/2], [len2/2, WALL_T/2, WALL_H/2])
                else: # Solid
                    create_box([cx, cy, WALL_H/2], [ROOM_W/2, WALL_T/2, WALL_H/2])

        # Draw Vertical Walls
        for c in range(COLS + 1):
            for r in range(ROWS):
                state = v_walls[c, r]
                if state == 2: continue 

                cx = offset_x + c * ROOM_W
                cy = offset_y + r * ROOM_W + ROOM_W / 2
                
                if state == 1: # Door
                    total_solid_len = ROOM_W - DOOR_W
                    len1 = np.random.uniform(MIN_WALL_LEN, total_solid_len - MIN_WALL_LEN)
                    len2 = total_solid_len - len1
                    pos1 = (cy - ROOM_W/2) + len1/2
                    pos2 = (cy + ROOM_W/2) - len2/2
                    create_box([cx, pos1, WALL_H/2], [WALL_T/2, len1/2, WALL_H/2])
                    create_box([cx, pos2, WALL_H/2], [WALL_T/2, len2/2, WALL_H/2])
                else: # Solid
                    create_box([cx, cy, WALL_H/2], [WALL_T/2, ROOM_W/2, WALL_H/2])

        # 3. CEILING
        grid_w = COLS * ROOM_W
        grid_h = ROWS * ROOM_W
        center_x = offset_x + grid_w / 2
        center_y = offset_y + grid_h / 2
        create_box([center_x, center_y, WALL_H], [grid_w/2, grid_h/2, 0.05], color=[0.8, 0.8, 0.8, 0.3]) 

        # 4. STATIC DEBRIS
        env_dir = os.path.dirname(os.path.abspath(__file__))
        asset_path = os.path.join(env_dir, "../assets/box.urdf")

        for _ in range(NUM_DEBRIS):
            r = np.random.randint(0, ROWS)
            c = np.random.randint(0, COLS)
            
            room_center_x = offset_x + c * ROOM_W + ROOM_W / 2
            room_center_y = offset_y + r * ROOM_W + ROOM_W / 2
            
            safe_margin = (ROOM_W / 2) - 0.5 
            dx = np.random.uniform(-safe_margin, safe_margin)
            dy = np.random.uniform(-safe_margin, safe_margin)
            rand_scale = np.random.uniform(7.0, 15.0) 

            p.loadURDF(asset_path,
                       [room_center_x + dx, room_center_y + dy, 0.5],
                       p.getQuaternionFromEuler([0, 0, np.random.uniform(0, 3.14)]),
                       useFixedBase=True,
                       globalScaling=rand_scale,
                       physicsClientId=self.CLIENT)