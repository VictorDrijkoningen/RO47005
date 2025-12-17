# ==============================================================================
# RRT* PLANNER 
# ==============================================================================

import numpy as np


class RRTStarPlanner:
    """
    RRT* motion planning algorithm for 3D space.
    """
    
    def __init__(self, max_iter=5000, goal_sample_rate=0.15, max_step=0.3):
        """
        Args:
            max_iter: Maximum iterations
            goal_sample_rate: Probability to sample goal directly
            max_step: Maximum steering distance
        """
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.max_step = max_step
        
        # For visualization
        self.tree_nodes = []
        self.tree_edges = []
    
    
    def sample_config(self, bounds, goal):
        """SAMPLE: Random configuration in C-space"""
        if np.random.random() < self.goal_sample_rate:
            return np.array(goal, dtype=float)
        
        q_rand = np.array([
            np.random.uniform(bounds['x'][0], bounds['x'][1]),
            np.random.uniform(bounds['y'][0], bounds['y'][1]),
            np.random.uniform(bounds['z'][0], bounds['z'][1])
        ], dtype=float)
        return q_rand
    
    
    def distance(self, q1, q2):
        """METRIC: Euclidean distance in C-space"""
        return np.linalg.norm(np.array(q2, dtype=float) - np.array(q1, dtype=float))
    
    
    def steer(self, q_from, q_toward):
        """STEER: Move from q_from toward q_toward by max_step"""
        q_from = np.array(q_from, dtype=float)
        q_toward = np.array(q_toward, dtype=float)
        
        direction = q_toward - q_from
        dist = self.distance(q_from, q_toward)
        
        if dist < 1e-9:
            return q_toward.copy()
        
        # Normalize and limit step
        direction = direction / dist
        step_dist = min(dist, self.max_step)
        q_new = q_from + step_dist * direction
        
        return q_new
    
    
    def is_collision_free(self, env, q1, q2, num_samples=20):
        """COLLISION_FREE: Check if line segment is collision-free"""
        q1 = np.array(q1, dtype=float)
        q2 = np.array(q2, dtype=float)
        
        # Sample points along the line segment
        for i in range(num_samples + 1):
            t = i / num_samples
            q_sample = q1 + t * (q2 - q1)
            
            if env.check_collision(q_sample):
                return False  # Collision!
        
        return True  # All samples clear
    
    
    def nearest(self, q_rand, nodes):
        """NEAREST: Find closest node in tree"""
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, node in enumerate(nodes):
            dist = self.distance(q_rand, node)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx
    
    
    def near(self, q_new, nodes, r):
        """NEAR: Find all nodes within radius r"""
        neighbors = []
        for i, node in enumerate(nodes):
            dist = self.distance(q_new, node)
            if 0 < dist < r:  # Must be > 0 to exclude self
                neighbors.append(i)
        return neighbors
    
    
    def radius(self, n):
        """RRT* radius: r = c * sqrt(ln(n) / n)"""
        if n < 2:
            return 2.0
        # For 3D, typical c is 2**(1/3) * sqrt(pi) ≈ 2.5
        return min(2.0, 2.5 * np.sqrt(np.log(n) / n))
    
    
    def plan(self, env, start, goal):
        """
        RRT* ALGORITHM 
        
        Returns:
            waypoints: [start, ..., goal] path
            success: True if path found
        """
        
        start = np.array(start, dtype=float)
        goal = np.array(goal, dtype=float)
        
        # Get environment bounds from occupancy map
        occ_map, x_min, y_min, z_min, resolution = env.get_occupancy_map()
        rows, cols, height = occ_map.shape
        
        bounds = {
            'x': [x_min, x_min + cols * resolution],
            'y': [y_min, y_min + rows * resolution],
            'z': [z_min, z_min + height * resolution]
        }
        
        print(f"\n{'='*70}")
        print("RRT* ALGORITHM")
        print(f"{'='*70}")
        print(f"Start: {start}")
        print(f"Goal: {goal}")
        print(f"Bounds X: [{bounds['x'][0]:.2f}, {bounds['x'][1]:.2f}]")
        print(f"Bounds Y: [{bounds['y'][0]:.2f}, {bounds['y'][1]:.2f}]")
        print(f"Bounds Z: [{bounds['z'][0]:.2f}, {bounds['z'][1]:.2f}]")
        print(f"{'='*70}\n")
        
        # Initialize tree with start node
        nodes = [start.copy()]
        costs = [0.0]  # cost from start
        parents = [-1]  # parent index (-1 = start has no parent)
        
        best_goal_idx = -1
        best_goal_cost = float('inf')
        
        # ========== MAIN RRT* LOOP ==========
        for iteration in range(self.max_iter):
            
            # Progress output
            if iteration % 500 == 0:
                status = f"Iter {iteration:5d}/{self.max_iter} | Nodes: {len(nodes):5d}"
                if best_goal_idx != -1:
                    status += f" | Goal cost: {best_goal_cost:.2f}m"
                print(status)
            
            # ===== STEP 1: SAMPLE =====
            q_rand = self.sample_config(bounds, goal)
            
            # ===== STEP 2: NEAREST =====
            nearest_idx = self.nearest(q_rand, nodes)
            q_near = nodes[nearest_idx]
            
            # ===== STEP 3: STEER =====
            q_new = self.steer(q_near, q_rand)
            
            # ===== STEP 4: COLLISION CHECK =====
            if not self.is_collision_free(env, q_near, q_new, num_samples=20):
                continue  # Skip this sample
            
            # ===== STEP 5: NEAR =====
            r = self.radius(len(nodes))
            neighbor_indices = self.near(q_new, nodes, r)
            
            # ===== STEP 6: CHOOSE PARENT =====
            # Find best parent from neighbors
            best_parent_idx = nearest_idx
            best_cost_to_new = costs[nearest_idx] + self.distance(q_near, q_new)
            
            for neighbor_idx in neighbor_indices:
                q_neighbor = nodes[neighbor_idx]
                
                # Check if connection is collision-free
                if not self.is_collision_free(env, q_neighbor, q_new, num_samples=20):
                    continue
                
                # Calculate cost through this neighbor
                cost_via_neighbor = costs[neighbor_idx] + self.distance(q_neighbor, q_new)
                
                # Update if cheaper
                if cost_via_neighbor < best_cost_to_new:
                    best_parent_idx = neighbor_idx
                    best_cost_to_new = cost_via_neighbor
            
            # ===== STEP 7: ADD NODE TO TREE =====
            nodes.append(q_new.copy())
            costs.append(best_cost_to_new)
            parents.append(best_parent_idx)
            
            new_node_idx = len(nodes) - 1
            
            # Store for visualization
            self.tree_nodes.append(q_new.copy())
            self.tree_edges.append((q_near.copy(), q_new.copy()))
            
            # ===== STEP 8: REWIRE =====
            # Update neighbors if we offer a better parent
            for neighbor_idx in neighbor_indices:
                q_neighbor = nodes[neighbor_idx]
                
                # Check if connection is collision-free
                if not self.is_collision_free(env, q_new, q_neighbor, num_samples=20):
                    continue
                
                # Cost if we reroute through new node
                cost_via_new = best_cost_to_new + self.distance(q_new, q_neighbor)
                
                # Rewire if cheaper
                if cost_via_new < costs[neighbor_idx]:
                    # 1. Calculate reduction
                    cost_reduction = costs[neighbor_idx] - cost_via_new
                    
                    # 2. Update immediate neighbor
                    parents[neighbor_idx] = new_node_idx
                    costs[neighbor_idx] = cost_via_new
                    
                    # 3. PROPAGATE cost reduction to all descendants (BFS)
                    # This ensures the whole subtree benefits from the shortcut
                    stack = [neighbor_idx]
                    while stack:
                        curr_idx = stack.pop()
                        # Find all nodes that have 'curr_idx' as parent
                        # (Note: This linear search is slow for huge trees, 
                        #  but fine for Python RRT demo < 5000 nodes)
                        children = [i for i, p in enumerate(parents) if p == curr_idx]
                        
                        for child in children:
                            costs[child] -= cost_reduction
                            stack.append(child)
            
            # ===== STEP 9: CHECK GOAL =====
            dist_to_goal = self.distance(q_new, goal)
            
            if dist_to_goal <= self.max_step:
                # Can we connect directly to goal?
                if self.is_collision_free(env, q_new, goal, num_samples=20):
                    goal_cost = best_cost_to_new + dist_to_goal
                    
                    if goal_cost < best_goal_cost:
                        best_goal_idx = new_node_idx
                        best_goal_cost = goal_cost
                        print(f"  [GOAL FOUND] Cost: {goal_cost:.2f}m at iteration {iteration}")
        
        # ========== EXTRACT PATH ==========
        print(f"\n{'='*70}")
        
        if best_goal_idx == -1:
            print("✗ NO PATH FOUND - Increase iterations or max_step")
            print(f"{'='*70}\n")
            return None, False
        
        # Backtrack from goal to start
        waypoints = []
        current_idx = best_goal_idx
        
        while current_idx != -1:
            waypoints.append(nodes[current_idx].copy())
            current_idx = parents[current_idx]
        
        waypoints.reverse()
        waypoints.append(goal.copy())
        
        print(f"✓ PATH FOUND!")
        print(f"  Path cost: {best_goal_cost:.2f}m")
        print(f"  Waypoints: {len(waypoints)}")
        print(f"  Tree nodes: {len(nodes)}")
        print(f"{'='*70}\n")
        
        return waypoints, True
    
    
    def get_smooth_trajectory(self, waypoints, samples_per_segment=50):
        """
        Interpolate waypoints to dense trajectory.
        Linear interpolation between consecutive waypoints.
        """
        if waypoints is None or len(waypoints) < 2:
            return None
         
        trajectory = []
        
        for i in range(len(waypoints) - 1):
            start = np.array(waypoints[i], dtype=float)
            end = np.array(waypoints[i + 1], dtype=float)
            
            # Linear interpolation
            for t in np.linspace(0, 1, samples_per_segment, endpoint=False):
                point = start + t * (end - start)
                trajectory.append(point)
        
        # Add final waypoint
        trajectory.append(np.array(waypoints[-1], dtype=float))
        
        return np.array(trajectory)
    
# WAS ADDED FOR VISUALIZATION PURPOSES, CAN BE REMOVED IF NOT NEEDED
    def visualize_tree_and_path(self, waypoints, title="RRT* Planning Result"):
        """Visualize RRT* tree and final path"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        if waypoints is None:
            print("Cannot visualize - no path found")
            return
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot tree edges
        for edge in self.tree_edges:
            start, end = edge
            ax.plot([start[0], end[0]], 
                    [start[1], end[1]], 
                    [start[2], end[2]], 
                    'gray', alpha=0.2, linewidth=0.5)
        
        # Plot tree nodes
        if self.tree_nodes:
            tree_array = np.array(self.tree_nodes)
            ax.scatter(tree_array[:, 0], tree_array[:, 1], tree_array[:, 2],
                    color='lightblue', s=10, alpha=0.5, label='Tree nodes')
        
        # Plot final path in BOLD RED
        waypoints_array = np.array(waypoints)
        ax.plot(waypoints_array[:, 0], waypoints_array[:, 1], waypoints_array[:, 2],
                'r-', linewidth=4, label='Final path', alpha=0.9)
        
        # Plot start and goal
        ax.scatter(*waypoints[0], color='green', s=300, marker='o', 
                label='Start', edgecolors='black', linewidths=2, zorder=5)
        ax.scatter(*waypoints[-1], color='red', s=300, marker='*', 
                label='Goal', edgecolors='black', linewidths=2, zorder=5)
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.ion()
        plt.show(block=True)
        # plt.pause(0.5)
