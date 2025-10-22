# bev_repulsion_planner.py

import numpy as np

def apply_repulsive_force(initial_waypoints, obstacle_points, strength=0.5, radius=0.8):
    """
    Modifies a trajectory by applying a repulsive force from nearby obstacles.

    Args:
        initial_waypoints (np.ndarray): The original path from PlannerNet, shape (N, 2 or 3).
        obstacle_points (np.ndarray): A list of obstacle coordinates, shape (M, 2).
        strength (float): How strongly obstacles push waypoints away.
        radius (float): The maximum distance (in meters) at which an obstacle has an effect.

    Returns:
        np.ndarray: The modified, safer path.
    """
    if obstacle_points.shape[0] == 0:
        return initial_waypoints  # No obstacles, no change needed

    modified_waypoints = initial_waypoints.copy()
    
    # We only care about the (x, y) coordinates for repulsion
    waypoints_xy = initial_waypoints[:, :2]

    for i, wp in enumerate(waypoints_xy):
        total_repulsion_vector = np.zeros(2)

        # Calculate vector from all obstacles to the current waypoint
        vectors_to_wp = wp - obstacle_points
        distances = np.linalg.norm(vectors_to_wp, axis=1)

        # Filter for obstacles within the effective radius
        nearby_mask = (distances < radius) & (distances > 1e-6) # Avoid division by zero
        if not np.any(nearby_mask):
            continue

        nearby_vectors = vectors_to_wp[nearby_mask]
        nearby_distances = distances[nearby_mask]

        # Calculate force magnitude: increases quadratically as distance decreases
        # Force = strength * (1/distance - 1/radius)^2
        # This makes the force very strong up close and zero at the edge of the radius.
        magnitudes = strength * ((1.0 / nearby_distances) - (1.0 / radius))**2

        # Normalize the vectors to get direction, then apply magnitude
        directions = nearby_vectors / nearby_distances[:, np.newaxis]
        repulsion_vectors = directions * magnitudes[:, np.newaxis]

        # Sum all forces to get the total push on the waypoint
        total_repulsion_vector = np.sum(repulsion_vectors, axis=0)
        
        # Apply the push to the waypoint's (x, y) coordinates
        modified_waypoints[i, :2] += total_repulsion_vector

    return modified_waypoints
