#!/usr/bin/env python3
"""
Semantic BEV Map Generator Node
Subscribes to a semantic point cloud and generates a 2D BEV map
for Rviz visualization.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
import struct
import time

class SemanticBEVNode(Node):
    """
    Generates a 2D BEV map from a 3D semantic point cloud.
    
    Each cell in the BEV map is colored according to the semantic
    label of the highest point within that cell's 3D column.
    """
    
    def __init__(self):
        super().__init__('semantic_bev_node')
        
        # ---  tunable ROS Parameters ---
        self.declare_parameter('z_min', 0.15)
        self.declare_parameter('z_max', 1.0)
        self.declare_parameter('grid_resolution', 0.1) # meters per cell
        self.declare_parameter('grid_size_x', 30.0)    # total width in meters
        self.declare_parameter('grid_size_y', 30.0)    # total height in meters
        # self.declare_parameter('input_topic', '/semantic_pointcloud')
        self.declare_parameter('input_topic', '/pointcloud')
        self.declare_parameter('output_topic', '/semantic_bev_map')
        
        # Get parameters
        self.z_min = self.get_parameter('z_min').get_parameter_value().double_value
        self.z_max = self.get_parameter('z_max').get_parameter_value().double_value
        self.resolution = self.get_parameter('grid_resolution').get_parameter_value().double_value
        self.size_x = self.get_parameter('grid_size_x').get_parameter_value().double_value
        self.size_y = self.get_parameter('grid_size_y').get_parameter_value().double_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        
        # --- Grid Setup ---
        # Grid dimensions in number of cells
        self.cells_x = int(self.size_x / self.resolution)
        self.cells_y = int(self.size_y / self.resolution)
        
        # Grid origin (bottom-left corner in the BEV frame)
        # We center the grid around the origin (0,0)
        self.grid_origin_x = -self.size_x / 2.0
        self.grid_origin_y = -self.size_y / 2.0
        
        # --- Colormap ---
        # Generate a simple, visually distinct colormap for 256 labels
        self.colormap = self._create_colormap()
        
        
        
        # --- ROS Communications ---
        self.pc_sub = self.create_subscription(
            PointCloud2,
            input_topic,
            self.cloud_callback,
            10  # QoS profile
        )
        self.bev_pub = self.create_publisher(PointCloud2, output_topic, 10)
        
        self.get_logger().info("Semantic BEV Node started.")
        self.get_logger().info(f"  Input Topic: {input_topic}")
        self.get_logger().info(f"  Output Topic: {output_topic}")
        self.get_logger().info(f"  Grid Size: {self.cells_x}x{self.cells_y} cells")
        self.get_logger().info(f"  Resolution: {self.resolution} m/cell")
        self.get_logger().info(f"  Z-Filter Range: [{self.z_min}, {self.z_max}] m")

    def _create_colormap(self):
        """Generates a simple, pseudo-random colormap for 256 labels."""
        cmap = np.zeros((256, 3), dtype=np.uint8)
        for i in range(1, 256): # Skip 0 (background)
            # Use prime multipliers for pseudo-random colors
            cmap[i, 0] = (i * 23) % 255  # Red
            cmap[i, 1] = (i * 51) % 255  # Green
            cmap[i, 2] = (i * 73) % 255  # Blue
        cmap[0] = [0, 0, 0] # Label 0 (often 'unlabeled') is black
        return cmap

    def cloud_callback(self, msg: PointCloud2):
        """Processes the incoming semantic point cloud."""
        t_start = time.perf_counter()
        
        # 1. Initialize BEV grids
        # We use the 'highest point wins' logic
        # -1 = no label, -inf = no height
        bev_labels = np.full((self.cells_y, self.cells_x), -1, dtype=np.int32)
        bev_heights = np.full((self.cells_y, self.cells_x), -np.inf, dtype=np.float32)

        # 2. Iterate through points and fill grid
        # We need x, y, z, and the label field.
        # The 'label' field is UINT32 as defined in the reconstruction node.
        for point in pc2.read_points(msg, field_names=('x', 'y', 'z', 'label'), skip_nans=True):
            x, y, z, label = point[0], point[1], point[2], point[3]
            
            # --- Z-Filter ---
            if not (self.z_min <= z <= self.z_max):
                continue
            
            # --- World to Grid Conversion ---
            # Find the cell index for this point
            grid_c = int((x - self.grid_origin_x) / self.resolution) # Column (x-axis)
            grid_r = int((y - self.grid_origin_y) / self.resolution) # Row (y-axis)
            
            # --- Bounds Check ---
            if not (0 <= grid_c < self.cells_x and 0 <= grid_r < self.cells_y):
                continue
                
            # --- 'Highest Point Wins' Logic ---
            # If this point is higher than the current highest point in this cell
            if z > bev_heights[grid_r, grid_c]:
                bev_heights[grid_r, grid_c] = z
                bev_labels[grid_r, grid_c] = label
        
        # 3. Create BEV PointCloud message from the grid
        bev_points = []
        for r in range(self.cells_y):
            for c in range(self.cells_x):
                label = bev_labels[r, c]
                
                # Skip empty cells
                if label == -1:
                    continue
                
                # Get color from colormap
                color = self.colormap[label % 256] # Modulo for safety
                r_val, g_val, b_val = int(color[0]), int(color[1]), int(color[2])
                
                # Pack RGB into a single float (for PointCloud2 'rgb' field)
                rgb_uint32 = (r_val << 16) | (g_val << 8) | b_val
                rgb_float32 = struct.unpack('f', struct.pack('I', rgb_uint32))[0]
                
                # Get world coordinates (center of the cell)
                x_world = self.grid_origin_x + (c + 0.5) * self.resolution
                y_world = self.grid_origin_y + (r + 0.5) * self.resolution
                z_world = 0.0 # Flatten to z=0 for a pure 2D BEV
                
                bev_points.append([x_world, y_world, z_world, rgb_float32])
        
        # 4. Publish the BEV PointCloud
        if not bev_points:
            # self.get_logger().warn("No points in BEV map.", throttle_duration_sec=5.0)
            return

        # Create header
        # The BEV map is in the same frame as the input cloud
        header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
        
        # Define fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Create and publish the cloud
        bev_cloud_msg = pc2.create_cloud(header, fields, bev_points)
        self.bev_pub.publish(bev_cloud_msg)
        
        t_total = (time.perf_counter() - t_start) * 1000
        # self.get_logger().info(f"Generated BEV map in {t_total:.2f}ms")


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    node = SemanticBEVNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down BEV node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
