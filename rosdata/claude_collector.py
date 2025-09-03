import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix, JointState, BatteryState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

import cv2
from cv_bridge import CvBridge
import numpy as np
import json
import os
import time
import argparse
import csv
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
import psutil
import gc

# ------------------------------------------------
# 1) Configuration and Optimization Settings
# ------------------------------------------------

SESSION_TIMESTAMP = time.strftime("%Y%m%d_%H%M", time.localtime())
BASE_DIR = f"../data/{SESSION_TIMESTAMP}"
os.makedirs(BASE_DIR, exist_ok=True)

# Performance optimization settings
MAX_WORKERS = min(4, psutil.cpu_count())  # Limit thread pool size
BUFFER_SIZE = 100  # Queue buffer size
IMAGE_COMPRESSION_QUALITY = 85  # JPEG compression quality (1-100)
DEPTH_COMPRESSION = True  # Use compressed numpy save format

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataPoint:
    """Unified data structure for all sensor data"""
    timestamp: str
    data_type: str
    data: Any
    priority: int = 1  # 1=high (sensor data), 2=medium (images), 3=low (depth)

# Folder structure
folders = [
    "images/front",
    "depth/rs2",
    "images/realsense_color",
    "depth/realsense_depth",
]
for folder in folders:
    os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

# ------------------------------------------------
# 2) Optimized QoS Settings
# ------------------------------------------------

# Different QoS profiles for different data types
qos_sensor_data = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5  # Reduced depth for sensor data
)

qos_image_data = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,  # Less strict for images
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=2  # Small buffer for images
)

qos_high_freq = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1  # Minimal buffer for high-frequency data
)


class OptimizedROS2DataCollector(Node):
    def __init__(self, enabled_topics):
        super().__init__('optimized_ros2_data_collector')
        
        # Use callback groups for parallel processing
        self.sensor_callback_group = ReentrantCallbackGroup()
        self.image_callback_group = ReentrantCallbackGroup()
        
        self.bridge = CvBridge()
        
        # Thread-safe data storage with timestamps
        self.data_lock = threading.RLock()
        self.latest_data = {}
        self.data_timestamps = {}  # Track when each data type was last updated
        
        # Asynchronous file writing queues
        self.csv_queue = Queue(maxsize=BUFFER_SIZE)
        self.image_queue = Queue(maxsize=BUFFER_SIZE)
        self.depth_queue = Queue(maxsize=BUFFER_SIZE)
        
        # Thread pool for file I/O
        self.io_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        
        # Initialize CSV
        self.csv_path = os.path.join(BASE_DIR, "data.csv")
        self.csv_header = [
            'timestamp', 'data_age_ms',  # Added data freshness indicator
            'odom_pos_x', 'odom_pos_y', 'odom_pos_z',
            'odom_orient_x', 'odom_orient_y', 'odom_orient_z', 'odom_orient_w',
            'gnss1_lat', 'gnss1_lon', 'gnss1_alt',
            'gnss2_lat', 'gnss2_lon', 'gnss2_alt',
            'manual_linear_x', 'manual_linear_y', 'manual_angular_z',
            'gx5_orient_x', 'gx5_orient_y', 'gx5_orient_z', 'gx5_orient_w',
            'gx5_angvel_x', 'gx5_angvel_y', 'gx5_angvel_z',
            'gx5_linacc_x', 'gx5_linacc_y', 'gx5_linacc_z',
            'mcu_orient_x', 'mcu_orient_y', 'mcu_orient_z', 'mcu_orient_w',
            'mcu_angvel_x', 'mcu_angvel_y', 'mcu_angvel_z',
            'mcu_linacc_x', 'mcu_linacc_y', 'mcu_linacc_z',
            'joint_names', 'joint_positions', 'joint_velocities', 'joint_efforts',
            'battery_voltage', 'battery_percentage', 'battery_current',
            'pose_x', 'pose_y', 'pose_z',
            'pose_orient_x', 'pose_orient_y', 'pose_orient_z', 'pose_orient_w',
            # Data quality indicators
            'missing_data_count', 'image_saved', 'depth_saved'
        ]
        
        self._init_csv()
        
        # Image compression settings
        self.jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, IMAGE_COMPRESSION_QUALITY]
        
        # Statistics tracking
        self.stats = {
            'total_cycles': 0,
            'missing_data_cycles': 0,
            'images_saved': 0,
            'depths_saved': 0,
            'queue_overflows': 0
        }
        
        # Start worker threads
        self._start_worker_threads()
        
        # Create subscriptions with optimized QoS
        self._create_subscriptions(enabled_topics)
        
        # Timer with adaptive frequency
        self.target_frequency = 10.0  # Hz
        self.actual_frequency = 10.0
        self.timer_ = self.create_timer(1.0/self.target_frequency, self.timer_callback)
        
        logger.info(f"Optimized data collector initialized with {MAX_WORKERS} workers")

    def _init_csv(self):
        """Initialize CSV file with proper headers"""
        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_header)

    def _start_worker_threads(self):
        """Start background worker threads for file I/O"""
        self.running = True
        
        # CSV writer thread
        self.csv_thread = threading.Thread(target=self._csv_worker, daemon=True)
        self.csv_thread.start()
        
        # Image writer thread
        self.image_thread = threading.Thread(target=self._image_worker, daemon=True)
        self.image_thread.start()
        
        # Depth writer thread  
        self.depth_thread = threading.Thread(target=self._depth_worker, daemon=True)
        self.depth_thread.start()

    def _csv_worker(self):
        """Background thread for CSV writing"""
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            while self.running:
                try:
                    row = self.csv_queue.get(timeout=1.0)
                    if row is None:  # Shutdown signal
                        break
                    writer.writerow(row)
                    f.flush()
                    self.csv_queue.task_done()
                except:
                    continue

    def _image_worker(self):
        """Background thread for image saving with compression"""
        while self.running:
            try:
                data_point = self.image_queue.get(timeout=1.0)
                if data_point is None:  # Shutdown signal
                    break
                
                filename, image = data_point
                # Use optimized JPEG compression
                success = cv2.imwrite(filename, image, self.jpeg_params)
                if success:
                    self.stats['images_saved'] += 1
                
                self.image_queue.task_done()
                
                # Periodic garbage collection
                if self.stats['images_saved'] % 100 == 0:
                    gc.collect()
                    
            except:
                continue

    def _depth_worker(self):
        """Background thread for depth data saving"""
        while self.running:
            try:
                data_point = self.depth_queue.get(timeout=1.0)
                if data_point is None:  # Shutdown signal
                    break
                
                filename, depth_array = data_point
                # Use compressed numpy format
                if DEPTH_COMPRESSION:
                    np.savez_compressed(filename.replace('.npy', '.npz'), depth=depth_array)
                else:
                    np.save(filename, depth_array)
                
                self.stats['depths_saved'] += 1
                self.depth_queue.task_done()
                
            except:
                continue

    def _create_subscriptions(self, enabled_topics):
        """Create ROS2 subscriptions with optimized QoS"""
        
        # High-priority sensor data
        self.create_subscription(
            Odometry, "/gx5/nav/odom", self.odom_callback,
            qos_sensor_data, callback_group=self.sensor_callback_group)
        
        self.create_subscription(
            NavSatFix, "/gx5/gnss1/fix", self.gps_fix1_callback,
            qos_sensor_data, callback_group=self.sensor_callback_group)
        
        self.create_subscription(
            NavSatFix, "/gx5/gnss2/fix", self.gps_fix2_callback,
            qos_sensor_data, callback_group=self.sensor_callback_group)
        
        self.create_subscription(
            Imu, "/gx5/imu/data", self.imu_gx5_callback,
            qos_high_freq, callback_group=self.sensor_callback_group)
        
        self.create_subscription(
            Imu, "/mcu/state/imu", self.imu_mcu_callback,
            qos_high_freq, callback_group=self.sensor_callback_group)
        
        self.create_subscription(
            BatteryState, "/mcu/state/battery", self.battery_callback,
            qos_sensor_data, callback_group=self.sensor_callback_group)
        
        self.create_subscription(
            JointState, "/mcu/state/jointURDF", self.joint_urdf_callback,
            qos_sensor_data, callback_group=self.sensor_callback_group)
        
        self.create_subscription(
            Twist, "/mcu/command/manual_twist", self.manual_twist_callback,
            qos_sensor_data, callback_group=self.sensor_callback_group)
        
        # Image data with separate callback group
        self.create_subscription(
            Image, "/argus/ar0234_front_left/image_raw", self.front_image_callback,
            qos_image_data, callback_group=self.image_callback_group)
        
        self.create_subscription(
            Image, "/camera/camera/color/image_raw", self.realsense_color_callback,
            qos_image_data, callback_group=self.image_callback_group)
        
        # Depth data
        self.create_subscription(
            Image, "/mcu/state/rs2/depth", self.depth_rs2_callback,
            qos_image_data, callback_group=self.image_callback_group)
        
        self.create_subscription(
            Image, "/camera/camera/depth/image_rect_raw", self.realsense_depth_callback,
            qos_image_data, callback_group=self.image_callback_group)

    def get_timestamp(self):
        """Get current timestamp in milliseconds"""
        return int(time.time() * 1000)

    def timer_callback(self):
        """Optimized 10Hz timer callback"""
        current_time = self.get_timestamp()
        
        with self.data_lock:
            # Calculate data freshness
            max_age_ms = 0
            missing_count = 0
            
            for data_type, timestamp in self.data_timestamps.items():
                age_ms = current_time - timestamp
                max_age_ms = max(max_age_ms, age_ms)
                if age_ms > 500:  # Data older than 500ms considered stale
                    missing_count += 1
            
            # Build CSV row efficiently
            row_data = self._build_csv_row(current_time, max_age_ms, missing_count)
            
            # Queue CSV writing (non-blocking)
            try:
                self.csv_queue.put_nowait(row_data)
            except:
                self.stats['queue_overflows'] += 1
                logger.warning("CSV queue overflow")
            
            # Handle image/depth saving
            images_saved, depths_saved = self._queue_media_saves(current_time)
            
            # Update statistics
            self.stats['total_cycles'] += 1
            if missing_count > 0:
                self.stats['missing_data_cycles'] += 1
            
            # Periodic status report
            if self.stats['total_cycles'] % 100 == 0:
                self._log_statistics()

    def _build_csv_row(self, timestamp, max_age_ms, missing_count):
        """Build CSV row data efficiently"""
        row = [str(timestamp), max_age_ms]
        
        # Odometry
        odom = self.latest_data.get('odom', {})
        row.extend([
            odom.get('pos_x', 'nan'), odom.get('pos_y', 'nan'), odom.get('pos_z', 'nan'),
            odom.get('orient_x', 'nan'), odom.get('orient_y', 'nan'), 
            odom.get('orient_z', 'nan'), odom.get('orient_w', 'nan')
        ])
        
        # GNSS data
        gnss1 = self.latest_data.get('gnss1', {})
        row.extend([gnss1.get('latitude', 'nan'), gnss1.get('longitude', 'nan'), gnss1.get('altitude', 'nan')])
        
        gnss2 = self.latest_data.get('gnss2', {})
        row.extend([gnss2.get('latitude', 'nan'), gnss2.get('longitude', 'nan'), gnss2.get('altitude', 'nan')])
        
        # Manual twist
        twist = self.latest_data.get('manual_twist', {})
        row.extend([twist.get('linear_x', 'nan'), twist.get('linear_y', 'nan'), twist.get('angular_z', 'nan')])
        
        # IMU data (GX5)
        gx5 = self.latest_data.get('imu_gx5', {})
        row.extend([
            gx5.get('orient_x', 'nan'), gx5.get('orient_y', 'nan'), gx5.get('orient_z', 'nan'), gx5.get('orient_w', 'nan'),
            gx5.get('angvel_x', 'nan'), gx5.get('angvel_y', 'nan'), gx5.get('angvel_z', 'nan'),
            gx5.get('linacc_x', 'nan'), gx5.get('linacc_y', 'nan'), gx5.get('linacc_z', 'nan')
        ])
        
        # IMU data (MCU)
        mcu = self.latest_data.get('imu_mcu', {})
        row.extend([
            mcu.get('orient_x', 'nan'), mcu.get('orient_y', 'nan'), mcu.get('orient_z', 'nan'), mcu.get('orient_w', 'nan'),
            mcu.get('angvel_x', 'nan'), mcu.get('angvel_y', 'nan'), mcu.get('angvel_z', 'nan'),
            mcu.get('linacc_x', 'nan'), mcu.get('linacc_y', 'nan'), mcu.get('linacc_z', 'nan')
        ])
        
        # Joint states
        joint_urdf = self.latest_data.get('joint_urdf', {})
        row.extend([
            str(joint_urdf.get('names', '[]')), str(joint_urdf.get('positions', '[]')),
            str(joint_urdf.get('velocities', '[]')), str(joint_urdf.get('efforts', '[]'))
        ])
        
        # Battery
        battery = self.latest_data.get('battery', {})
        row.extend([battery.get('voltage', 'nan'), battery.get('percentage', 'nan'), battery.get('current', 'nan')])
        
        # Pose (if available)
        pose = self.latest_data.get('pose', {})
        row.extend([
            pose.get('x', 'nan'), pose.get('y', 'nan'), pose.get('z', 'nan'),
            pose.get('orient_x', 'nan'), pose.get('orient_y', 'nan'), 
            pose.get('orient_z', 'nan'), pose.get('orient_w', 'nan')
        ])
        
        # Quality indicators
        row.extend([missing_count, 0, 0])  # Will be updated with actual save counts
        
        return row

    def _queue_media_saves(self, timestamp):
        """Queue image and depth data for background saving"""
        images_saved = 0
        depths_saved = 0
        
        # Queue front camera image
        if hasattr(self, 'latest_front_image') and self.latest_front_image is not None:
            filename = os.path.join(BASE_DIR, "images", "front", f"{timestamp}.jpg")
            try:
                self.image_queue.put_nowait((filename, self.latest_front_image.copy()))
                self.latest_front_image = None
                images_saved += 1
            except:
                pass
        
        # Queue RealSense color image
        if hasattr(self, 'latest_realsense_color') and self.latest_realsense_color is not None:
            filename = os.path.join(BASE_DIR, "images", "realsense_color", f"{timestamp}.jpg")
            try:
                self.image_queue.put_nowait((filename, self.latest_realsense_color.copy()))
                self.latest_realsense_color = None
                images_saved += 1
            except:
                pass
        
        # Queue depth data
        if hasattr(self, 'latest_rs2_depth') and self.latest_rs2_depth is not None:
            filename = os.path.join(BASE_DIR, "depth", "rs2", f"{timestamp}.npy")
            try:
                self.depth_queue.put_nowait((filename, self.latest_rs2_depth.copy()))
                self.latest_rs2_depth = None
                depths_saved += 1
            except:
                pass
        
        if hasattr(self, 'latest_realsense_depth') and self.latest_realsense_depth is not None:
            filename = os.path.join(BASE_DIR, "depth", "realsense_depth", f"{timestamp}.npy")
            try:
                self.depth_queue.put_nowait((filename, self.latest_realsense_depth.copy()))
                self.latest_realsense_depth = None
                depths_saved += 1
            except:
                pass
        
        return images_saved, depths_saved

    def _log_statistics(self):
        """Log performance statistics"""
        total = self.stats['total_cycles']
        missing_rate = (self.stats['missing_data_cycles'] / total) * 100 if total > 0 else 0
        
        logger.info(f"Stats: {total} cycles, {missing_rate:.1f}% missing data, "
                   f"{self.stats['images_saved']} images, {self.stats['depths_saved']} depths, "
                   f"{self.stats['queue_overflows']} overflows")

    # ------------------------------------------------
    # Optimized Callbacks
    # ------------------------------------------------
    
    def _update_data(self, data_type: str, data: dict):
        """Thread-safe data update with timestamp tracking"""
        current_time = self.get_timestamp()
        with self.data_lock:
            self.latest_data[data_type] = data
            self.data_timestamps[data_type] = current_time

    def front_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_front_image = cv_image
        except Exception as e:
            logger.error(f"Front image conversion error: {e}")

    def realsense_color_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_realsense_color = cv_image
        except Exception as e:
            logger.error(f"RealSense color conversion error: {e}")

    def depth_rs2_callback(self, msg):
        try:
            depth_array = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_rs2_depth = depth_array
        except Exception as e:
            logger.error(f"RS2 depth conversion error: {e}")

    def realsense_depth_callback(self, msg):
        try:
            depth_array = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_realsense_depth = depth_array
        except Exception as e:
            logger.error(f"RealSense depth conversion error: {e}")

    def odom_callback(self, msg):
        data = {
            'pos_x': msg.pose.pose.position.x,
            'pos_y': msg.pose.pose.position.y,
            'pos_z': msg.pose.pose.position.z,
            'orient_x': msg.pose.pose.orientation.x,
            'orient_y': msg.pose.pose.orientation.y,
            'orient_z': msg.pose.pose.orientation.z,
            'orient_w': msg.pose.pose.orientation.w,
        }
        self._update_data('odom', data)

    def gps_fix1_callback(self, msg):
        data = {
            'latitude': msg.latitude,
            'longitude': msg.longitude,
            'altitude': msg.altitude
        }
        self._update_data('gnss1', data)

    def gps_fix2_callback(self, msg):
        data = {
            'latitude': msg.latitude,
            'longitude': msg.longitude,
            'altitude': msg.altitude
        }
        self._update_data('gnss2', data)

    def manual_twist_callback(self, msg):
        data = {
            'linear_x': msg.linear.x,
            'linear_y': msg.linear.y,
            'angular_z': msg.angular.z
        }
        self._update_data('manual_twist', data)

    def imu_gx5_callback(self, msg):
        data = {
            'orient_x': msg.orientation.x,
            'orient_y': msg.orientation.y,
            'orient_z': msg.orientation.z,
            'orient_w': msg.orientation.w,
            'angvel_x': msg.angular_velocity.x,
            'angvel_y': msg.angular_velocity.y,
            'angvel_z': msg.angular_velocity.z,
            'linacc_x': msg.linear_acceleration.x,
            'linacc_y': msg.linear_acceleration.y,
            'linacc_z': msg.linear_acceleration.z
        }
        self._update_data('imu_gx5', data)

    def imu_mcu_callback(self, msg):
        data = {
            'orient_x': msg.orientation.x,
            'orient_y': msg.orientation.y,
            'orient_z': msg.orientation.z,
            'orient_w': msg.orientation.w,
            'angvel_x': msg.angular_velocity.x,
            'angvel_y': msg.angular_velocity.y,
            'angvel_z': msg.angular_velocity.z,
            'linacc_x': msg.linear_acceleration.x,
            'linacc_y': msg.linear_acceleration.y,
            'linacc_z': msg.linear_acceleration.z
        }
        self._update_data('imu_mcu', data)

    def joint_urdf_callback(self, msg):
        data = {
            'names': list(msg.name),
            'positions': list(msg.position),
            'velocities': list(msg.velocity),
            'efforts': list(msg.effort)
        }
        self._update_data('joint_urdf', data)

    def battery_callback(self, msg):
        data = {
            'voltage': msg.voltage,
            'percentage': msg.percentage,
            'current': msg.current,
        }
        self._update_data('battery', data)

    def destroy_node(self):
        """Cleanup resources properly"""
        logger.info("Shutting down data collector...")
        self.running = False
        
        # Signal worker threads to stop
        self.csv_queue.put(None)
        self.image_queue.put(None)
        self.depth_queue.put(None)
        
        # Wait for threads to finish
        self.csv_thread.join(timeout=2)
        self.image_thread.join(timeout=2)
        self.depth_thread.join(timeout=2)
        
        # Shutdown executor
        self.io_executor.shutdown(wait=True)
        
        self._log_statistics()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description="Optimized ROS2 Data Collector")
    parser.add_argument('--enable', nargs='+', default=[], 
                       help="Enable additional topics")
    parser.add_argument('--compression', type=int, default=85,
                       help="JPEG compression quality (1-100)")
    parser.add_argument('--workers', type=int, default=MAX_WORKERS,
                       help="Number of worker threads")

    args = parser.parse_args()
    
    # Update global settings
    global IMAGE_COMPRESSION_QUALITY, MAX_WORKERS
    IMAGE_COMPRESSION_QUALITY = args.compression
    MAX_WORKERS = args.workers

    rclpy.init()
    
    # Use MultiThreadedExecutor for better performance
    executor = MultiThreadedExecutor(num_threads=MAX_WORKERS)
    
    node = OptimizedROS2DataCollector(set(args.enable))
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
