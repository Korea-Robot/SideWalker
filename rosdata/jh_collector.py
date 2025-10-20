import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix, JointState, BatteryState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import cv2
from cv_bridge import CvBridge
import numpy as np
import json
import os
import time
import argparse
import csv

# ------------------------------------------------
# 1) Global Path/Time Settings
# ✅ Create default save path based on execution time (Year/Month/Day/Hour/Minute)
# ------------------------------------------------

SESSION_TIMESTAMP = time.strftime("%Y%m%d_%H%M", time.localtime())
BASE_DIR = f"../data/{SESSION_TIMESTAMP}"
os.makedirs(BASE_DIR, exist_ok=True)

# Folder structure
folders = [
    "images/front",         # Front camera
    "depth/rs2",            # Depth (RS2)
    # ----------------- ADDED -----------------
    "images/realsense_color", # RealSense color camera
    "depth/realsense_depth",  # RealSense depth camera
    # -----------------------------------------
    # Optional folders
    # "images/side_left",
    # "images/side_right",
]
for folder in folders:
    os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

# Optional folders (created conditionally) are made at initialization
# e.g., "images/side_left", "depth/rs3", "pointcloud", "pose" etc.


# ------------------------------------------------
# 2) QoS Settings
# ✅ QoS settings (RELIABLE & VOLATILE)
# ------------------------------------------------
# Quality of service

qos_profile = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10
)


class ROS2DataCollector(Node):
    def __init__(self, enabled_topics):
        super().__init__('ros2_data_collector')
        self.bridge = CvBridge()

        ####################################################################
        # ✅ The basic method is to create one CSV and continuously overwrite it.
        # Open CSV file (write mode, write header)
        self.csv_path = os.path.join(BASE_DIR, f"data.csv")
        self.csv_file = open(self.csv_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # Define data columns to be collected
        # CSV Header (example) - modify/add as needed
        self.csv_header = [
            'timestamp',           # Timestamp from 10Hz timer (in ms)
            # Odometry
            'odom_pos_x', 'odom_pos_y', 'odom_pos_z',
            'odom_orient_x', 'odom_orient_y', 'odom_orient_z', 'odom_orient_w',
            # GNSS1
            'gnss1_lat', 'gnss1_lon', 'gnss1_alt',
            # GNSS2
            'gnss2_lat', 'gnss2_lon', 'gnss2_alt',
            # Manual twist
            'manual_linear_x', 'manual_linear_y', 'manual_angular_z',
            # IMU gx5
            'gx5_orient_x', 'gx5_orient_y', 'gx5_orient_z', 'gx5_orient_w',
            'gx5_angvel_x', 'gx5_angvel_y', 'gx5_angvel_z',
            'gx5_linacc_x', 'gx5_linacc_y', 'gx5_linacc_z',
            # IMU mcu
            'mcu_orient_x', 'mcu_orient_y', 'mcu_orient_z', 'mcu_orient_w',
            'mcu_angvel_x', 'mcu_angvel_y', 'mcu_angvel_z',
            'mcu_linacc_x', 'mcu_linacc_y', 'mcu_linacc_z',
            # Joint URDF
            'joint_names', 'joint_positions', 'joint_velocities', 'joint_efforts',
            # ----------------- ADDED -----------------
            # Battery State
            'battery_voltage', 'battery_percentage', 'battery_current',
            # -----------------------------------------
            ## Not currently in use.
            # Pose (optional) == pose position info in stand mode
            'pose_x', 'pose_y', 'pose_z',
            'pose_orient_x', 'pose_orient_y', 'pose_orient_z', 'pose_orient_w',
        ]

        self.csv_writer.writerow(self.csv_header)
        self.csv_file.flush() # Immediately write the header
        # Up to here, the basic data format is defined.

        #  ✅ Data to be updated at a 10Hz cycle
        # ------------------------------------------------
        # 2-1) Dictionary to store the most recently received messages
        # ------------------------------------------------
        self.latest_data = {
            # e.g., "odom": {"pos_x": ..., "pos_y": ...},
            #      "gnss1": {...}, etc.
        }

        # Variables to hold raw data like images/depth
        self.latest_front_image = None
        self.latest_rs2_depth = None
        # ----------------- ADDED -----------------
        self.latest_realsense_color_image = None
        self.latest_realsense_depth_image = None
        # -----------------------------------------
        # Optional
        self.latest_side_left_image = None
        self.latest_side_right_image = None
        self.latest_rear_image = None
        self.latest_rs3_depth = None
        self.latest_pointcloud_data = None


        # ------------------------------------------------
        # 3)  ✅ Register ROS2 Subscriptions
        # ------------------------------------------------

        # (1) Front Camera
        # self.create_subscription(Image, "/argus/ar0234_front_left/image_raw",
        #                          self.front_image_callback, qos_profile)

        # # (2) Depth (RS2)
        # self.create_subscription(Image, "/mcu/state/rs2/depth",
        #                          self.depth_rs2_callback, qos_profile)

        # # (3) Odometry
        # self.create_subscription(Odometry, "/gx5/nav/odom",
        #                          self.odom_callback, qos_profile)

        # (4) GNSS1
        self.create_subscription(NavSatFix, "/gx5/gnss1/fix",
                                 self.gps_fix1_callback, qos_profile)

        # (5) GNSS2
        self.create_subscription(NavSatFix, "/gx5/gnss2/fix",
                                 self.gps_fix2_callback, qos_profile)

        # (6) Manual Control Command
        self.create_subscription(Twist, "/mcu/command/manual_twist",
                                 self.manual_twist_callback, qos_profile)

        # (7) GX5 IMU
        self.create_subscription(Imu, "/gx5/imu/data",
                                 self.imu_gx5_callback, qos_profile)

        # (8) MCU IMU
        self.create_subscription(Imu, "/mcu/state/imu",
                                 self.imu_mcu_callback, qos_profile)

        # (9) JointState
        self.create_subscription(JointState, "/mcu/state/jointURDF",
                                 self.joint_urdf_callback, qos_profile)

        # ----------------- ADDED -----------------
        # (10) RealSense Color Camera
        self.create_subscription(Image, "/camera/camera/color/image_raw",
                                 self.realsense_color_callback, qos_profile)

        # (11) RealSense Depth Camera
        self.create_subscription(Image, "/camera/camera/depth/image_rect_raw",
                                 self.realsense_depth_callback, qos_profile)

        # (12) Battery State
        self.create_subscription(BatteryState, "/mcu/state/battery",
                                 self.battery_callback, qos_profile)
        # -----------------------------------------

        """
        # ---------------------------------------------
        # (Option 1) RS3 Depth
        # ---------------------------------------------
        if "rs3" in enabled_topics:
            self.create_subscription(Image, "/mcu/state/rs3/depth",
                                     self.depth_rs3_callback, qos_profile)
            os.makedirs(os.path.join(BASE_DIR, "depth/rs3"), exist_ok=True)

        # ---------------------------------------------
        # (Option 2) Point Cloud
        # ---------------------------------------------
        if "pointcloud" in enabled_topics:
            self.create_subscription(PointCloud2, "/mcu/state/pointcloud",
                                     self.pointcloud_callback, qos_profile)
            os.makedirs(os.path.join(BASE_DIR, "pointcloud"), exist_ok=True)

        # ---------------------------------------------
        # (Option 3) Pose
        # ---------------------------------------------
        if "pose" in enabled_topics:
            os.makedirs(os.path.join(BASE_DIR, "pose"), exist_ok=True)
            self.create_subscription(Pose, "/mcu/command/pose",
                                     self.pose_callback, qos_profile)

        # ---------------------------------------------
        # (Option 4) Side/Rear Cameras
        # ---------------------------------------------
        if "image_side_left" in enabled_topics:
            side_left_dir = os.path.join(BASE_DIR, "images", "side_left")
            os.makedirs(side_left_dir, exist_ok=True)
            self.create_subscription(Image, "/argus/ar0234_side_left/image_raw",
                                     self.side_left_callback, qos_profile)

        if "image_side_right" in enabled_topics:
            side_right_dir = os.path.join(BASE_DIR, "images", "side_right")
            os.makedirs(side_right_dir, exist_ok=True)
            self.create_subscription(Image, "/argus/ar0234_side_right/image_raw",
                                     self.side_right_callback, qos_profile)

        if "image_rear" in enabled_topics:
            rear_dir = os.path.join(BASE_DIR, "images", "rear")
            os.makedirs(rear_dir, exist_ok=True)
            self.create_subscription(Image, "/argus/ar0234_rear/image_raw",
                                     self.rear_image_callback, qos_profile)
        """

        # ------------------------------------------------
        # 4) 10Hz Timer Setup
        # ------------------------------------------------
        self.timer_ = self.create_timer(0.1, self.timer_callback) # 10Hz


    #  ✅ Node close & CSV save
    def destroy_node(self):
        super().destroy_node()
        if self.csv_file:
            self.csv_file.close()



    # ------------------------------------------------
    #  ✅ Utility: Timestamp (in milliseconds) - based on the timer
    # ------------------------------------------------
    def get_timestamp(self):
        return str(int(time.time() * 1000))


    #  ✅ This is the most important piece of code that ensures the same timestamp.
    # ------------------------------------------------
    # 10Hz Timer Callback (CSV logging and Image/Depth saving)
    # ------------------------------------------------
    def timer_callback(self):
        # 1) Generate a common timestamp (at 10Hz)
        timestamp = self.get_timestamp()

        # 2) Log the latest received data as a single row in the CSV
        #    (fill with None or 'nan' if data is missing)
        row_dict = {}

        # (1) odometry
        odom = self.latest_data.get('odom', {})
        row_dict['odom_pos_x'] = odom.get('pos_x', 'nan')
        row_dict['odom_pos_y'] = odom.get('pos_y', 'nan')
        row_dict['odom_pos_z'] = odom.get('pos_z', 'nan')
        row_dict['odom_orient_x'] = odom.get('orient_x', 'nan')
        row_dict['odom_orient_y'] = odom.get('orient_y', 'nan')
        row_dict['odom_orient_z'] = odom.get('orient_z', 'nan')
        row_dict['odom_orient_w'] = odom.get('orient_w', 'nan')

        # (2) gnss1
        gnss1 = self.latest_data.get('gnss1', {})
        row_dict['gnss1_lat'] = gnss1.get('latitude', 'nan')
        row_dict['gnss1_lon'] = gnss1.get('longitude', 'nan')
        row_dict['gnss1_alt'] = gnss1.get('altitude', 'nan')

        # (3) gnss2
        gnss2 = self.latest_data.get('gnss2', {})
        row_dict['gnss2_lat'] = gnss2.get('latitude', 'nan')
        row_dict['gnss2_lon'] = gnss2.get('longitude', 'nan')
        row_dict['gnss2_alt'] = gnss2.get('altitude', 'nan')

        # (4) manual twist
        twist = self.latest_data.get('manual_twist', {})
        row_dict['manual_linear_x'] = twist.get('linear_x', 'nan')
        row_dict['manual_linear_y'] = twist.get('linear_y', 'nan')
        row_dict['manual_angular_z'] = twist.get('angular_z', 'nan')

        # (5) gx5 IMU
        gx5 = self.latest_data.get('imu_gx5', {})
        row_dict['gx5_orient_x'] = gx5.get('orient_x', 'nan')
        row_dict['gx5_orient_y'] = gx5.get('orient_y', 'nan')
        row_dict['gx5_orient_z'] = gx5.get('orient_z', 'nan')
        row_dict['gx5_orient_w'] = gx5.get('orient_w', 'nan')
        row_dict['gx5_angvel_x'] = gx5.get('angvel_x', 'nan')
        row_dict['gx5_angvel_y'] = gx5.get('angvel_y', 'nan')
        row_dict['gx5_angvel_z'] = gx5.get('angvel_z', 'nan')
        row_dict['gx5_linacc_x'] = gx5.get('linacc_x', 'nan')
        row_dict['gx5_linacc_y'] = gx5.get('linacc_y', 'nan')
        row_dict['gx5_linacc_z'] = gx5.get('linacc_z', 'nan')

        # (6) mcu IMU
        mcu = self.latest_data.get('imu_mcu', {})
        row_dict['mcu_orient_x'] = mcu.get('orient_x', 'nan')
        row_dict['mcu_orient_y'] = mcu.get('orient_y', 'nan')
        row_dict['mcu_orient_z'] = mcu.get('orient_z', 'nan')
        row_dict['mcu_orient_w'] = mcu.get('orient_w', 'nan')
        row_dict['mcu_angvel_x'] = mcu.get('angvel_x', 'nan')
        row_dict['mcu_angvel_y'] = mcu.get('angvel_y', 'nan')
        row_dict['mcu_angvel_z'] = mcu.get('angvel_z', 'nan')
        row_dict['mcu_linacc_x'] = mcu.get('linacc_x', 'nan')
        row_dict['mcu_linacc_y'] = mcu.get('linacc_y', 'nan')
        row_dict['mcu_linacc_z'] = mcu.get('linacc_z', 'nan')

        # (7) joint_urdf
        joint_urdf = self.latest_data.get('joint_urdf', {})
        row_dict['joint_names'] = str(joint_urdf.get('names', '[]'))
        row_dict['joint_positions'] = str(joint_urdf.get('positions', '[]'))
        row_dict['joint_velocities'] = str(joint_urdf.get('velocities', '[]'))
        row_dict['joint_efforts'] = str(joint_urdf.get('efforts', '[]'))

        # ----------------- ADDED -----------------
        # (8) battery
        battery = self.latest_data.get('battery', {})
        row_dict['battery_voltage'] = battery.get('voltage', 'nan')
        row_dict['battery_percentage'] = battery.get('percentage', 'nan')
        row_dict['battery_current'] = battery.get('current', 'nan')
        # -----------------------------------------

        # 3) Convert to a list in the correct order for the actual CSV writing
        row = [timestamp]
        for col in self.csv_header[1:]: # Skip the first column (=timestamp) as it's already added
            row.append(row_dict.get(col, 'nan'))

        self.csv_writer.writerow(row)
        self.csv_file.flush()

        # 4) Save files like images / depth / point clouds
        #    - Save only at the 10Hz timing (instead of in the callback)
        #    - Here, the "most recent" message is saved and then the variable is reset to None
        if self.latest_front_image is not None:
            front_img = self.latest_front_image
            front_filename = os.path.join(BASE_DIR, "images", "front", f"{timestamp}.jpg")
            cv2.imwrite(front_filename, front_img)
            self.latest_front_image = None

        if self.latest_rs2_depth is not None:
            depth_array = self.latest_rs2_depth
            depth_filename = os.path.join(BASE_DIR, "depth", "rs2", f"{timestamp}.npy")
            np.save(depth_filename, depth_array)
            self.latest_rs2_depth = None

        # ----------------- ADDED -----------------
        if self.latest_realsense_color_image is not None:
            realsense_img = self.latest_realsense_color_image
            realsense_filename = os.path.join(BASE_DIR, "images", "realsense_color", f"{timestamp}.jpg")
            cv2.imwrite(realsense_filename, realsense_img)
            self.latest_realsense_color_image = None

        if self.latest_realsense_depth_image is not None:
            depth_array = self.latest_realsense_depth_image
            depth_filename = os.path.join(BASE_DIR, "depth", "realsense_depth", f"{timestamp}.npy")
            np.save(depth_filename, depth_array)
            self.latest_realsense_depth_image = None
        # -----------------------------------------


    # ------------------------------------------------
    #  ✅ Callbacks: Upon message reception, only update "latest_data" or "latest_XXX"
    # ------------------------------------------------

    # (1) Front Camera
    def front_image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_front_image = cv_image

    # (2) Depth (RS2)
    def depth_rs2_callback(self, msg):
        depth_array = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.latest_rs2_depth = depth_array

    # ----------------- ADDED -----------------
    # (10) RealSense Color Camera
    def realsense_color_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_realsense_color_image = cv_image

    # (11) RealSense Depth Camera
    def realsense_depth_callback(self, msg):
        depth_array = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.latest_realsense_depth_image = depth_array
    
    # (12) Battery State
    def battery_callback(self, msg: BatteryState):
        self.latest_data['battery'] = {
            'voltage': msg.voltage,
            'percentage': msg.percentage,
            'current': msg.current,
        }
    # -----------------------------------------

    # (Option 4) Side / Rear Cameras
    def side_left_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_side_left_image = cv_image

    def side_right_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_side_right_image = cv_image

    def rear_image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_rear_image = cv_image

    # (Option 1) Depth (RS3)
    def depth_rs3_callback(self, msg):
        depth_array = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.latest_rs3_depth = depth_array

    # (Option 2) PointCloud
    def pointcloud_callback(self, msg):
        self.latest_pointcloud_data = msg.data # As raw binary

    # (3) Odometry
    def odom_callback(self, msg: Odometry):
        self.latest_data['odom'] = {
            'pos_x': msg.pose.pose.position.x,
            'pos_y': msg.pose.pose.position.y,
            'pos_z': msg.pose.pose.position.z,
            'orient_x': msg.pose.pose.orientation.x,
            'orient_y': msg.pose.pose.orientation.y,
            'orient_z': msg.pose.pose.orientation.z,
            'orient_w': msg.pose.pose.orientation.w,
        }

    # (4) GPS (gnss1)
    def gps_fix1_callback(self, msg: NavSatFix):
        self.latest_data['gnss1'] = {
            'latitude': msg.latitude,
            'longitude': msg.longitude,
            'altitude': msg.altitude
        }

    # (5) GPS (gnss2)
    def gps_fix2_callback(self, msg: NavSatFix):
        self.latest_data['gnss2'] = {
            'latitude': msg.latitude,
            'longitude': msg.longitude,
            'altitude': msg.altitude
        }

    # (6) Manual Control
    def manual_twist_callback(self, msg: Twist):
        self.latest_data['manual_twist'] = {
            'linear_x': msg.linear.x,
            'linear_y': msg.linear.y,
            'angular_z': msg.angular.z
        }

    # (7) GX5 IMU
    def imu_gx5_callback(self, msg: Imu):
        self.latest_data['imu_gx5'] = {
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

    # (8) MCU IMU
    def imu_mcu_callback(self, msg: Imu):
        self.latest_data['imu_mcu'] = {
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

    # (9) JointState
    def joint_urdf_callback(self, msg: JointState):
        self.latest_data['joint_urdf'] = {
            'names': list(msg.name),
            'positions': list(msg.position),
            'velocities': list(msg.velocity),
            'efforts': list(msg.effort)
        }

    # (Option 3) Pose
    def pose_callback(self, msg: Pose):
        self.latest_data['pose'] = {
            'x': msg.position.x,
            'y': msg.position.y,
            'z': msg.position.z,
            'orient_x': msg.orientation.x,
            'orient_y': msg.orientation.y,
            'orient_z': msg.orientation.z,
            'orient_w': msg.orientation.w,
        }


def main():
    parser = argparse.ArgumentParser(description="ROS2 Data Collector with optional topics (10Hz sync).")
    parser.add_argument('--enable', nargs='+',
                        help="Enable additional topics: rs3, pointcloud, pose, image_side_left, image_side_right, image_rear",
                        default=[])

    args = parser.parse_args()

    rclpy.init()

    node = ROS2DataCollector(set(args.enable))
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
