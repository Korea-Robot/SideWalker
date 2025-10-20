import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import JointState
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
# 1) 전역 경로/시간 관련 설정
# ✅ 실행 시간 기준으로 기본 저장 경로 생성 (연도/월/날짜/시간/분)
# ------------------------------------------------

SESSION_TIMESTAMP = time.strftime("%Y%m%d_%H%M", time.localtime())
BASE_DIR = f"../data/{SESSION_TIMESTAMP}"
os.makedirs(BASE_DIR, exist_ok=True)

# 폴더 구조
folders = [
    "images/front",   # 전방 카메라
    # "images/side_left",   # 전방 카메라
    # "images/side_right",   # 전방 카메라
    "depth/rs2",      # 뎁스(RS2)
    # 옵션들
]
for folder in folders:
    os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

# 옵션별 폴더(조건부 생성)는 생성 시점에 만듦
# ex) "images/side_left", "depth/rs3", "pointcloud", "pose" 등


# ------------------------------------------------
# 2) QoS 설정
# ✅ QoS 설정 (RELIABLE & VOLATILE 설정)
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
        # ✅ 기본적으로 csv를 하나 만들고 거기게 계속 덮어 씌우는 방식
        # CSV 파일 열기 (쓰기 모드, 헤더 작성)
        self.csv_path = os.path.join(BASE_DIR, f"data.csv")
        self.csv_file = open(self.csv_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # 취득할 데이터 칼럼들 설정
        # CSV 헤더(예시) - 필요에 맞게 수정/추가
        self.csv_header = [
            'timestamp',          # 10Hz 타이머 기준 시간 (ms 단위)
            # 오도메트리
            'odom_pos_x', 'odom_pos_y', 'odom_pos_z',
            'odom_orient_x', 'odom_orient_y', 'odom_orient_z', 'odom_orient_w',
            # GNSS1
            'gnss1_lat', 'gnss1_lon', 'gnss1_alt',
            # GNSS2
            'gnss2_lat', 'gnss2_lon', 'gnss2_alt',
            # manual twist
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

            ## 현재는 사용하지 않음.
            # Pose (옵션) == stand mode에서의 포즈 위치 정보
            'pose_x', 'pose_y', 'pose_z',
            'pose_orient_x', 'pose_orient_y', 'pose_orient_z', 'pose_orient_w',


            # pointcloud는 일단 아예 제거.
            # PointCloud (옵션) - CSV에 경로만 기록할지 여부는 필요에 따라 결정
            # 'pointcloud_file',    # 예시로 파일 경로를 적을 수 있음
            # 기타...
        ]

        self.csv_writer.writerow(self.csv_header)
        self.csv_file.flush()  # 헤더 즉시 기록
        # 여기 까지는 기본적인 데이터 형식을 정했음.

        #  ✅ 10헤르츠 주기로 업데이트 할 데이터
        # ------------------------------------------------
        # 2-1) 최근 수신 메시지를 저장할 딕셔너리
        # ------------------------------------------------
        self.latest_data = {
            # 예: "odom": {"pos_x": ..., "pos_y": ...},
            #     "gnss1": {...}, 등등
        }

        # 이미지/뎁스 등의 Raw 데이터를 보관할 변수
        self.latest_front_image = None
        self.latest_rs2_depth = None
        # 옵션
        self.latest_side_left_image = None
        self.latest_side_right_image = None
        self.latest_rear_image = None
        self.latest_rs3_depth = None
        self.latest_pointcloud_data = None

        #
        # ------------------------------------------------
        # 3)  ✅ ROS2 Subscription 등록
        # ------------------------------------------------

        # (1) 전방 카메라
        self.create_subscription(Image, "/argus/ar0234_front_left/image_raw",
                                 self.front_image_callback, qos_profile)

        # (2) 뎁스 (RS2)
        self.create_subscription(Image, "/mcu/state/rs2/depth",
                                 self.depth_rs2_callback, qos_profile)

        # (3) 오도메트리
        self.create_subscription(Odometry, "/gx5/nav/odom",
                                 self.odom_callback, qos_profile)

        # (4) GNSS1
        self.create_subscription(NavSatFix, "/gx5/gnss1/fix",
                                 self.gps_fix1_callback, qos_profile)

        # (5) GNSS2
        self.create_subscription(NavSatFix, "/gx5/gnss2/fix",
                                 self.gps_fix2_callback, qos_profile)

        # (6) 수동 제어 명령
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

        # ---------------------------------------------
        # (옵션1) RS3 뎁스
        # ---------------------------------------------
        if "rs3" in enabled_topics:
            self.create_subscription(Image, "/mcu/state/rs3/depth",
                                     self.depth_rs3_callback, qos_profile)
            os.makedirs(os.path.join(BASE_DIR, "depth/rs3"), exist_ok=True)

        # ---------------------------------------------
        # (옵션2) 포인트 클라우드
        # ---------------------------------------------
        if "pointcloud" in enabled_topics:
            self.create_subscription(PointCloud2, "/mcu/state/pointcloud",
                                     self.pointcloud_callback, qos_profile)
            os.makedirs(os.path.join(BASE_DIR, "pointcloud"), exist_ok=True)

        # ---------------------------------------------
        # (옵션3) Pose
        # ---------------------------------------------
        if "pose" in enabled_topics:
            os.makedirs(os.path.join(BASE_DIR, "pose"), exist_ok=True)
            self.create_subscription(Pose, "/mcu/command/pose",
                                     self.pose_callback, qos_profile)

        # ---------------------------------------------
        # (옵션4) 사이드/리어 카메라
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

        # ------------------------------------------------
        # 4) 10Hz 타이머 설정
        # ------------------------------------------------
        self.timer_ = self.create_timer(0.1, self.timer_callback)  # 10Hz


    #  ✅ Node close & CSV save
    def destroy_node(self):
        super().destroy_node()
        if self.csv_file:
            self.csv_file.close()



    # ------------------------------------------------
    #  ✅ 유틸: 타임스탬프 (밀리초 단위) - 타이머 기준
    # ------------------------------------------------
    def get_timestamp(self):
        return str(int(time.time() * 1000))


    #  ✅ 동일한 timestamp를 갖도록 해주는 코드라인. 제일중요함.
    # ------------------------------------------------
    # 10Hz 타이머 콜백 (CSV 기록 및 이미지/뎁스 저장)
    # ------------------------------------------------
    def timer_callback(self):
        # 1) 공통 timestamp 생성 (10Hz 기준)
        timestamp = self.get_timestamp()

        # 2) 현재까지 수신된 최신 데이터를 CSV 한 행으로 기록
        #    (없으면 None 또는 'nan'으로 채움)
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
        # 리스트를 str로 변환해서 저장(예: CSV 컬럼에)
        row_dict['joint_names'] = str(joint_urdf.get('names', '[]'))
        row_dict['joint_positions'] = str(joint_urdf.get('positions', '[]'))
        row_dict['joint_velocities'] = str(joint_urdf.get('velocities', '[]'))
        row_dict['joint_efforts'] = str(joint_urdf.get('efforts', '[]'))

        # # (8) pose (옵션)
        # pose = self.latest_data.get('pose', {})
        # row_dict['pose_x'] = pose.get('x', 'nan')
        # row_dict['pose_y'] = pose.get('y', 'nan')
        # row_dict['pose_z'] = pose.get('z', 'nan')
        # row_dict['pose_orient_x'] = pose.get('orient_x', 'nan')
        # row_dict['pose_orient_y'] = pose.get('orient_y', 'nan')
        # row_dict['pose_orient_z'] = pose.get('orient_z', 'nan')
        # row_dict['pose_orient_w'] = pose.get('orient_w', 'nan')

        # # (9) pointcloud (옵션) - 파일 경로만 기록 예시
        # pcd_info = self.latest_data.get('pointcloud', {})
        # row_dict['pointcloud_file'] = pcd_info.get('filename', 'nan')

        # 3) 실제 CSV에 기록할 순서를 맞춰 리스트로 변환
        row = [timestamp]
        for col in self.csv_header[1:]:  # 첫 컬럼(=timestamp)은 이미 넣었으므로
            row.append(row_dict.get(col, 'nan'))

        self.csv_writer.writerow(row)
        self.csv_file.flush()

        # 4) 이미지 / 뎁스 / 포인트클라우드 등 파일 저장
        #    - 10Hz 타이밍에 맞춰서만 저장 (콜백에서 저장 대신)
        #    - 여기서는 "가장 최신" 메시지를 저장 후 None 처리
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

        #  ✅ 옵션들 side,rear cameras, rear depth
        # if self.latest_side_left_image is not None:
        #     side_left_filename = os.path.join(BASE_DIR, "images", "side_left", f"{timestamp}.jpg")
        #     cv2.imwrite(side_left_filename, self.latest_side_left_image)
        #     self.latest_side_left_image = None

        # if self.latest_side_right_image is not None:
        #     side_right_filename = os.path.join(BASE_DIR, "images", "side_right", f"{timestamp}.jpg")
        #     cv2.imwrite(side_right_filename, self.latest_side_right_image)
        #     self.latest_side_right_image = None

        # if self.latest_rear_image is not None:
        #     rear_filename = os.path.join(BASE_DIR, "images", "rear", f"{timestamp}.jpg")
        #     cv2.imwrite(rear_filename, self.latest_rear_image)
        #     self.latest_rear_image = None

        # if self.latest_rs3_depth is not None:
        #     rs3_filename = os.path.join(BASE_DIR, "depth", "rs3", f"{timestamp}.npy")
        #     np.save(rs3_filename, self.latest_rs3_depth)
        #     self.latest_rs3_depth = None

        #####  point 클라우드 정보는 사용안함 너무큼..
        # if self.latest_pointcloud_data is not None:
        #     pc_filename = os.path.join(BASE_DIR, "pointcloud", f"{timestamp}.pcd")
        #     with open(pc_filename, "wb") as f:
        #         f.write(self.latest_pointcloud_data)
        #     # CSV에 기록할 filename을 업데이트 (다음 루프에 반영)
        #     self.latest_data['pointcloud'] = {"filename": pc_filename}
        #     self.latest_pointcloud_data = None









    # ------------------------------------------------
    #  ✅ callbacks : 메시지 수신 시 "latest_data" 또는 "latest_XXX" 만 업데이트
    # ------------------------------------------------

    # (1) 전방 카메라
    def front_image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_front_image = cv_image

    # (옵션4) 사이드 / 리어 카메라
    def side_left_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_side_left_image = cv_image

    def side_right_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_side_right_image = cv_image

    def rear_image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_rear_image = cv_image

    # (2) 뎁스 (RS2)
    def depth_rs2_callback(self, msg):
        depth_array = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.latest_rs2_depth = depth_array

    # (옵션1) 뎁스 (RS3)
    def depth_rs3_callback(self, msg):
        depth_array = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.latest_rs3_depth = depth_array

    # (옵션2) 포인트클라우드
    def pointcloud_callback(self, msg):
        self.latest_pointcloud_data = msg.data  # 바이너리 그대로

    # (3) 오도메트리
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

    # (6) 수동 제어
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

    # (옵션3) Pose
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

"""
작동 원리 요약
콜백 함수: 토픽 수신 시, 해당 데이터를 self.latest_data[...] 또는 self.latest_XXX(이미지) 등 변수에 저장만 합니다.

10Hz 타이머(self.timer_)에서:

공통 timestamp = get_timestamp()(ms 단위) 생성
latest_data에서 필요한 정보들을 꺼내 CSV 한 줄을 작성해 파일에 Append
latest_front_image 등 이미지/뎁스 데이터가 있으면 이 시점에 파일로 기록 후 변수 None으로 초기화
타임 싱크:

CSV와 이미지/뎁스 파일 모두 **동일한 timestamp**로 파일명/행에 기록되므로,
후처리 시 파일명으로 매칭하여 센서 데이터를 손쉽게 동기화할 수 있습니다.
이렇게 하면 각 센서 토픽의 실제 수신 주기와 무관하게, “10Hz 간격”으로 데이터가 정렬된 CSV + 이미지/뎁스 파일을 얻게 됩니다.

"""
