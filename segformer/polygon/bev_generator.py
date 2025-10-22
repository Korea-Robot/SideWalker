import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from transforms3d.quaternions import quat2mat
import math

# ==============================================================================
# --- ⚙️ 카메라 및 로봇 설정 ---
# ==============================================================================

def intrinsics_from_fov(width: int,
                        height: int,
                        fov_h_deg: float,
                        fov_v_deg: float,
                        cx: float | None = None,
                        cy: float | None = None) -> np.ndarray:
    """
    해상도(W,H)와 수평/수직 FOV(도)로 카메라 내참행렬 K를 계산.
    """
    fov_h = math.radians(fov_h_deg)
    fov_v = math.radians(fov_v_deg)

    fx = width  / (2.0 * math.tan(fov_h / 2.0))
    fy = height / (2.0 * math.tan(fov_v / 2.0))

    if cx is None:
        cx = width  / 2.0
    if cy is None:
        cy = height / 2.0

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=float)
    return K


# RealSense D435 기본 설정 (실제 카메라 파라미터로 수정 필요)
DEPTH_INTRINSICS = np.array([
    [336.1,   0.0, 320.0],
    [  0.0, 433.1, 240.0],
    [  0.0,   0.0,   1.0]
])

# 카메라 외부 파라미터 (로봇 베이스 좌표계 대비 카메라 위치)
TRANS = [-0.015, 0.22, 0.05]
QUAT = [0.49, -0.51, 0.5, -0.5]  # [x, y, z, w]

def create_hmt(translation, quaternion):
    """Translation과 Quaternion으로 4x4 동차 변환 행렬(HMT)을 생성"""
    rot_matrix = quat2mat([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    hmt = np.eye(4)
    hmt[:3, :3] = rot_matrix
    hmt[:3, 3] = translation
    return hmt

EXTRINSIC_HMT = create_hmt(TRANS, QUAT)

# ==============================================================================
# --- 🚀 BEV 생성 함수 ---
# ==============================================================================

def unproject_depth_to_pointcloud(depth_map, camera_k):
    """깊이 맵을 3D 포인트 클라우드로 변환"""
    fx, fy = camera_k[0, 0], camera_k[1, 1]
    cx, cy = camera_k[0, 2], camera_k[1, 2]
    height, width = depth_map.shape

    u, v = np.meshgrid(np.arange(width), np.arange(height))

    valid_mask = (depth_map > 0) & np.isfinite(depth_map)
    z = np.where(valid_mask, depth_map, 0)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.stack((x, y, z), axis=-1).reshape(-1, 3)

def apply_transform_to_pointcloud(points, hmt):
    """4x4 동차 변환 행렬을 포인트 클라우드에 적용"""
    ones = np.ones((points.shape[0], 1))
    homo_points = np.hstack((points, ones))
    transformed_points = homo_points @ hmt.T
    return transformed_points[:, :3]

def create_bev_from_pointcloud(depth_image,
                               intrinsics,
                               extrinsics_hmt,
                               bev_resolution=0.05,
                               bev_size_m=10.0,
                               z_min=0.1,
                               z_max=1.0):
    """깊이 이미지로 BEV 이미지 생성"""
    # 1. 카메라 좌표계의 3D 포인트 클라우드로 변환
    points_camera_frame = unproject_depth_to_pointcloud(depth_image, intrinsics)

    # 2. 로봇 베이스 좌표계로 변환
    points_robot_frame = apply_transform_to_pointcloud(points_camera_frame, extrinsics_hmt)

    # 3. 높이(z) 필터링
    height_filter = (points_robot_frame[:, 2] > z_min) & (points_robot_frame[:, 2] < z_max)
    points_filtered = points_robot_frame[height_filter]

    # 4. 2D BEV 그리드에 투영
    bev_pixel_size = int(bev_size_m / bev_resolution)
    bev_image = np.zeros((bev_pixel_size, bev_pixel_size), dtype=np.uint8)

    x_robot = points_filtered[:, 0]
    y_robot = points_filtered[:, 1]

    u_bev = (bev_pixel_size // 2 - y_robot / bev_resolution).astype(int)
    v_bev = (bev_pixel_size - 1 - x_robot / bev_resolution).astype(int)

    valid_bev_indices = (u_bev >= 0) & (u_bev < bev_pixel_size) & \
                        (v_bev >= 0) & (v_bev < bev_pixel_size)

    u_bev_valid = u_bev[valid_bev_indices]
    v_bev_valid = v_bev[valid_bev_indices]

    bev_image[v_bev_valid, u_bev_valid] = 255

    return bev_image

# ==============================================================================
# --- 🤖 ROS2 Node ---
# ==============================================================================

class RealSenseBEVNode(Node):
    def __init__(self):
        super().__init__('realsense_bev_node')
        
        # ROS2 Setup
        self.bridge = CvBridge()
        self.depth_sub = self.create_subscription(
            Image, 
            '/camera/camera/depth/image_rect_raw', 
            self.depth_callback, 
            10
        )
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)
        self.odom_sub = self.create_subscription(
            Odometry, 
            '/rko_lio/odometry', 
            self.odom_callback, 
            10
        )
        
        # BEV 파라미터
        self.bev_resolution = 0.05
        self.bev_size_m = 10.0
        self.z_min = 0.2
        self.z_max = 1.0
        
        # 데이터 저장
        self.latest_bev = None
        self.latest_depth = None
        self.depth_scale = 0.001  # RealSense depth scale (mm to m)
        
        # Matplotlib 설정
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 7))
        self.fig.suptitle('RealSense Depth & BEV Visualization', fontsize=16)
        
        # Depth 이미지 subplot
        self.ax_depth = self.axes[0]
        self.ax_depth.set_title('Depth Image')
        self.im_depth = self.ax_depth.imshow(np.zeros((480, 640)), cmap='jet', vmin=0, vmax=5)
        self.ax_depth.axis('off')
        plt.colorbar(self.im_depth, ax=self.ax_depth, label='Depth (m)')
        
        # BEV subplot
        self.ax_bev = self.axes[1]
        self.ax_bev.set_title('Bird\'s Eye View (BEV)')
        bev_pixel_size = int(self.bev_size_m / self.bev_resolution)
        self.im_bev = self.ax_bev.imshow(np.zeros((bev_pixel_size, bev_pixel_size)), 
                                         cmap='gray', vmin=0, vmax=255)
        
        # 로봇 위치 표시
        robot_pos_x = bev_pixel_size // 2
        robot_pos_y = bev_pixel_size - 1
        self.robot_marker, = self.ax_bev.plot(robot_pos_x, robot_pos_y, 
                                               'rv', markersize=12, label='Robot')
        self.ax_bev.set_xlabel('Left/Right')
        self.ax_bev.set_ylabel('Forward')
        self.ax_bev.legend()
        
        plt.tight_layout()
        
        # 업데이트 타이머
        self.timer = self.create_timer(0.1, self.update_plot)
        
        self.get_logger().info('RealSense BEV Node initialized')
    
    def depth_callback(self, msg):
        """Depth 이미지 콜백"""
        try:
            # ROS Image를 numpy array로 변환
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # depth 스케일 적용 (mm -> m)
            depth_image_m = depth_image.astype(np.float32) * self.depth_scale
            
            # NaN과 inf 제거
            depth_image_m = np.nan_to_num(depth_image_m, nan=0.0, posinf=0.0, neginf=0.0)
            
            self.latest_depth = depth_image_m
            
            # BEV 생성
            self.latest_bev = create_bev_from_pointcloud(
                depth_image=depth_image_m,
                intrinsics=DEPTH_INTRINSICS,
                extrinsics_hmt=EXTRINSIC_HMT,
                bev_resolution=self.bev_resolution,
                bev_size_m=self.bev_size_m,
                z_min=self.z_min,
                z_max=self.z_max
            )
            
        except Exception as e:
            self.get_logger().error(f'Error in depth callback: {e}')
    
    def odom_callback(self, msg):
        """Odometry 콜백"""
        # 필요한 경우 오도메트리 정보 활용
        pass
    
    def update_plot(self):
        """Matplotlib 플롯 업데이트"""
        if self.latest_depth is not None and self.latest_bev is not None:
            # Depth 이미지 업데이트
            self.im_depth.set_data(self.latest_depth)
            
            # BEV 이미지 업데이트
            self.im_bev.set_data(self.latest_bev)
            
            # 화면 갱신
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

# ==============================================================================
# --- 🎯 Main ---
# ==============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = RealSenseBEVNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close('all')

if __name__ == '__main__':
    main()
