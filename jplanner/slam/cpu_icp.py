#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import torch # PyTorch 임포트 (CUDA 사용)
import struct # RGB 패킹을 위해

# --- SLAM을 위한 새로운 임포트 ---
import open3d as o3d
# ---------------------------------

class PointCloudBEVNode(Node):
    """
    Depth 이미지를 3D Point Cloud와 BEV Map으로 변환하고 발행하는 노드.
    [SLAM 버전]
    - Odometry: Open3D ICP를 사용하여 CPU에서 자체 계산 (odom 의존 X)
    - Mapping: "Highest Point Wins"를 사용하여 GPU에서 BEV 맵 누적
    """

    def __init__(self):
        super().__init__('pointcloud_bev_node')

        # --- 1. 기본 모듈 초기화 ---
        self.bridge = CvBridge()
        self.device = torch.device('cuda')
        self.get_logger().info(f'🚀 CUDA GPU 가속 활성화 (PyTorch, {self.device})')
        self.get_logger().info(f'🤖 Open3D ICP Odometry 활성화 (CPU)')

        # --- 2. ROS 파라미터 선언 (PCL + BEV) ---
        # Point Cloud 파라미터
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('pointcloud_topic', '/pointcloud')
        # (SLAM: source_frame은 사용하지만 target_frame은 자체 계산하므로 사용 X)
        self.declare_parameter('source_frame', 'camera_depth_optical_frame')
        self.declare_parameter('global_frame_id', 'map') # 누적된 맵의 고정 프레임 ID

        # 카메라 내부 파라미터
        self.declare_parameter('cam.fx', 431.0625)
        self.declare_parameter('cam.fy', 431.0625)
        self.declare_parameter('cam.cx', 434.492)
        self.declare_parameter('cam.cy', 242.764)
        self.declare_parameter('cam.height', 480)
        self.declare_parameter('cam.width', 848)

        # PCL 다운샘플링 (Y축, X축)
        self.declare_parameter('pcl.downsample_y', 9)
        self.declare_parameter('pcl.downsample_x', 6)

        # BEV 파라미터
        self.declare_parameter('bev_topic', '/bev_map')
        self.declare_parameter('bev.z_min', -0.25)
        self.declare_parameter('bev.z_max', 1.0)
        self.declare_parameter('bev.resolution', 0.05)
        self.declare_parameter('bev.size_x', 30.0)
        self.declare_parameter('bev.size_y', 30.0)

        # ICP (Odometry) 파라미터
        self.declare_parameter('icp.downsample_y', 12) # ICP는 더 거칠게 샘플링 (속도 향상)
        self.declare_parameter('icp.downsample_x', 9)
        self.declare_parameter('icp.threshold', 0.02) # 2cm
        self.declare_parameter('icp.max_iteration', 30)
        self.declare_parameter('icp.min_points', 100) # ICP 수행 최소 포인트 수

        # --- 3. 파라미터 값 할당 ---
        depth_topic = self.get_parameter('depth_topic').value
        pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.global_frame_id = self.get_parameter('global_frame_id').value

        self.fx = self.get_parameter('cam.fx').value
        self.fy = self.get_parameter('cam.fy').value
        self.cx = self.get_parameter('cam.cx').value
        self.cy = self.get_parameter('cam.cy').value
        self.cam_height = self.get_parameter('cam.height').value
        self.cam_width = self.get_parameter('cam.width').value

        self.downsample_y = self.get_parameter('pcl.downsample_y').value
        self.downsample_x = self.get_parameter('pcl.downsample_x').value

        bev_topic = self.get_parameter('bev_topic').value
        self.z_min = self.get_parameter('bev.z_min').value
        self.z_max = self.get_parameter('bev.z_max').value
        self.resolution = self.get_parameter('bev.resolution').value
        self.size_x = self.get_parameter('bev.size_x').value
        self.size_y = self.get_parameter('bev.size_y').value

        self.cells_x = int(self.size_x / self.resolution)
        self.cells_y = int(self.size_y / self.resolution)
        self.grid_origin_x = -self.size_x / 2.0
        self.grid_origin_y = -self.size_y / 2.0

        # --- 4. ROS 통신 설정 ---
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(
            Image, depth_topic, self.depth_callback, qos_profile
        )
        self.pointcloud_pub = self.create_publisher(PointCloud2, pointcloud_topic, qos_profile)
        self.bev_pub = self.create_publisher(PointCloud2, bev_topic, qos_profile)

        # (TF 리스너 삭제)

        # --- 5. Point Cloud 필드 정의 ---
        self.pointcloud_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        self.point_step = 16 # 4 bytes * 4 fields

        # --- 6. GPU 파라미터 초기화 ---
        self._init_gpu_parameters()
        
        # --- 7. SLAM을 위한 상태 변수 초기화 ---
        self._init_slam_parameters()

        self.get_logger().info('✅ PointCloud + BEV SLAM Node initialized')
        self.get_logger().info(f"  PCL Topic (Global): {pointcloud_topic}")
        self.get_logger().info(f"  BEV Topic (Global): {bev_topic}")
        self.get_logger().info(f"  Global Frame ID: {self.global_frame_id}")

    def _init_gpu_parameters(self):
        """GPU에서 사용할 파라미터 미리 생성"""
        # 1. PCL 재구성을 위한 픽셀 그리드
        v, u = torch.meshgrid(
            torch.arange(self.cam_height, device=self.device, dtype=torch.float32),
            torch.arange(self.cam_width, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        self.u_grid = u
        self.v_grid = v
        self.fx_tensor = torch.tensor(self.fx, device=self.device, dtype=torch.float32)
        self.fy_tensor = torch.tensor(self.fy, device=self.device, dtype=torch.float32)
        self.cx_tensor = torch.tensor(self.cx, device=self.device, dtype=torch.float32)
        self.cy_tensor = torch.tensor(self.cy, device=self.device, dtype=torch.float32)

        # 2. BEV 생성을 위한 파라미터 (GPU 텐서)
        self.z_min_t = torch.tensor(self.z_min, device=self.device, dtype=torch.float32)
        self.z_max_t = torch.tensor(self.z_max, device=self.device, dtype=torch.float32)
        self.z_range_t = self.z_max_t - self.z_min_t
        self.resolution_t = torch.tensor(self.resolution, device=self.device, dtype=torch.float32)
        self.grid_origin_x_t = torch.tensor(self.grid_origin_x, device=self.device, dtype=torch.float32)
        self.grid_origin_y_t = torch.tensor(self.grid_origin_y, device=self.device, dtype=torch.float32)

        # 3. BEV 높이 맵 (SLAM: "전역" 맵으로 사용)
        self.global_bev_heights_flat = torch.full(
            (self.cells_y * self.cells_x,),
            -torch.inf,
            device=self.device,
            dtype=torch.float32
        )
        self.get_logger().info(f'GPU 파라미터 초기화 완료 ({self.cam_height}x{self.cam_width})')

    def _init_slam_parameters(self):
        """SLAM(ICP Odometry)을 위한 CPU/Open3D 파라미터 초기화"""
        # 1. 이전 프레임의 PCL (Open3D 형식)
        self.previous_pcl_o3d = None
        
        # 2. 시작 지점 기준 현재 카메라의 누적 변환 행렬 (NumPy)
        self.global_transform_np = np.eye(4, dtype=np.float32)

        # 3. ICP 다운샘플링
        self.icp_downsample_y = self.get_parameter('icp.downsample_y').value
        self.icp_downsample_x = self.get_parameter('icp.downsample_x').value
        self.icp_min_points = self.get_parameter('icp.min_points').value

        # 4. Open3D ICP 파라미터
        self.icp_threshold = self.get_parameter('icp.threshold').value
        self.icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=self.get_parameter('icp.max_iteration').value
        )
        self.get_logger().info('CPU ICP Odometry 파라미터 초기화 완료')

    def depth_callback(self, msg):
        """Depth 이미지를 수신하여 Odometry 계산 및 PCL/BEV 누적"""
        try:
            # --- 1. Depth 이미지 -> NumPy (CPU) ---
            depth_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding=msg.encoding
            ).astype(np.float32) / 1000.0
            
            # --- 2. NumPy -> GPU 텐서 ---
            depth_tensor = torch.from_numpy(depth_image).to(self.device)

            # --- 3. 3D 재구성 (GPU) ---
            # (H, W, 3) 형태의 (카메라 로컬) 포인트 클라우드
            pointcloud_cam = self.depth_to_pointcloud_gpu(depth_tensor)

            # --- 4. ICP를 위한 PCL 준비 (CPU 변환) ---
            # (ICP는 CPU 연산이므로 데이터를 GPU->CPU로 가져와야 함)
            icp_sampled = pointcloud_cam[::self.icp_downsample_y, ::self.icp_downsample_x, :]
            points_icp = icp_sampled.reshape(-1, 3)
            valid_mask = points_icp[:, 2] > 0.1 # Z > 10cm (너무 가까운 노이즈 제거)
            points_icp_np = points_icp[valid_mask].cpu().numpy()

            if points_icp_np.shape[0] < self.icp_min_points:
                self.get_logger().warn('ICP를 위한 포인트 부족, 프레임 건너뜀', throttle_duration_sec=1.0)
                return

            current_pcl_o3d = o3d.geometry.PointCloud()
            current_pcl_o3d.points = o3d.utility.Vector3dVector(points_icp_np)
            # (성능 향상을 위해 Voxel Downsampling 추가 권장)
            current_pcl_o3d = current_pcl_o3d.voxel_down_sample(voxel_size=0.05)


            # --- 5. Odometry 계산 (CPU - ICP) ---
            if self.previous_pcl_o3d is None:
                # 첫 프레임: 현재 PCL을 '이전'으로 저장하고 종료
                self.previous_pcl_o3d = current_pcl_o3d
                self.get_logger().info('SLAM 첫 프레임 초기화 완료.')
                return

            # (CPU에서 ICP 수행 - 이 부분이 병목 구간입니다)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                current_pcl_o3d,        # Source (현재 프레임)
                self.previous_pcl_o3d,  # Target (이전 프레임)
                self.icp_threshold,
                np.eye(4), # Initial guess
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                self.icp_criteria
            )
            
            # ICP 결과(relative_transform)로 전역 변환(global_transform) 갱신
            # reg_p2p.transformation은 "현재 -> 이전" 변환입니다.
            # 우리는 "이전 -> 현재" 변환이 필요하므로 역행렬(inverse)을 취합니다.
            relative_transform_np = np.linalg.inv(reg_p2p.transformation).astype(np.float32)
            
            # (Global = Global_Old * Relative)
            self.global_transform_np = self.global_transform_np @ relative_transform_np

            # 다음 프레임을 위해 현재 PCL을 저장
            self.previous_pcl_o3d = current_pcl_o3d

            # --- 6. 전역 좌표계로 변환 (GPU) ---
            # (계산된 self.global_transform_np 사용)
            transformed_cloud = self.apply_transform_gpu(
                pointcloud_cam, # (주의: ICP용 PCL이 아닌, 원본 전체 PCL 사용)
                self.global_transform_np
            )

            # --- 7. 메시지 발행 (PCL, BEV) ---
            stamp = msg.header.stamp
            
            # Fork 1: 3D 포인트 클라우드 처리 및 발행 (누적된 맵 기준)
            self.process_and_publish_pointcloud(transformed_cloud, stamp)

            # Fork 2: BEV 맵 처리 및 발행 (누적된 맵 기준)
            self.process_and_publish_bev(transformed_cloud, stamp)

        except Exception as e:
            self.get_logger().error(f'Point Cloud/BEV/ICP 처리 오류: {e}')

    def depth_to_pointcloud_gpu(self, depth_tensor):
        """GPU를 이용한 Depth to Point Cloud 변환 (카메라 좌표계)"""
        z = depth_tensor
        x = (self.u_grid - self.cx_tensor) * z / self.fx_tensor
        y = (self.v_grid - self.cy_tensor) * z / self.fy_tensor
        return torch.stack([x, y, z], dim=-1)

    def apply_transform_gpu(self, points, matrix):
        """GPU를 이용한 좌표 변환"""
        original_shape = points.shape
        points_flat = points.reshape(-1, 3)

        matrix_tensor = torch.from_numpy(matrix).to(self.device, dtype=torch.float32)

        ones = torch.ones((points_flat.shape[0], 1), device=self.device, dtype=torch.float32)
        homogeneous = torch.cat([points_flat, ones], dim=1)

        transformed = torch.mm(homogeneous, matrix_tensor.T)
        return transformed[:, :3].reshape(original_shape)

    def process_and_publish_pointcloud(self, transformed_cloud, stamp):
        """3D 포인트 클라우드를 다운샘플링, 색상 적용 후 발행 (전역 맵)"""
        sampled = transformed_cloud[::self.downsample_y, ::self.downsample_x, :]
        points = sampled.reshape(-1, 3)
        
        # (SLAM: Z > 0 필터는 카메라 기준이 아닌 맵 기준이므로 수정/제거 필요할 수 있음)
        valid_mask = points[:, 2] > (self.z_min - 0.1) # BEV z_min보다 약간 아래
        points = points[valid_mask]

        points_np = points.cpu().numpy()
        num_points = points_np.shape[0]
        if num_points == 0:
            return

        colors = np.zeros((num_points, 3), dtype=np.uint8)
        colors[:, 0] = 200
        colors[:, 1] = 100
        colors[:, 2] = 208

        pointcloud_msg = self.create_pointcloud_msg(
            points_np, colors, stamp, self.global_frame_id # <-- 고정된 "map" 프레임
        )
        self.pointcloud_pub.publish(pointcloud_msg)

    def process_and_publish_bev(self, transformed_cloud, stamp):
        """
        'transformed_cloud' (H, W, 3) GPU 텐서를 사용하여
        GPU에서 **누적된 전역** BEV 맵을 생성하고 발행합니다.
        """
        # 1. (H, W, 3) -> (N, 3) -> (x_flat, y_flat, z_flat)
        x_flat = transformed_cloud[..., 0].ravel()
        y_flat = transformed_cloud[..., 1].ravel()
        z_flat = transformed_cloud[..., 2].ravel()

        # 2. Z-필터 마스크 (GPU)
        mask = (z_flat > self.z_min_t) & (z_flat < self.z_max_t)

        # 3. 월드 좌표 -> 그리드 인덱스 변환 (GPU)
        grid_c = ((x_flat - self.grid_origin_x_t) / self.resolution_t).long()
        grid_r = ((y_flat - self.grid_origin_y_t) / self.resolution_t).long()

        # 4. 바운더리 체크 마스크 (GPU)
        mask &= (grid_c >= 0) & (grid_c < self.cells_x) & \
                (grid_r >= 0) & (grid_r < self.cells_y)

        # 5. 유효한 포인트만 필터링 (GPU)
        valid_z = z_flat[mask]
        if valid_z.shape[0] == 0:
            return # 이 프레임에서 BEV에 추가할 포인트 없음

        valid_r = grid_r[mask]
        valid_c = grid_c[mask]

        # 6. 2D 인덱스 -> 1D 선형 인덱스 (GPU)
        linear_indices = valid_r * self.cells_x + valid_c

        # 7. "Highest Point Wins" (GPU Scatter-Max)
        # 7.1. (삭제!!!) 맵을 초기화하지 않습니다.
        # self.global_bev_heights_flat.fill_(-torch.inf) # <-- 누적을 위해 이 줄 삭제!

        # 7.2. 전역 맵(global_bev_heights_flat)에 업데이트
        self.global_bev_heights_flat.index_reduce_(
            dim=0,
            index=linear_indices,
            source=valid_z,
            reduce="amax",
            include_self=True # <-- True로 변경 (기존 맵 값과 새 값 중 최대값 선택)
        )

        # 8. 유효한 셀만 추출 (GPU)
        # (전체 맵에서 유효한 셀을 매번 다시 계산)
        valid_bev_mask = self.global_bev_heights_flat > -torch.inf

        valid_indices_flat = torch.where(valid_bev_mask)[0]
        if valid_indices_flat.shape[0] == 0:
            return

        height_values = self.global_bev_heights_flat[valid_bev_mask]

        # 9. 1D 인덱스 -> 2D 인덱스 (GPU)
        r_idx_bev = torch.div(valid_indices_flat, self.cells_x, rounding_mode='floor')
        c_idx_bev = valid_indices_flat % self.cells_x

        # 10. BEV 포인트의 월드 좌표 계산 (GPU)
        x_world = self.grid_origin_x_t + (c_idx_bev.float() + 0.5) * self.resolution_t
        y_world = self.grid_origin_y_t + (r_idx_bev.float() + 0.5) * self.resolution_t
        z_world = torch.zeros_like(x_world)

        # 11. 높이(Z) 값 -> RGB 색상 변환 (GPU)
        rgb_float32_gpu = self._height_to_color_gpu(height_values)

        # 12. (X, Y, Z, RGB) 데이터 결합 (GPU)
        bev_data_gpu = torch.stack(
            [x_world, y_world, z_world, rgb_float32_gpu],
            dim=-1
        )

        # 13. GPU -> CPU 전송
        bev_data_np = bev_data_gpu.cpu().numpy()

        # 14. PointCloud2 메시지 생성 (CPU)
        bev_msg = self._create_cloud_from_data(
            bev_data_np, stamp, self.global_frame_id # <-- 고정된 "map" 프레임
        )

        # 15. 발행
        self.bev_pub.publish(bev_msg)


    def _height_to_color_gpu(self, z):
        """
        GPU 텐서(z)를 입력받아 'Jet' Colormap RGB 텐서를 반환합니다.
        (z_min ~ z_max 상대 좌표 기준)
        """
        # 정규화 [0, 1] -> [0, 4]
        z_norm = (z - self.z_min_t) / self.z_range_t
        z_norm = torch.clamp(z_norm, 0.0, 1.0) * 4.0

        r = torch.zeros_like(z_norm)
        g = torch.zeros_like(z_norm)
        b = torch.zeros_like(z_norm)

        # (z_norm < 1.0) : Blue -> Cyan
        mask = z_norm < 1.0
        b[mask] = 1.0
        g[mask] = z_norm[mask]

        # (1.0 <= z_norm < 2.0) : Cyan -> Green
        mask = (z_norm >= 1.0) & (z_norm < 2.0)
        g[mask] = 1.0
        b[mask] = 2.0 - z_norm[mask]

        # (2.0 <= z_norm < 3.0) : Green -> Yellow
        mask = (z_norm >= 2.0) & (z_norm < 3.0)
        g[mask] = 1.0
        r[mask] = z_norm[mask] - 2.0

        # (z_norm >= 3.0) : Yellow -> Red
        mask = z_norm >= 3.0
        r[mask] = 1.0
        g[mask] = 4.0 - z_norm[mask]

        r_val = (r * 255).long()
        g_val = (g * 255).long()
        b_val = (b * 255).long()

        # Bitwise shift (<<) 대신 곱셈 사용
        rgb_packed_gpu = (r_val * 65536) + (g_val * 256) + b_val

        rgb_uint32_gpu = rgb_packed_gpu.to(torch.uint32)
        return rgb_uint32_gpu.view(torch.float32)

    # (transform_to_matrix 함수는 삭제)

    def create_pointcloud_msg(self, points_np, colors_np, stamp, frame_id):
        """
        (N, 3) points와 (N, 3) uint8 colors NumPy 배열로
        PointCloud2 메시지를 생성합니다. (PCL용)
        """
        header = Header(stamp=stamp, frame_id=frame_id)

        rgb_uint32 = (
            (colors_np[:, 0].astype(np.uint32) << 16) |
            (colors_np[:, 1].astype(np.uint32) << 8) |
            (colors_np[:, 2].astype(np.uint32))
        )
        rgb_float32 = rgb_uint32.view(np.float32)

        pointcloud_data = np.hstack([
            points_np.astype(np.float32),
            rgb_float32.reshape(-1, 1)
        ])

        num_points = pointcloud_data.shape[0]
        return PointCloud2(
            header=header,
            height=1,
            width=num_points,
            fields=self.pointcloud_fields,
            is_bigendian=False,
            point_step=self.point_step,
            row_step=self.point_step * num_points,
            data=pointcloud_data.tobytes(),
            is_dense=True,
        )

    def _create_cloud_from_data(self, point_data_np, stamp, frame_id):
        """
        (N, 4) [x, y, z, rgb_float32] NumPy 배열로
        PointCloud2 메시지를 생성합니다. (BEV용)
        """
        header = Header(stamp=stamp, frame_id=frame_id)
        num_points = point_data_np.shape[0]

        return PointCloud2(
            header=header,
            height=1,
            width=num_points,
            fields=self.pointcloud_fields,
            is_bigendian=False,
            point_step=self.point_step,
            row_step=self.point_step * num_points,
            data=point_data_np.astype(np.float32).tobytes(),
            is_dense=True,
        )


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    node = PointCloudBEVNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
