#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, TransformException
from transforms3d.quaternions import quat2mat
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import torch # PyTorch 임포트 (CUDA 사용)
import struct # RGB 패킹을 위해

class PointCloudBEVNode(Node):
    """
    Depth 이미지를 3D Point Cloud와 BEV Map으로 변환하고 발행하는 노드.
    모든 주요 연산은 PyTorch CUDA GPU 가속을 사용합니다.
    """

    def __init__(self):
        super().__init__('pointcloud_bev_node')

        # --- 1. 기본 모듈 초기화 ---
        self.bridge = CvBridge()
        self.device = torch.device('cuda')
        self.get_logger().info(f'🚀 CUDA GPU 가속 활성화 (PyTorch, {self.device})')

        # --- 2. ROS 파라미터 선언 (PCL + BEV) ---
        # Point Cloud 파라미터
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('pointcloud_topic', '/pointcloud')
        self.declare_parameter('source_frame', 'camera_depth_optical_frame')
        self.declare_parameter('target_frame', 'camera_link') # TF 변환 최종 좌표계

        # 카메라 내부 파라미터 (Intel RealSense D435 848x480 기준)
        # Intrinsic 
        self.declare_parameter('cam.fx', 431.0625)
        self.declare_parameter('cam.fy', 431.0625)
        self.declare_parameter('cam.cx', 434.492)
        self.declare_parameter('cam.cy', 242.764)
        # Resolution
        self.declare_parameter('cam.height', 480)
        self.declare_parameter('cam.width', 848)

        # PCL 다운샘플링 (Y축, X축)
        self.declare_parameter('pcl.downsample_y', 9)
        self.declare_parameter('pcl.downsample_x', 6)

        # BEV 파라미터
        self.declare_parameter('bev_topic', '/bev_map')
        self.declare_parameter('bev.z_min', -0.15)       # BEV 높이 필터 최소값
        self.declare_parameter('bev.z_max', 1.0)        # BEV 높이 필터 최대값
        self.declare_parameter('bev.resolution', 0.1)   # BEV 그리드 해상도 (m/cell)
        self.declare_parameter('bev.size_x', 30.0)      # BEV 맵 전체 X 크기 (m)
        self.declare_parameter('bev.size_y', 30.0)      # BEV 맵 전체 Y 크기 (m)

        # --- 3. 파라미터 값 할당 ---
        # PCL 파라미터
        depth_topic = self.get_parameter('depth_topic').value
        pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.source_frame = self.get_parameter('source_frame').value
        self.target_frame = self.get_parameter('target_frame').value

        self.fx = self.get_parameter('cam.fx').value
        self.fy = self.get_parameter('cam.fy').value
        self.cx = self.get_parameter('cam.cx').value
        self.cy = self.get_parameter('cam.cy').value
        self.cam_height = self.get_parameter('cam.height').value
        self.cam_width = self.get_parameter('cam.width').value

        self.downsample_y = self.get_parameter('pcl.downsample_y').value
        self.downsample_x = self.get_parameter('pcl.downsample_x').value

        # BEV 파라미터
        bev_topic = self.get_parameter('bev_topic').value
        self.z_min = self.get_parameter('bev.z_min').value
        self.z_max = self.get_parameter('bev.z_max').value
        self.resolution = self.get_parameter('bev.resolution').value
        self.size_x = self.get_parameter('bev.size_x').value
        self.size_y = self.get_parameter('bev.size_y').value

        # BEV 그리드 설정
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

        # 구독자 (Depth Image)
        self.create_subscription(
            Image, depth_topic, self.depth_callback, qos_profile
        )

        # 발행자 (Point Cloud & BEV Map)
        self.pointcloud_pub = self.create_publisher(PointCloud2, pointcloud_topic, qos_profile)
        self.bev_pub = self.create_publisher(PointCloud2, bev_topic, qos_profile)

        # TF 리스너
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- 5. Point Cloud 필드 정의 (PCL과 BEV 공통) ---
        self.pointcloud_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        self.point_step = 16 # 4 bytes * 4 fields

        # --- 6. GPU 파라미터 초기화 ---
        self._init_gpu_parameters()

        self.get_logger().info('✅ PointCloud + BEV Node initialized (GPU Only)')
        self.get_logger().info(f"  PCL Topic: {pointcloud_topic}")
        self.get_logger().info(f"  BEV Topic: {bev_topic}")
        self.get_logger().info(f"  BEV Grid: {self.cells_x}x{self.cells_y} cells @ {self.resolution} m")

    def _init_gpu_parameters(self):
        """GPU에서 사용할 파라미터 미리 생성 (콜백 함수 내 부하 감소)"""

        # 1. PCL 재구성을 위한 픽셀 그리드 (카메라 좌표계)
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

        # 3. BEV 높이 맵 (재사용을 위해 클래스 변수로 선언)
        # -inf로 채워진 1D 텐서 (scatter 연산을 위해)
        self.bev_heights_flat = torch.full(
            (self.cells_y * self.cells_x,),
            -torch.inf,
            device=self.device,
            dtype=torch.float32
        )

        self.get_logger().info(f'GPU 파라미터 초기화 완료 ({self.cam_height}x{self.cam_width})')

    def depth_callback(self, msg):
        """Depth 이미지를 수신하여 PCL과 BEV 동시 처리"""
        try:
            # --- 1. Depth 이미지 -> NumPy (CPU) ---
            depth_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding=msg.encoding
            ).astype(np.float32) / 1000.0

            # --- 2. NumPy -> GPU 텐서 ---
            depth_tensor = torch.from_numpy(depth_image).to(self.device)

            # --- 3. 3D 재구성 (GPU) ---
            # (H, W, 3) 형태의 카메라 좌표계 포인트 클라우드
            pointcloud_cam = self.depth_to_pointcloud_gpu(depth_tensor)

            # --- 4. TF 조회 (CPU) ---
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, self.source_frame, rclpy.time.Time()
            )
            transform_matrix = self.transform_to_matrix(transform)

            # --- 5. 좌표 변환 (GPU) ---
            # (H, W, 3) 형태의 로봇('target_frame') 좌표계 포인트 클라우드
            transformed_cloud = self.apply_transform_gpu(pointcloud_cam, transform_matrix)

            # --- 6. 메시지 발행 (PCL, BEV) ---
            stamp = msg.header.stamp # self.get_clock().now().to_msg()

            # Fork 1: 3D 포인트 클라우드 처리 및 발행
            self.process_and_publish_pointcloud(transformed_cloud, stamp)

            # Fork 2: BEV 맵 처리 및 발행
            self.process_and_publish_bev(transformed_cloud, stamp)

        except TransformException as e:
            self.get_logger().warn(f'TF 변환 실패: {e}', throttle_duration_sec=1.0)
        except Exception as e:
            self.get_logger().error(f'Point Cloud/BEV 처리 오류: {e}')

    def depth_to_pointcloud_gpu(self, depth_tensor):
        """GPU를 이용한 Depth to Point Cloud 변환 (카메라 좌표계)"""
        z = depth_tensor
        x = (self.u_grid - self.cx_tensor) * z / self.fx_tensor
        y = (self.v_grid - self.cy_tensor) * z / self.fy_tensor

        # (H, W, 3) 형태로 스택
        return torch.stack([x, y, z], dim=-1)

    def apply_transform_gpu(self, points, matrix):
        """GPU를 이용한 좌표 변환"""
        original_shape = points.shape
        points_flat = points.reshape(-1, 3)

        matrix_tensor = torch.from_numpy(matrix).to(self.device, dtype=torch.float32)

        # 동차 좌표 (N, 4)
        ones = torch.ones((points_flat.shape[0], 1), device=self.device, dtype=torch.float32)
        homogeneous = torch.cat([points_flat, ones], dim=1)

        # 변환 (N, 4) @ (4, 4)^T = (N, 4)
        transformed = torch.mm(homogeneous, matrix_tensor.T)

        # (N, 3) -> (H, W, 3)
        return transformed[:, :3].reshape(original_shape)

    def process_and_publish_pointcloud(self, transformed_cloud, stamp):
        """3D 포인트 클라우드를 다운샘플링, 색상 적용 후 발행"""

        # 1. 다운샘플링 (GPU)
        sampled = transformed_cloud[::self.downsample_y, ::self.downsample_x, :]

        # 2. Flatten (GPU) -> (N_sampled, 3)
        points = sampled.reshape(-1, 3)

        # 3. 유효한 포인트 필터링 (Z > 0)
        # 변환 후 z=0 (혹은 음수)가 된 포인트 제거
        valid_mask = points[:, 2] > 0.01 # Z > 1cm
        points = points[valid_mask]

        # 4. GPU -> CPU 이동
        points_np = points.cpu().numpy()

        # 5. 색상 생성 (CPU)
        num_points = points_np.shape[0]
        if num_points == 0:
            return # 발행할 포인트 없음

        colors = np.zeros((num_points, 3), dtype=np.uint8)
        colors[:, 0] = 200 # R (핑크/보라)
        colors[:, 1] = 100 # G
        colors[:, 2] = 208 # B

        # 6. PointCloud2 메시지 생성 (CPU)
        pointcloud_msg = self.create_pointcloud_msg(
            points_np, colors, stamp, self.target_frame
        )

        # 7. 발행
        self.pointcloud_pub.publish(pointcloud_msg)

    def process_and_publish_bev(self, transformed_cloud, stamp):
        """
        'transformed_cloud' (H, W, 3) GPU 텐서를 사용하여
        GPU에서 BEV 맵을 생성하고 발행합니다.
        """

        # 1. (H, W, 3) -> (N, 3) -> (x_flat, y_flat, z_flat)
        # .ravel()은 1D 뷰를 생성 (복사 없음)
        x_flat = transformed_cloud[..., 0].ravel()
        y_flat = transformed_cloud[..., 1].ravel()
        z_flat = transformed_cloud[..., 2].ravel()

        # 2. Z-필터 마스크 (GPU)
        mask = (z_flat > self.z_min_t) & (z_flat < self.z_max_t)

        # 3. 월드 좌표 -> 그리드 인덱스 변환 (GPU)
        # .long() == .to(torch.int64)
        grid_c = ((x_flat - self.grid_origin_x_t) / self.resolution_t).long()
        grid_r = ((y_flat - self.grid_origin_y_t) / self.resolution_t).long()

        # 4. 바운더리 체크 마스크 (GPU)
        mask &= (grid_c >= 0) & (grid_c < self.cells_x) & \
                (grid_r >= 0) & (grid_r < self.cells_y)

        # 5. 유효한 포인트만 필터링 (GPU)
        valid_z = z_flat[mask]
        if valid_z.shape[0] == 0:
            return # BEV 맵에 유효한 포인트 없음

        valid_r = grid_r[mask]
        valid_c = grid_c[mask]

        # 6. 2D 인덱스 -> 1D 선형 인덱스 (GPU)
        # (r, c) -> r * num_cols + c
        linear_indices = valid_r * self.cells_x + valid_c

        # 7. "Highest Point Wins" (GPU Scatter-Max)
        # 7.1. 재사용하는 높이 맵 텐서를 -inf로 초기화
        self.bev_heights_flat.fill_(-torch.inf)

        # 7.2. scatter_reduce_ (PyTorch 1.12+) 또는 index_reduce_
        # 동일한 'linear_indices'를 가진 'valid_z' 값들 중 최대값(amax)을
        # 'bev_heights_flat'의 해당 인덱스에 저장합니다.
        self.bev_heights_flat.index_reduce_(
            dim=0,
            index=linear_indices,
            source=valid_z,
            reduce="amax",
            include_self=False # fill_(-inf) 했으므로 기존 값 무시
        )

        # 8. 유효한 셀만 추출 (GPU)
        # -inf가 아닌, 즉 포인트가 하나라도 할당된 셀만 찾기
        valid_bev_mask = self.bev_heights_flat > -torch.inf

        # 유효한 셀의 1D 인덱스
        valid_indices_flat = torch.where(valid_bev_mask)[0]
        if valid_indices_flat.shape[0] == 0:
            return # 발행할 BEV 포인트 없음

        # 유효한 셀의 높이 값
        height_values = self.bev_heights_flat[valid_bev_mask]

        # 9. 1D 인덱스 -> 2D 인덱스 (GPU)
        r_idx_bev = torch.div(valid_indices_flat, self.cells_x, rounding_mode='floor')
        c_idx_bev = valid_indices_flat % self.cells_x

        # 10. BEV 포인트의 월드 좌표 계산 (GPU)
        # 각 셀의 중앙 좌표
        x_world = self.grid_origin_x_t + (c_idx_bev.float() + 0.5) * self.resolution_t
        y_world = self.grid_origin_y_t + (r_idx_bev.float() + 0.5) * self.resolution_t
        z_world = torch.zeros_like(x_world) # BEV 맵이므로 Z=0

        # 11. 높이(Z) 값 -> RGB 색상 변환 (GPU)
        rgb_float32_gpu = self._height_to_color_gpu(height_values)

        # 12. (X, Y, Z, RGB) 데이터 결합 (GPU)
        # .unsqueeze(1) : (N,) -> (N, 1)
        bev_data_gpu = torch.stack(
            [x_world, y_world, z_world, rgb_float32_gpu],
            dim=-1 # (N, 4)
        )

        # 13. GPU -> CPU 전송
        bev_data_np = bev_data_gpu.cpu().numpy()

        # 14. PointCloud2 메시지 생성 (CPU)
        bev_msg = self._create_cloud_from_data(
            bev_data_np, stamp, self.target_frame
        )

        # 15. 발행
        self.bev_pub.publish(bev_msg)


    def _height_to_color_gpu(self, z):
            """
            GPU 텐서(z)를 입력받아 'Jet' Colormap RGB 텐서를 반환합니다.
            (전체 벡터 연산) - Bitwise Shift가 아닌 곱셈 연산 사용
            """
            # 정규화 [0, 1] -> [0, 4]
            z_norm = (z - self.z_min_t) / self.z_range_t
            z_norm = torch.clamp(z_norm, 0.0, 1.0) * 4.0

            # float 텐서로 초기화
            r = torch.zeros_like(z_norm)
            g = torch.zeros_like(z_norm)
            b = torch.zeros_like(z_norm)

            # 마스크를 이용한 구간별 색상 계산
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

            # [0, 1] float -> [0, 255]
            # .long() (int64)으로 변환하여 곱셈/덧셈 준비
            r_val = (r * 255).long()
            g_val = (g * 255).long()
            b_val = (b * 255).long()

            # --- FIX ---
            # Bitwise shift (<<) 대신 곱셈 사용
            # (r_uint << 16) | (g_uint << 8) | b_uint  <- 이 연산이 CUDA에서 uint32로 지원 안 됨
            rgb_packed_gpu = (r_val * 65536) + (g_val * 256) + b_val

            # packed int64 -> uint32로 캐스팅 (데이터 손실 없음, 24비트만 사용하므로)
            rgb_uint32_gpu = rgb_packed_gpu.to(torch.uint32)
            # --- END FIX ---

            # .view(torch.float32) : uint32 -> float32 비트 재해석
            return rgb_uint32_gpu.view(torch.float32)



    def transform_to_matrix(self, transform):
        """ROS Transform 메시지를 4x4 동차 변환 행렬(NumPy)로 변환"""
        t = transform.transform.translation
        translation = np.array([t.x, t.y, t.z])

        r = transform.transform.rotation
        quat = [r.w, r.x, r.y, r.z] # transforms3d (w, x, y, z) 순서
        rotation_matrix = quat2mat(quat)

        matrix = np.eye(4)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = translation
        return matrix

    def create_pointcloud_msg(self, points_np, colors_np, stamp, frame_id):
        """
        (N, 3) points와 (N, 3) uint8 colors NumPy 배열로
        PointCloud2 메시지를 생성합니다. (PCL용)
        """
        header = Header(stamp=stamp, frame_id=frame_id)

        # 1. RGB 색상 패킹 (CPU)
        # (R, G, B) 3바이트 -> 4바이트 uint32 -> 4바이트 float32
        rgb_uint32 = (
            (colors_np[:, 0].astype(np.uint32) << 16) |
            (colors_np[:, 1].astype(np.uint32) << 8) |
            (colors_np[:, 2].astype(np.uint32))
        )
        rgb_float32 = rgb_uint32.view(np.float32)

        # 2. (N, 3) XYZ와 (N, 1) RGB(float32) 결합
        pointcloud_data = np.hstack([
            points_np.astype(np.float32),
            rgb_float32.reshape(-1, 1)
        ])

        # 3. 메시지 생성
        num_points = pointcloud_data.shape[0]
        return PointCloud2(
            header=header,
            height=1,
            width=num_points,
            fields=self.pointcloud_fields,
            is_bigendian=False,
            point_step=self.point_step, # 16
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
            point_step=self.point_step, # 16
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
