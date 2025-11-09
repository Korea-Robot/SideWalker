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
import torch
import struct

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

        # --- 2. ROS 파라미터 선언 ---
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('pointcloud_topic', '/pointcloud')
        self.declare_parameter('source_frame', 'camera_depth_optical_frame')
        self.declare_parameter('target_frame', 'camera_link')

        # Intrinsic
        self.declare_parameter('cam.fx', 431.0625)
        self.declare_parameter('cam.fy', 431.0625)
        self.declare_parameter('cam.cx', 434.492)
        self.declare_parameter('cam.cy', 242.764)
        self.declare_parameter('cam.height', 480)
        self.declare_parameter('cam.width', 848)

        # PCL 다운샘플링
        self.declare_parameter('pcl.downsample_y', 3)
        self.declare_parameter('pcl.downsample_x', 2)

        # BEV 파라미터
        self.declare_parameter('bev_topic', '/bev_map')
        self.declare_parameter('bev.z_min', 0.1)
        self.declare_parameter('bev.z_max', 1.0)
        self.declare_parameter('bev.resolution', 0.05)
        self.declare_parameter('bev.size_x', 30.0)
        self.declare_parameter('bev.size_y', 40.0)

        # [NEW] 로봇 안전 영역 설정을 위한 파라미터 (BEV Grid가 이 영역을 포함해야 함)
        # 카메라가 (0,0)일 때 로봇은 뒤쪽에 위치하므로 음수 좌표 사용
        # 로봇 크기 1m x 1m, 카메라가 로봇 맨 앞 중앙에 있다고 가정
        self.declare_parameter('robot.safe_min_x', -1.0) # 카메라 뒤 1m
        self.declare_parameter('robot.safe_max_x', 0.0)  # 카메라 위치까지
        self.declare_parameter('robot.safe_min_y', -0.5) # 왼쪽 0.5m
        self.declare_parameter('robot.safe_max_y', 0.5)  # 오른쪽 0.5m

        # --- 3. 파라미터 값 할당 ---
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
        
        # [중요 변경] 로봇이 카메라 뒤에 있다면, BEV 그리드도 뒤쪽을 포함해야 합니다.
        # 예: 그리드 시작점을 -5.0m로 설정하여 카메라 뒤쪽도 보이게 함
        # 만약 0.0으로 두면 로봇 본체는 그리드 아예 밖에 존재하게 됩니다.
        # 사용자의 의도에 맞게 "로봇 영역을 표현"하려면 원점을 뒤로 당겨야 합니다.
        self.grid_origin_x = -2.0  # 카메라 뒤 2m 부터 그리드 시작 (수정 제안)
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

        # --- 5. Point Cloud 필드 정의 ---
        self.pointcloud_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        self.point_step = 16

        # --- 6. GPU 파라미터 초기화 ---
        self._init_gpu_parameters()

        self.get_logger().info('✅ PointCloud + BEV Node initialized (GPU Only)')
        self.get_logger().info(f"  BEV Grid Origin X: {self.grid_origin_x} m (Must be < 0 to see robot body)")




    def _init_gpu_parameters(self):
        """GPU 파라미터 및 Virtual Fence(경계선)가 그려진 기본 맵 생성"""
        # ... (이전과 동일한 그리드/파라미터 초기화) ...
        # 1. PCL 재구성 그리드
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

        # 2. BEV 파라미터
        self.z_min_t = torch.tensor(self.z_min, device=self.device, dtype=torch.float32)
        self.z_max_t = torch.tensor(self.z_max, device=self.device, dtype=torch.float32)
        self.z_range_t = self.z_max_t - self.z_min_t
        self.resolution_t = torch.tensor(self.resolution, device=self.device, dtype=torch.float32)
        self.grid_origin_x_t = torch.tensor(self.grid_origin_x, device=self.device, dtype=torch.float32)
        self.grid_origin_y_t = torch.tensor(self.grid_origin_y, device=self.device, dtype=torch.float32)

        # --- [핵심 변경] Virtual Fence 생성 시작 ---

        # 3. 기본 맵을 'Free(z_min)'로 초기화 (2D 형태로 작업 후 flatten)
        self.default_bev_grid = torch.full(
            (self.cells_y, self.cells_x),
            self.z_min, # 기본은 모두 주행 가능
            device=self.device,
            dtype=torch.float32
        )

        # 4. 로봇 펜스 (Robot Boundary Line) 그리기
        # 월드 좌표 -> 그리드 좌표 변환 함수 (클램핑 포함)
        def to_grid_x(world_x):
            idx = int((world_x-self.grid_origin_x)/ self.resolution)
            return max(0,min(idx,self.cells_x-1)) 
            # return torch.clamp(((world_x - self.grid_origin_x) / self.resolution).long(), 0, self.cells_x - 1)
        def to_grid_y(world_y): 
            idx = int((world_y-self.grid_origin_y)/ self.resolution)
            return max(0,min(idx,self.cells_y-1)) 
            # return torch.clamp(((world_y - self.grid_origin_y) / self.resolution).long(), 0, self.cells_y - 1)

        r_min_x = self.get_parameter('robot.safe_min_x').value
        r_max_x = self.get_parameter('robot.safe_max_x').value
        r_min_y = self.get_parameter('robot.safe_min_y').value
        r_max_y = self.get_parameter('robot.safe_max_y').value

        gx_min = to_grid_x(r_min_x); gx_max = to_grid_x(r_max_x)
        gy_min = to_grid_y(r_min_y); gy_max = to_grid_y(r_max_y)

        # 테두리 그리기 (상하좌우 라인)
        self.default_bev_grid[gy_min:gy_max+1, gx_min] = self.z_max_t # 뒤쪽 라인
        self.default_bev_grid[gy_min:gy_max+1, gx_max] = self.z_max_t # 앞쪽 라인
        self.default_bev_grid[gy_min, gx_min:gx_max+1] = self.z_max_t # 오른쪽 라인
        self.default_bev_grid[gy_max, gx_min:gx_max+1] = self.z_max_t # 왼쪽 라인

        # 5. FOV 펜스 (FOV Boundary Line) 그리기
        # 로봇 좌표계 기준: X가 전방, Y가 좌측
        # FOV 라인을 따라 점들을 생성하고 그리드에 찍습니다.
        num_points = 2000 # 라인을 조밀하게 그리기 위한 점 개수
        x_r = torch.linspace(0, self.size_x, num_points, device=self.device) # 로봇 전방 0m ~ 최대 거리

        # 왼쪽 FOV 경계선 (이미지 u=0) -> 로봇 좌표계 y_r 계산
        # 카메라 좌표계: xc = -cx * zc / fx
        # 로봇 좌표계 변환(xc->-yr, zc->xr): -yr = -cx * xr / fx  => yr = (cx / fx) * xr
        y_r_left = (self.cx_tensor / self.fx_tensor) * x_r

        # 오른쪽 FOV 경계선 (이미지 u=width)
        # 카메라 좌표계: xc = (width - cx) * zc / fx
        # 로봇 좌표계 변환: -yr = (width - cx) * xr / fx => yr = -((width - cx) / fx) * xr
        y_r_right = -((self.cam_width - self.cx_tensor) / self.fx_tensor) * x_r

        # 그리드 좌표로 변환
        gx_fov = ((x_r - self.grid_origin_x_t) / self.resolution_t).long()
        gy_left = ((y_r_left - self.grid_origin_y_t) / self.resolution_t).long()
        gy_right = ((y_r_right - self.grid_origin_y_t) / self.resolution_t).long()

        # 유효한 그리드 범위 내의 점만 필터링하여 찍기
        mask_l = (gx_fov >= 0) & (gx_fov < self.cells_x) & (gy_left >= 0) & (gy_left < self.cells_y)
        self.default_bev_grid[gy_left[mask_l], gx_fov[mask_l]] = self.z_max_t

        mask_r = (gx_fov >= 0) & (gx_fov < self.cells_x) & (gy_right >= 0) & (gy_right < self.cells_y)
        self.default_bev_grid[gy_right[mask_r], gx_fov[mask_r]] = self.z_max_t

        # 6. 최종 기본 맵 Flatten (재사용을 위해)
        self.default_bev_flat = self.default_bev_grid.flatten()

        # --- Virtual Fence 생성 끝 ---

        # 측정용 임시 버퍼
        self.measured_bev_flat = torch.empty_like(self.default_bev_flat)
        self.bev_heights_flat = torch.empty_like(self.default_bev_flat) # 혹시 몰라 추가

        # 좌표 변환 행렬 (Camera -> Robot)
        self.transform_matrix = np.array([
            [0.,0.,1.,0.0],  # Cam Z -> Robot X (Forward)
            [-1.,0.,0.,0.], # Cam X -> Robot -Y (Left)
            [0.,-1.,0.,0.], # Cam Y -> Robot -Z (Up)
            [0.,0.,0.,1.]
        ], dtype=np.float32)

        self.get_logger().info('✅ GPU 파라미터 및 Virtual Fence 초기화 완료')



    def depth_callback(self, msg):
        # ... (기존과 동일) ...
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding).astype(np.float32) / 1000.0
            depth_tensor = torch.from_numpy(depth_image).to(self.device)
            pointcloud_cam = self.depth_to_pointcloud_gpu(depth_tensor)
            transformed_cloud = self.apply_transform_gpu(pointcloud_cam, self.transform_matrix)

            stamp = msg.header.stamp
            self.process_and_publish_pointcloud(transformed_cloud, stamp)
            self.process_and_publish_bev(transformed_cloud, stamp) # 변경된 함수 호출

        except Exception as e:
             self.get_logger().error(f'Error: {e}')

    # ... (depth_to_pointcloud_gpu, apply_transform_gpu, process_and_publish_pointcloud는 동일) ...
    def depth_to_pointcloud_gpu(self, depth_tensor):
        z = depth_tensor
        x = (self.u_grid - self.cx_tensor) * z / self.fx_tensor 
        y = (self.v_grid - self.cy_tensor) * z / self.fy_tensor
        return torch.stack([x, y, z], dim=-1)

    def apply_transform_gpu(self, points, matrix):
        original_shape = points.shape
        points_flat = points.reshape(-1, 3)
        matrix_tensor = torch.from_numpy(matrix).to(self.device, dtype=torch.float32)
        homogeneous = torch.cat([points_flat, torch.ones((points_flat.shape[0], 1), device=self.device)], dim=1)
        transformed = torch.mm(homogeneous, matrix_tensor.T)
        return transformed[:, :3].reshape(original_shape)

    def process_and_publish_pointcloud(self, transformed_cloud, stamp):
        sampled = transformed_cloud[::self.downsample_y, ::self.downsample_x, :]
        points = sampled.reshape(-1, 3)
        points_np = points.cpu().numpy()
        if points_np.shape[0] == 0: return
        colors = np.zeros((points_np.shape[0], 3), dtype=np.uint8)
        colors[:, 0] = 200; colors[:, 1] = 100; colors[:, 2] = 200
        msg = self.create_pointcloud_msg(points_np, colors, stamp, self.target_frame)
        self.pointcloud_pub.publish(msg)

    def process_and_publish_bev(self, transformed_cloud, stamp):
        """
        BEV 맵 생성: FOV 밖은 Occupied, 로봇 영역은 Free, 관측된 영역은 측정값 사용.
        """
        # 1. Point Cloud Flatten
        x_flat = transformed_cloud[..., 0].ravel()
        y_flat = transformed_cloud[..., 1].ravel()
        z_flat = transformed_cloud[..., 2].ravel()

        # 2. 유효한 포인트 필터링 (높이 및 그리드 범위)
        mask = (z_flat > self.z_min_t) & (z_flat < self.z_max_t)
        grid_c = ((x_flat - self.grid_origin_x_t) / self.resolution_t).long()
        grid_r = ((y_flat - self.grid_origin_y_t) / self.resolution_t).long()
        mask &= (grid_c >= 0) & (grid_c < self.cells_x) & \
                (grid_r >= 0) & (grid_r < self.cells_y)

        valid_z = z_flat[mask]
        valid_r = grid_r[mask]
        valid_c = grid_c[mask]

        # 3. [핵심 변경] 기본 맵 복사 (FOV 밖=Occupied, 로봇=Free 상태)
        # 매 프레임마다 final_bev_flat을 default 상태로 시작합니다.
        final_bev_flat = self.default_bev_flat.clone()

        # 관측 데이터가 있을 경우에만 덮어쓰기 진행
        if valid_z.shape[0] > 0:
            linear_indices = valid_r * self.cells_x + valid_c

            # 4. 현재 프레임 측정값 계산
            # 측정용 버퍼를 -inf로 초기화 (관측되지 않음을 의미)
            self.measured_bev_flat.fill_(-torch.inf)
            
            # 관측된 위치 중 가장 높은 값(amax) 저장
            self.measured_bev_flat.index_reduce_(
                dim=0,
                index=linear_indices,
                source=valid_z, 
                reduce="amax",
                include_self=False
            )

            # 5. 기본 맵 위에 측정값 덮어쓰기
            # 측정된 값(-inf가 아닌 값)이 있는 셀만 마스킹
            observed_mask = self.measured_bev_flat > -torch.inf
            # 해당 셀들을 측정된 실제 높이 값으로 교체
            final_bev_flat[observed_mask] = self.measured_bev_flat[observed_mask]

        # --- 이하 발행 로직은 동일하지만, final_bev_flat을 사용 ---
        
        # BEV 전체를 발행하면 데이터가 너무 커질 수 있으니, 
        # 필요하다면 여기서도 -inf(완전 미지의 영역)인 부분은 제외할 수 있습니다.
        # 하지만 요구사항이 "FOV 밖도 Occupied로 채워달라"는 것이므로
        # z_max로 채워진 final_bev_flat 전체를 발행하거나, 값이 있는 곳만 발행합니다.
        # 여기서는 효율성을 위해 Occupied(z_max) 또는 Free(z_min) 또는 측정값이 있는 곳 모두 발행합니다.
        # (사실상 그리드 전체 발행이 될 수 있음. 부하가 크면 조절 필요)
        
        # 예시: 모든 유효한 셀 발행 (배경 포함)
        # 만약 대역폭이 문제라면, occupied인 부분만 발행하는 것도 방법입니다.
        valid_indices_flat = torch.arange(final_bev_flat.shape[0], device=self.device)
        height_values = final_bev_flat

        # 1D -> 2D 인덱스
        r_idx_bev = torch.div(valid_indices_flat, self.cells_x, rounding_mode='floor')
        c_idx_bev = valid_indices_flat % self.cells_x

        # 월드 좌표 계산
        x_world = self.grid_origin_x_t + (c_idx_bev.float() + 0.5) * self.resolution_t
        y_world = self.grid_origin_y_t + (r_idx_bev.float() + 0.5) * self.resolution_t
        z_world = torch.zeros_like(x_world)

        # 색상 변환 및 메시지 생성
        rgb_float32_gpu = self._height_to_color_gpu(height_values)
        bev_data_gpu = torch.stack([x_world, y_world, z_world, rgb_float32_gpu], dim=-1)
        bev_msg = self._create_cloud_from_data(bev_data_gpu.cpu().numpy(), stamp, self.target_frame)
        self.bev_pub.publish(bev_msg)

    # ... (_height_to_color_gpu, transform_to_matrix, create_pointcloud_msg 등 기존 메서드 유지) ...
    def _height_to_color_gpu(self, z):
            z_norm = (z - self.z_min_t) / self.z_range_t
            z_norm = torch.clamp(z_norm, 0.0, 1.0) * 4.0
            r = torch.zeros_like(z_norm); g = torch.zeros_like(z_norm); b = torch.zeros_like(z_norm)
            mask = z_norm < 1.0
            b[mask] = 1.0; g[mask] = z_norm[mask]
            mask = (z_norm >= 1.0) & (z_norm < 2.0)
            g[mask] = 1.0; b[mask] = 2.0 - z_norm[mask]
            mask = (z_norm >= 2.0) & (z_norm < 3.0)
            g[mask] = 1.0; r[mask] = z_norm[mask] - 2.0
            mask = z_norm >= 3.0
            r[mask] = 1.0; g[mask] = 4.0 - z_norm[mask]
            rgb_packed_gpu = ((r * 255).long() * 65536) + ((g * 255).long() * 256) + (b * 255).long()
            return rgb_packed_gpu.to(torch.uint32).view(torch.float32)

    def create_pointcloud_msg(self, points_np, colors_np, stamp, frame_id):
        header = Header(stamp=stamp, frame_id=frame_id)
        rgb_uint32 = ((colors_np[:, 0].astype(np.uint32) << 16) | (colors_np[:, 1].astype(np.uint32) << 8) | (colors_np[:, 2].astype(np.uint32)))
        pointcloud_data = np.hstack([points_np.astype(np.float32), rgb_uint32.view(np.float32).reshape(-1, 1)])
        return PointCloud2(header=header, height=1, width=pointcloud_data.shape[0], fields=self.pointcloud_fields, is_bigendian=False, point_step=self.point_step, row_step=self.point_step * pointcloud_data.shape[0], data=pointcloud_data.tobytes(), is_dense=True)

    def _create_cloud_from_data(self, point_data_np, stamp, frame_id):
        return PointCloud2(header=Header(stamp=stamp, frame_id=frame_id), height=1, width=point_data_np.shape[0], fields=self.pointcloud_fields, is_bigendian=False, point_step=self.point_step, row_step=self.point_step * point_data_np.shape[0], data=point_data_np.astype(np.float32).tobytes(), is_dense=True)

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudBEVNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
