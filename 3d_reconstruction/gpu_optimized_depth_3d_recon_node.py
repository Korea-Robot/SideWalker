import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, TransformException
from transforms3d.quaternions import quat2mat
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time
from collections import deque

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False


class PointCloudReconstructionNode(Node):
    """
    Depth 이미지를 3D Point Cloud로 변환하고 TF를 적용하여 로봇 좌표계로 변환하는 노드
    GPU 가속 지원 (CUDA/PyTorch)
    """
    
    def __init__(self):
        super().__init__('pointcloud_reconstruction_node')
        
        # OpenCV와 ROS 이미지 변환을 위한 브리지
        self.bridge = CvBridge()
        
        # QoS 프로파일 설정 (안정적인 통신을 위해)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Depth 카메라 토픽 구독
        self.create_subscription(
            Image, 
            '/camera/camera/depth/image_rect_raw', 
            self.depth_callback, 
            qos_profile
        )
        
        # Point Cloud 발행자
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, 
            '/pointcloud', 
            qos_profile
        )
        
        # TF 변환을 위한 버퍼 및 리스너
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 좌표계 설정
        self.source_frame = 'camera_depth_optical_frame'
        self.target_frame = 'body'
        
        # 카메라 내부 Depth 파라미터 (Intel RealSense D455 기준)
        self.fx = 431.0625  # Focal length X
        self.fy = 431.0625  # Focal length Y
        self.cx = 434.492   # Principal point X
        self.cy = 242.764   # Principal point Y
        
        # 카메라 내부 RGB 파라미터 (Intel RealSense D455 기준)
        """
        height: 720
        width: 1280

        distortion_model: plumb_bob
        d:
        - -0.05555006489157677
        - 0.06587371975183487
        - 5.665919161401689e-05
        - 0.0014403886161744595
        - -0.02127622440457344

        k:
        - 645.4923095703125
        - 0.0
        - 653.0325927734375
        - 0.0
        - 644.4183349609375
        - 352.2890930175781
        """
        self.rgb_fx = 645.4923  # Focal length X
        self.rgb_fy = 644.4183  # Focal length Y
        self.rgb_cx = 653.03259   # Principal point X
        self.rgb_cy = 352.28909   # Principal point Y

        # 다운샘플링 비율
        self.downsample_y = 6 #12
        self.downsample_x = 4 #8 
        
        # Point Cloud 필드 정의
        self.pointcloud_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        # GPU 설정
        self.use_gpu = TORCH_AVAILABLE and CUDA_AVAILABLE
        if self.use_gpu:
            self.device =torch.device('cuda') # torch.device('cuda')
            self.get_logger().info('🚀 CUDA GPU 가속 활성화')
            # GPU에 카메라 파라미터 미리 올려놓기
            self._init_gpu_parameters()
        else:
            self.device = torch.device('cpu')
            if TORCH_AVAILABLE:
                self.get_logger().info('⚠️  CUDA 불가능 - CPU PyTorch 사용')
            else:
                self.get_logger().info('⚠️  PyTorch 없음 - NumPy 사용')
        
        # Latency 측정을 위한 변수
        self.timing_history = {
            'total': deque(maxlen=50),
            'conversion': deque(maxlen=50),
            'depth_to_pc': deque(maxlen=50),
            'tf_lookup': deque(maxlen=50),
            'transform': deque(maxlen=50),
            'downsample': deque(maxlen=50),
            'msg_create': deque(maxlen=50),
            'publish': deque(maxlen=50),
        }
        self.frame_count = 0
        self.last_report_time = time.time()
        
        self.get_logger().info('Point Cloud Reconstruction Node initialized')

    def _init_gpu_parameters(self):
        """GPU에서 사용할 파라미터 미리 생성"""
        # 이미지 크기 (480x848 기준)
        height, width = 480, 848
        
        # 픽셀 좌표 그리드 미리 생성
        v, u = torch.meshgrid(
            torch.arange(height, device=self.device, dtype=torch.float32),
            torch.arange(width, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        
        # 카메라 내부 파라미터 적용한 계수 미리 계산
        self.u_grid = u
        self.v_grid = v
        self.fx_tensor = torch.tensor(self.fx, device=self.device, dtype=torch.float32)
        self.fy_tensor = torch.tensor(self.fy, device=self.device, dtype=torch.float32)
        self.cx_tensor = torch.tensor(self.cx, device=self.device, dtype=torch.float32)
        self.cy_tensor = torch.tensor(self.cy, device=self.device, dtype=torch.float32)
        
        self.get_logger().info(f'GPU 파라미터 초기화 완료 ({height}x{width})')

    def depth_callback(self, msg):
        """Depth 이미지를 수신하여 Point Cloud로 변환하고 발행"""
        timings = {}
        t_start = time.perf_counter()
        
        try:
            # 1. Depth 이미지 변환
            t1 = time.perf_counter()
            depth_image = self.bridge.imgmsg_to_cv2(
                msg, 
                desired_encoding=msg.encoding
            ).astype(np.float32) / 1000.0
            timings['conversion'] = (time.perf_counter() - t1) * 1000
            
            # 2. Depth to Point Cloud
            t2 = time.perf_counter()
            if self.use_gpu:
                point_cloud = self.depth_to_pointcloud_gpu(depth_image)
            else:
                point_cloud = self.depth_to_pointcloud_cpu(depth_image)
            timings['depth_to_pc'] = (time.perf_counter() - t2) * 1000
            
            # 3. TF 조회
            t3 = time.perf_counter()
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time()
            )
            transform_matrix = self.transform_to_matrix(transform)
            timings['tf_lookup'] = (time.perf_counter() - t3) * 1000
            
            # 4. 변환 적용
            t4 = time.perf_counter()
            if self.use_gpu:
                transformed_cloud = self.apply_transform_gpu(point_cloud, transform_matrix)
            else:
                transformed_cloud = self.apply_transform_cpu(point_cloud, transform_matrix)
            timings['transform'] = (time.perf_counter() - t4) * 1000
            
            # 5. 다운샘플링 및 색상
            t5 = time.perf_counter()
            if self.use_gpu:
                points, colors = self.process_pointcloud_gpu(transformed_cloud)
            else:
                points, colors = self.process_pointcloud_cpu(transformed_cloud)
            timings['downsample'] = (time.perf_counter() - t5) * 1000
            
            # 6. 메시지 생성
            t6 = time.perf_counter()
            pointcloud_msg = self.create_pointcloud_msg(
                points, 
                colors, 
                self.target_frame
            )
            timings['msg_create'] = (time.perf_counter() - t6) * 1000
            
            # 7. 발행
            t7 = time.perf_counter()
            self.pointcloud_pub.publish(pointcloud_msg)
            timings['publish'] = (time.perf_counter() - t7) * 1000
            
            # 전체 시간
            timings['total'] = (time.perf_counter() - t_start) * 1000
            
            # 타이밍 기록
            self.record_timings(timings)
            
        except TransformException as e:
            self.get_logger().warn(f'TF 변환 실패: {e}', throttle_duration_sec=1.0)
        except Exception as e:
            self.get_logger().error(f'Point Cloud 처리 오류: {e}')

    def record_timings(self, timings):
        """타이밍 정보 기록 및 주기적 출력"""
        for key, value in timings.items():
            self.timing_history[key].append(value)
        
        self.frame_count += 1
        
        # 2초마다 통계 출력
        current_time = time.time()
        if current_time - self.last_report_time >= 2.0:
            self.print_timing_stats()
            self.last_report_time = current_time

    def print_timing_stats(self):
        """타이밍 통계 출력"""
        if self.frame_count == 0:
            return
        
        stats_msg = [
            f"\n{'='*60}",
            f"📊 Performance Stats (최근 {len(self.timing_history['total'])} frames)",
            f"{'='*60}",
            f"Backend: {'GPU (CUDA)' if self.use_gpu else 'CPU (NumPy)'}"
        ]
        
        for key in ['total', 'conversion', 'depth_to_pc', 'tf_lookup', 
                    'transform', 'downsample', 'msg_create', 'publish']:
            if self.timing_history[key]:
                times = list(self.timing_history[key])
                avg = np.mean(times)
                std = np.std(times)
                min_t = np.min(times)
                max_t = np.max(times)
                
                label_map = {
                    'total': '🔴 전체',
                    'conversion': '  ├─ 이미지 변환',
                    'depth_to_pc': '  ├─ Depth→PC 변환',
                    'tf_lookup': '  ├─ TF 조회',
                    'transform': '  ├─ 좌표 변환',
                    'downsample': '  ├─ 다운샘플링',
                    'msg_create': '  ├─ 메시지 생성',
                    'publish': '  └─ 발행',
                }
                
                stats_msg.append(
                    f"{label_map.get(key, key):20} "
                    f"avg: {avg:6.2f}ms  "
                    f"std: {std:5.2f}ms  "
                    f"[{min_t:5.2f} ~ {max_t:6.2f}]ms"
                )
        
        fps = len(self.timing_history['total']) / 2.0
        stats_msg.append(f"{'='*60}")
        stats_msg.append(f"FPS: {fps:.1f} Hz")
        stats_msg.append(f"{'='*60}\n")
        
        self.get_logger().info('\n'.join(stats_msg))

    def depth_to_pointcloud_gpu(self, depth_map):
        """GPU를 이용한 Depth to Point Cloud 변환"""
        # NumPy → Torch Tensor (GPU)
        depth_tensor = torch.from_numpy(depth_map).to(self.device)
        
        # 3D 좌표 계산
        z = depth_tensor
        x = (self.u_grid - self.cx_tensor) * z / self.fx_tensor
        y = (self.v_grid - self.cy_tensor) * z / self.fy_tensor
        
        # Stack (H, W, 3)
        pointcloud = torch.stack([x, y, z], dim=-1)
        
        return pointcloud

    def depth_to_pointcloud_cpu(self, depth_map):
        """CPU NumPy를 이용한 Depth to Point Cloud 변환"""
        height, width = depth_map.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        z = depth_map
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        return np.stack((x, y, z), axis=-1)

    def apply_transform_gpu(self, points, matrix):
        """GPU를 이용한 좌표 변환"""
        original_shape = points.shape
        points_flat = points.reshape(-1, 3)
        
        # Transform matrix를 GPU로
        matrix_tensor = torch.from_numpy(matrix).to(self.device, dtype=torch.float32)
        
        # 동차 좌표 (N, 4)
        ones = torch.ones((points_flat.shape[0], 1), device=self.device, dtype=torch.float32)
        homogeneous = torch.cat([points_flat, ones], dim=1)
        
        # 변환 (N, 4) @ (4, 4)^T = (N, 4)
        transformed = torch.mm(homogeneous, matrix_tensor.T)
        
        # 원래 shape 복원
        return transformed[:, :3].reshape(original_shape)

    def apply_transform_cpu(self, points, matrix):
        """CPU NumPy를 이용한 좌표 변환"""
        original_shape = points.shape
        points_flat = points.reshape(-1, 3)
        
        ones = np.ones((points_flat.shape[0], 1))
        homogeneous_points = np.hstack((points_flat, ones))
        
        transformed = homogeneous_points @ matrix.T
        
        return transformed[:, :3].reshape(original_shape)

    def process_pointcloud_gpu(self, pointcloud):
        """GPU를 이용한 다운샘플링 및 색상 적용"""
        # 다운샘플링 (slicing은 GPU에서도 빠름)
        sampled = pointcloud[::self.downsample_y, ::self.downsample_x, :]
        
        # Flatten
        points = sampled.reshape(-1, 3)
        
        # GPU에서 CPU로 이동
        points_np = points.cpu().numpy()
        
        # 색상 생성 (작은 배열이므로 CPU에서)
        num_points = points_np.shape[0]
        colors = np.zeros((num_points, 3), dtype=np.uint8)
        colors[:, 0] = 200  # R
        colors[:, 1] = 100  # G
        colors[:, 2] = 208  # B
        
        return points_np, colors

    def process_pointcloud_cpu(self, pointcloud):
        """CPU를 이용한 다운샘플링 및 색상 적용"""
        x_coords = pointcloud[::self.downsample_y, ::self.downsample_x, 0]
        y_coords = pointcloud[::self.downsample_y, ::self.downsample_x, 1]
        z_coords = pointcloud[::self.downsample_y, ::self.downsample_x, 2]
        
        points = np.stack((x_coords, y_coords, z_coords), axis=-1)
        points = points.reshape(-1, 3)
        
        num_points = points.shape[0]
        colors = np.zeros((num_points, 3), dtype=np.uint8)
        colors[:, 0] = 200
        colors[:, 1] = 100
        colors[:, 2] = 208
        
        return points, colors

    def transform_to_matrix(self, transform):
        """ROS Transform을 4x4 동차 변환 행렬로 변환"""
        t = transform.transform.translation
        translation = np.array([t.x, t.y, t.z])
        
        r = transform.transform.rotation
        quat = [r.w, r.x, r.y, r.z]
        rotation_matrix = quat2mat(quat)
        
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = translation
        
        return matrix

    def create_pointcloud_msg(self, points, colors, frame_id):
        """NumPy 배열을 ROS PointCloud2 메시지로 변환"""
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        
        # RGB 패킹
        rgb_uint32 = (
            (colors[:, 0].astype(np.uint32) << 16) |
            (colors[:, 1].astype(np.uint32) << 8) |
            (colors[:, 2].astype(np.uint32))
        )
        rgb_float32 = rgb_uint32.view(np.float32)
        
        # XYZ + RGB 결합
        pointcloud_data = np.hstack([
            points.astype(np.float32), 
            rgb_float32.reshape(-1, 1)
        ])
        
        return PointCloud2(
            header=header,
            height=1,
            width=pointcloud_data.shape[0],
            fields=self.pointcloud_fields,
            is_bigendian=False,
            point_step=16,
            row_step=16 * pointcloud_data.shape[0],
            data=pointcloud_data.tobytes(),
            is_dense=True,
        )


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    node = PointCloudReconstructionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
