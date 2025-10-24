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
import time

# 계산량 53배 감소.

class PointCloudReconstructionNode(Node):
    """
    Depth 이미지를 3D Point Cloud로 변환하고 TF를 적용하여 로봇 좌표계로 변환하는 노드
    (PyTorch CUDA GPU 가속 전용 버전)
    
    [최적화 적용됨]
    1. 정적 TF (Extrinsic Matrix) 1회 조회
    2. 뎁스 이미지 사전 다운샘플링 후 GPU 연산
    """
    
    def __init__(self):
        super().__init__('pointcloud_reconstruction_node')
        
        # OpenCV와 ROS 이미지 변환을 위한 브리지
        self.bridge = CvBridge()
        
        # QoS 프로파일 설정
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, 
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
        
        # 좌표계 설정
        self.source_frame = 'camera_depth_optical_frame' # 원본 좌표계 (카메라)
        # self.target_frame = 'camera_link' # 대상 좌표계 (로봇 베이스)
        self.target_frame = 'body' # 대상 좌표계 (로봇 베이스)
        
        # 카메라 내부 파라미터 (Intel RealSense 기준 - 848x480)
        self.fx = 431.0625 # X축 초점 거리
        self.fy = 431.0625 # Y축 초점 거리
        self.cx = 434.492  # 주점 X좌표
        self.cy = 242.764  # 주점 Y좌표
        
        # 다운샘플링 비율 (Y축으로 9픽셀마다, X축으로 6픽셀마다)
        self.downsample_y = 9
        self.downsample_x = 6
        
        # Point Cloud 필드 정의 (포인트당 16바이트)
        self.pointcloud_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        # GPU 설정 (CUDA 사용 고정)
        self.device = torch.device('cuda')
        self.get_logger().info('🚀 CUDA GPU 가속 활성화 (PyTorch 사용)')
        
        # GPU에서 사용할 파라미터 미리 계산하여 로드 (★최적화 2 적용됨)
        # 다운샘플링된 그리드를 미리 생성
        self._init_gpu_parameters()
        
        # TF 변환을 위한 버퍼 및 리스너
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # ★최적화 1: 정적 TF (Extrinsic Matrix)를 1회만 조회하여 저장
        self.extrinsic_matrix = None # 외부 파라미터 행렬
        self.get_logger().info(f"'{self.source_frame}' -> '{self.target_frame}' 정적 TF 대기 중...")
        
        while self.extrinsic_matrix is None and rclpy.ok():
            try:
                # TF 트리가 빌드될 때까지 1초마다 재시도
                transform_stamped = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    self.source_frame,
                    rclpy.time.Time(), # 가장 최신 TF
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
                # 조회 성공 시, 4x4 행렬로 변환하여 저장
                self.extrinsic_matrix = self.transform_to_matrix(transform_stamped)
                self.get_logger().info('✅ 정적 TF (Extrinsic Matrix) 조회 성공!')
            except TransformException as e:
                self.get_logger().warn('정적 TF 대기 중... (1초 후 재시도)')
                time.sleep(1.0) # rclpy.spin() 전이므로 time.sleep 사용
                
        if self.extrinsic_matrix is None:
            self.get_logger().error('정적 TF 조회 실패! 노드를 종료합니다.')
            rclpy.shutdown() # TF 없으면 실행 불가
            return
            
        self.get_logger().info('Point Cloud Reconstruction Node initialized (Optimized)')

    def _init_gpu_parameters(self):
        """
        GPU에서 사용할 파라미터 미리 생성
        (★최적화 2: 다운샘플링된 그리드를 생성)
        """
        
        # 원본 이미지 크기
        height, width = 480, 848
        
        # 픽셀 좌표(v, u)의 전체 그리드를 생성
        v, u = torch.meshgrid(
            torch.arange(height, device=self.device, dtype=torch.float32),
            torch.arange(width, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        
        # ★최적화: 전체 그리드를 다운샘플링 비율로 슬라이싱
        v_grid_sampled = v[::self.downsample_y, ::self.downsample_x]
        u_grid_sampled = u[::self.downsample_y, ::self.downsample_x]
        
        # 3D 계산에 필요한 상수들을 미리 GPU 텐서로 만들어 둠
        self.u_grid = u_grid_sampled
        self.v_grid = v_grid_sampled
        self.fx_tensor = torch.tensor(self.fx, device=self.device, dtype=torch.float32)
        self.fy_tensor = torch.tensor(self.fy, device=self.device, dtype=torch.float32)
        self.cx_tensor = torch.tensor(self.cx, device=self.device, dtype=torch.float32)
        self.cy_tensor = torch.tensor(self.cy, device=self.device, dtype=torch.float32)
        
        self.get_logger().info(f'GPU 파라미터 초기화 완료 (다운샘플링 그리드: {self.v_grid.shape})')

    def depth_callback(self, msg):
        """Depth 이미지를 수신하여 Point Cloud로 변환하고 발행"""
        
        try:
            # 1. Depth 이미지 변환 (ROS Image -> NumPy)
            depth_image_full = self.bridge.imgmsg_to_cv2(
                msg, 
                desired_encoding=msg.encoding
            ).astype(np.float32) / 1000.0
            
            # ★최적화 2: GPU로 보내기 전, CPU에서 뎁스 이미지를 다운샘플링
            depth_image_sampled = depth_image_full[::self.downsample_y, ::self.downsample_x]
            
            # 2. Depth to Point Cloud (GPU)
            # ★최적화: 다운샘플링된 작은 이미지만 GPU로 전송하여 계산
            point_cloud = self.depth_to_pointcloud_gpu(depth_image_sampled)
            
            # 3. TF 조회 (CPU)
            # ★최적화 1: `__init__`에서 미리 계산한 행렬을 즉시 사용
            # (삭제) transform = self.tf_buffer.lookup_transform(...)
            # (삭제) transform_matrix = self.transform_to_matrix(transform)
            
            # 4. 변환 적용 (GPU)
            transformed_cloud = self.apply_transform_gpu(point_cloud, self.extrinsic_matrix)
            
            # 5. 색상 적용 (GPU -> CPU)
            # ★최적화: 이미 다운샘플링되었으므로 추가 슬라이싱 불필요
            points, colors = self.process_pointcloud_gpu(transformed_cloud)
            
            # 6. PointCloud2 메시지 생성 (CPU)
            pointcloud_msg = self.create_pointcloud_msg(
                points, 
                colors, 
                self.target_frame # 헤더의 frame_id는 변환된 'body'
            )
            
            # 7. 발행
            self.pointcloud_pub.publish(pointcloud_msg)
            
        except TransformException as e:
            # (이 코드는 정적 TF를 사용하므로, 초기화 실패 외에는 거의 발생 안 함)
            self.get_logger().warn(f'TF 변환 실패: {e}', throttle_duration_sec=1.0)
        except Exception as e:
            self.get_logger().error(f'Point Cloud 처리 오류: {e}')

    def depth_to_pointcloud_gpu(self, depth_map_sampled):
        """
        GPU를 이용한 Depth to Point Cloud 변환
        (입력: 다운샘플링된 뎁스 맵)
        """
        
        # 1. NumPy(CPU) 배열을 PyTorch GPU 텐서로 복사
        # (원본보다 훨씬 작은 텐서가 복사됨)
        depth_tensor = torch.from_numpy(depth_map_sampled).to(self.device)
        
        # 2. 3D 좌표 계산 (Pinhole Camera Model)
        # z, u_grid, v_grid, ... 모두 다운샘플링된 동일한 shape (예: 54x142)
        z = depth_tensor
        x = (self.u_grid - self.cx_tensor) * z / self.fx_tensor
        y = (self.v_grid - self.cy_tensor) * z / self.fy_tensor
        
        # 3. (x, y, z) 텐서를 스택
        # 결과: (H_sampled, W_sampled, 3) 형태의 텐서
        pointcloud = torch.stack([x, y, z], dim=-1)
        
        return pointcloud

    def apply_transform_gpu(self, points, matrix):
        """GPU를 이용한 좌표 변환"""
        
        # 원본 형태 (H_sampled, W_sampled, 3) 저장
        original_shape = points.shape
        # (H_sampled, W_sampled, 3) -> (N_sampled, 3)으로 변환
        points_flat = points.reshape(-1, 3)
        
        # 1. 변환 행렬(NumPy)을 GPU 텐서로 복사
        matrix_tensor = torch.from_numpy(matrix).to(self.device, dtype=torch.float32)
        
        # 2. 동차 좌표(Homogeneous Coordinates) 생성
        ones = torch.ones((points_flat.shape[0], 1), device=self.device, dtype=torch.float32)
        homogeneous = torch.cat([points_flat, ones], dim=1)
        
        # 3. 행렬 곱셈으로 변환 적용
        transformed = torch.mm(homogeneous, matrix_tensor.T)
        
        # 4. 원본 형태로 복원
        return transformed[:, :3].reshape(original_shape)

    def process_pointcloud_gpu(self, pointcloud):
        """GPU -> CPU 변환 및 색상 적용"""
        
        # 1. 다운샘플링 (GPU)
        # ★최적화: 이 단계는 뎁스 이미지 사전 처리로 이동됨.
        # (삭제) sampled = pointcloud[::self.downsample_y, ::self.downsample_x, :]
        
        # 2. Flatten (GPU)
        # (입력 'pointcloud'가 이미 다운샘플링된 상태)
        points = pointcloud.reshape(-1, 3)
        
        # 3. GPU -> CPU로 데이터 이동
        points_np = points.cpu().numpy()
        
        # 4. 색상 생성 (CPU)
        num_points = points_np.shape[0]
        colors = np.zeros((num_points, 3), dtype=np.uint8)
        colors[:, 0] = 200 # R
        colors[:, 1] = 100 # G
        colors[:, 2] = 208 # B
        
        return points_np, colors

    def transform_to_matrix(self, transform):
        """ROS Transform 메시지를 4x4 동차 변환 행렬(NumPy)로 변환"""
        
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
        
        # RGB 색상 패킹
        rgb_uint32 = (
            (colors[:, 0].astype(np.uint32) << 16) |
            (colors[:, 1].astype(np.uint32) << 8) |
            (colors[:, 2].astype(np.uint32))
        )
        
        # uint32를 float32로 비트 레벨에서 재해석
        rgb_float32 = rgb_uint32.view(np.float32)
        
        # XYZ와 RGB 결합
        pointcloud_data = np.hstack([
            points.astype(np.float32), 
            rgb_float32.reshape(-1, 1)
        ])
        
        # PointCloud2 메시지 생성
        return PointCloud2(
            header=header,
            height=1, 
            width=pointcloud_data.shape[0], # 포인트 개수
            fields=self.pointcloud_fields, 
            is_bigendian=False,
            point_step=16, # 1개 포인트가 차지하는 바이트 (16)
            row_step=16 * pointcloud_data.shape[0], 
            data=pointcloud_data.tobytes(), 
            is_dense=True, 
        )


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    node = PointCloudReconstructionNode()
    
    # 노드가 성공적으로 초기화되었을 때만 spin
    if node.extrinsic_matrix is not None:
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        # 초기화(TF 조회) 실패 시 자동 종료
        pass


if __name__ == '__main__':
    main()
