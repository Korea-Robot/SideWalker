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

class PointCloudReconstructionNode(Node):
    """
    Depth 이미지를 3D Point Cloud로 변환하고 TF를 적용하여 로봇 좌표계로 변환하는 노드
    (PyTorch CUDA GPU 가속 전용 버전)
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
        self.source_frame = 'camera_depth_optical_frame' # 원본 좌표계 (카메라)
        self.target_frame = 'body' # 대상 좌표계 (로봇 베이스)
        
        # 카메라 내부 파라미터 (Intel RealSense 기준)
        self.fx = 431.0625 # X축 초점 거리
        self.fy = 431.0625 # Y축 초점 거리
        self.cx = 434.492  # 주점 X좌표
        self.cy = 242.764  # 주점 Y좌표
        
        # 다운샘플링 비율 (Y축으로 9픽셀마다, X축으로 6픽셀마다)
        self.downsample_y = 9
        self.downsample_x = 6
        
        # Point Cloud 필드 정의 (포인트당 16바이트)
        # x, y, z 각각 4바이트 (FLOAT32)
        # rgb 4바이트 (FLOAT32로 패킹된 UINT32)
        self.pointcloud_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        # GPU 설정 (CUDA 사용 고정)
        self.device = torch.device('cuda')
        self.get_logger().info('🚀 CUDA GPU 가속 활성화 (PyTorch 사용)')
        
        # GPU에서 사용할 파라미터 미리 계산하여 로드
        self._init_gpu_parameters()
        
        self.get_logger().info('Point Cloud Reconstruction Node initialized (GPU Only)')

    def _init_gpu_parameters(self):
        """GPU에서 사용할 파라미터 미리 생성"""
        
        # 이미지 크기 (Intel RealSense D435의 848x480 해상도 기준)
        height, width = 480, 848
        
        # 픽셀 좌표(v, u) 그리드를 미리 생성
        # indexing='ij'는 (height, width) 순서(행 우선)로 그리드를 생성
        v, u = torch.meshgrid(
            torch.arange(height, device=self.device, dtype=torch.float32),
            torch.arange(width, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        
        # 3D 계산에 필요한 상수들을 미리 GPU 텐서로 만들어 둠
        # z * (u - cx) / fx = x
        # z * (v - cy) / fy = y
        self.u_grid = u
        self.v_grid = v
        self.fx_tensor = torch.tensor(self.fx, device=self.device, dtype=torch.float32)
        self.fy_tensor = torch.tensor(self.fy, device=self.device, dtype=torch.float32)
        self.cx_tensor = torch.tensor(self.cx, device=self.device, dtype=torch.float32)
        self.cy_tensor = torch.tensor(self.cy, device=self.device, dtype=torch.float32)
        
        self.get_logger().info(f'GPU 파라미터 초기화 완료 ({height}x{width})')

    def depth_callback(self, msg):
        """Depth 이미지를 수신하여 Point Cloud로 변환하고 발행"""
        
        try:
            # 1. Depth 이미지 변환 (ROS Image -> NumPy)
            # D435 카메라는 보통 '16UC1' (16-bit unsigned char)로, 픽셀값이 mm 단위
            # 이를 float32로 변환하고 1000.0으로 나누어 미터(m) 단위로 변경
            depth_image = self.bridge.imgmsg_to_cv2(
                msg, 
                desired_encoding=msg.encoding
            ).astype(np.float32) / 1000.0
            
            # 2. Depth to Point Cloud (GPU)
            # NumPy 배열(CPU)을 GPU 텐서로 변환하고 3D 좌표 계산
            point_cloud = self.depth_to_pointcloud_gpu(depth_image)
            
            # 3. TF 조회 (CPU)
            # 'camera_depth_optical_frame'에서 'body'로의 변환(Transform)을 조회
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time() # 가장 최신의 TF 사용
            )
            # 조회한 TF를 4x4 동차 변환 행렬(NumPy)로 변환
            transform_matrix = self.transform_to_matrix(transform)
            
            # 4. 변환 적용 (GPU)
            # 카메라 좌표계의 3D 포인트들을 로봇('body') 좌표계로 변환
            transformed_cloud = self.apply_transform_gpu(point_cloud, transform_matrix)
            
            # 5. 다운샘플링 및 색상 적용 (GPU -> CPU)
            # 처리 속도 향상과 데이터 크기 감소를 위해 다운샘플링
            # 최종 NumPy 배열(CPU)과 색상 정보(CPU)를 얻음
            points, colors = self.process_pointcloud_gpu(transformed_cloud)
            
            # 6. PointCloud2 메시지 생성 (CPU)
            # NumPy 배열을 ROS PointCloud2 메시지 형식으로 패킹
            pointcloud_msg = self.create_pointcloud_msg(
                points, 
                colors, 
                self.target_frame # 헤더의 frame_id는 변환된 'body'
            )
            
            # 7. 발행
            self.pointcloud_pub.publish(pointcloud_msg)
            
        except TransformException as e:
            # TF 조회를 실패한 경우 (ex: TF 트리가 아직 연결되지 않음)
            self.get_logger().warn(f'TF 변환 실패: {e}', throttle_duration_sec=1.0)
        except Exception as e:
            # 기타 예외 처리
            self.get_logger().error(f'Point Cloud 처리 오류: {e}')

    def depth_to_pointcloud_gpu(self, depth_map):
        """GPU를 이용한 Depth to Point Cloud 변환"""
        
        # 1. NumPy(CPU) 배열을 PyTorch GPU 텐서로 복사
        depth_tensor = torch.from_numpy(depth_map).to(self.device)
        
        # 2. 3D 좌표 계산 (Pinhole Camera Model)
        # z값은 depth 텐서 값 그대로 사용
        z = depth_tensor
        
        # x = (u - cx) * z / fx
        # y = (v - cy) * z / fy
        # 모든 계산은 GPU에서 병렬로 수행됨
        x = (self.u_grid - self.cx_tensor) * z / self.fx_tensor
        y = (self.v_grid - self.cy_tensor) * z / self.fy_tensor
        
        # 3. (x, y, z) 텐서를 마지막 차원을 기준으로 스택
        # 결과: (Height, Width, 3) 형태의 3D 포인트 클라우드 텐서
        pointcloud = torch.stack([x, y, z], dim=-1)
        
        return pointcloud

    def apply_transform_gpu(self, points, matrix):
        """GPU를 이용한 좌표 변환"""
        
        # 원본 형태 (H, W, 3) 저장
        original_shape = points.shape
        # (H, W, 3) -> (N, 3)으로 변환 (N = H * W)
        points_flat = points.reshape(-1, 3)
        
        # 1. 변환 행렬(NumPy)을 GPU 텐서로 복사
        matrix_tensor = torch.from_numpy(matrix).to(self.device, dtype=torch.float32)
        
        # 2. 동차 좌표(Homogeneous Coordinates) 생성
        # (N, 3) -> (N, 4)로 만들기 위해 마지막에 1 추가
        ones = torch.ones((points_flat.shape[0], 1), device=self.device, dtype=torch.float32)
        homogeneous = torch.cat([points_flat, ones], dim=1)
        
        # 3. 행렬 곱셈으로 변환 적용
        # (N, 4) @ (4, 4)^T = (N, 4)
        # PyTorch의 torch.mm은 (A @ B.T)가 A @ B 보다 효율적일 수 있음
        # 여기서는 matrix_tensor.T (전치행렬)와 곱함
        transformed = torch.mm(homogeneous, matrix_tensor.T)
        
        # 4. (N, 4) -> (N, 3)으로 변환 (동차 좌표의 w 성분 제거)
        # 5. (N, 3) -> (H, W, 3) 원본 형태로 복원
        return transformed[:, :3].reshape(original_shape)

    def process_pointcloud_gpu(self, pointcloud):
        """GPU를 이용한 다운샘플링 및 색상 적용"""
        
        # 1. 다운샘플링 (GPU)
        # (H, W, 3) 텐서를 슬라이싱하여 다운샘플링
        # [::y, ::x]는 y 스텝, x 스텝마다 픽셀을 선택
        sampled = pointcloud[::self.downsample_y, ::self.downsample_x, :]
        
        # 2. Flatten (GPU)
        # (H_sampled, W_sampled, 3) -> (N_sampled, 3)
        points = sampled.reshape(-1, 3)
        
        # 3. GPU -> CPU로 데이터 이동
        # .cpu()는 데이터를 CPU 메모리로 복사
        # .numpy()는 CPU 텐서를 NumPy 배열로 변환
        # 이 시점에서 points_np는 CPU 메모리에 있는 NumPy 배열
        points_np = points.cpu().numpy()
        
        # 4. 색상 생성 (CPU)
        # 다운샘플링된 포인트 개수만큼 색상 배열 생성
        # 색상은 (R=200, G=100, B=208) (핑크/보라 계열)로 고정
        num_points = points_np.shape[0]
        colors = np.zeros((num_points, 3), dtype=np.uint8)
        colors[:, 0] = 200 # R
        colors[:, 1] = 100 # G
        colors[:, 2] = 208 # B
        
        return points_np, colors

    def transform_to_matrix(self, transform):
        """ROS Transform 메시지를 4x4 동차 변환 행렬(NumPy)로 변환"""
        
        # 1. Translation (이동) 벡터 추출
        t = transform.transform.translation
        translation = np.array([t.x, t.y, t.z])
        
        # 2. Rotation (회전) 쿼터니언 추출
        r = transform.transform.rotation
        # transforms3d 라이브러리는 (w, x, y, z) 순서를 사용
        quat = [r.w, r.x, r.y, r.z]
        
        # 3. 쿼터니언을 3x3 회전 행렬로 변환
        rotation_matrix = quat2mat(quat)
        
        # 4. 4x4 동차 변환 행렬 생성
        matrix = np.eye(4) # 4x4 단위 행렬로 시작
        matrix[:3, :3] = rotation_matrix # 좌상단 3x3에 회전 행렬 적용
        matrix[:3, 3] = translation   # 우측 3x1에 이동 벡터 적용
        
        return matrix

    def create_pointcloud_msg(self, points, colors, frame_id):
        """NumPy 배열을 ROS PointCloud2 메시지로 변환"""
        
        # 1. 헤더 생성
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        
        # 2. RGB 색상 패킹
        # PointCloud2의 'rgb' 필드는 4바이트 float32로 정의됨
        # (R, G, B) 3바이트를 4바이트 공간에 넣기 위해 비트 쉬프트 사용
        # (R << 16) | (G << 8) | (B)
        rgb_uint32 = (
            (colors[:, 0].astype(np.uint32) << 16) |
            (colors[:, 1].astype(np.uint32) << 8) |
            (colors[:, 2].astype(np.uint32))
        )
        
        # 3. uint32를 float32로 비트 레벨에서 재해석 (casting 아님)
        # NumPy의 .view()를 사용하여 메모리 해석 방식을 변경
        rgb_float32 = rgb_uint32.view(np.float32)
        
        # 4. XYZ (float32)와 RGB (float32로 패킹됨) 결합
        # (N, 3) 형태의 points와 (N, 1) 형태의 rgb_float32를 결합
        pointcloud_data = np.hstack([
            points.astype(np.float32), 
            rgb_float32.reshape(-1, 1)
        ])
        
        # 5. PointCloud2 메시지 생성
        return PointCloud2(
            header=header,
            height=1, # 1D (unorganized) 포인트 클라우드
            width=pointcloud_data.shape[0], # 포인트 개수
            fields=self.pointcloud_fields, # 필드 정의
            is_bigendian=False,
            point_step=16, # 1개 포인트가 차지하는 바이트 (x,y,z,rgb = 4*4 = 16)
            row_step=16 * pointcloud_data.shape[0], # 1줄(row)이 차지하는 바이트
            data=pointcloud_data.tobytes(), # 전체 데이터를 바이트 배열로 변환
            is_dense=True, # 유효하지 않은 (NaN, Inf) 포인트 없음
        )


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    node = PointCloudReconstructionNode()
    
    try:
        # 노드가 종료될 때까지 (Ctrl+C 등) 계속 실행
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 노드 종료
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
