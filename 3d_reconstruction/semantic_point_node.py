#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import cv2
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, TransformException
from transforms3d.quaternions import quat2mat
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time
from collections import deque
import message_filters

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image as PILImage

# ============================================================================
# Segformer 관련 설정
# ============================================================================

# Custom 모델 설정
MODEL_PATH = "./models/dynamic_object/best_model2.pth"
INFERENCE_SIZE = 512

CLASS_TO_IDX = {
    'background': 0, 'barricade': 1, 'bench': 2, 'bicycle': 3, 'bollard': 4,
    'bus': 5, 'car': 6, 'carrier': 7, 'cat': 8, 'chair': 9, 'dog': 10,
    'fire_hydrant': 11, 'kiosk': 12, 'motorcycle': 13, 'movable_signage': 14,
    'parking_meter': 15, 'person': 16, 'pole': 17, 'potted_plant': 18,
    'power_controller': 19, 'scooter': 20, 'stop': 21, 'stroller': 22,
    'table': 23, 'traffic_light': 24, 'traffic_light_controller': 25,
    'traffic_sign': 26, 'tree_trunk': 27, 'truck': 28, 'wheelchair': 29
}

NUM_LABELS = len(CLASS_TO_IDX)
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

try:
    CUDA_AVAILABLE = torch.cuda.is_available()
except:
    CUDA_AVAILABLE = False


# ============================================================================
# Segformer 모델 클래스
# ============================================================================
class DirectSegFormer(nn.Module):
    """학습 시 사용된 모델과 동일한 구조의 클래스"""
    def __init__(self, pretrained_model_name="nvidia/mit-b0", num_classes=30):
        super().__init__()
        try:
            self.original_model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                use_safetensors=True,
            )
        except ValueError as e:
            if "torch.load" in str(e):
                print(f"Warning: {e}")
                print("Creating model architecture without pretrained weights...")
                from transformers import SegformerConfig
                config = SegformerConfig.from_pretrained(pretrained_model_name)
                config.num_labels = num_classes
                self.original_model = SegformerForSemanticSegmentation(config)
            else:
                raise e
        
    def forward(self, x):   
        outputs = self.original_model(pixel_values=x)
        return outputs.logits


# ============================================================================
# Semantic Point Cloud Node
# ============================================================================
class SemanticPointCloudNode(Node):
    """
    Depth + RGB + Segformer를 결합하여 Semantic Point Cloud를 생성하는 노드
    """
    
    def __init__(self):
        super().__init__('semantic_pointcloud_node')
        
        # OpenCV와 ROS 이미지 변환을 위한 브리지
        self.bridge = CvBridge()
        
        # QoS 프로파일 설정
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # GPU 설정
        self.device = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')
        self.get_logger().info(f'🖥️  Device: {self.device}')
        
        # Segformer 모델 로드
        self.model = self.load_segformer_model()
        self.get_logger().info(f'✅ Segformer model loaded from {MODEL_PATH}')
        
        # 이미지 전처리 (512x512로 리사이즈)
        self.transform = transforms.Compose([
            transforms.Resize((INFERENCE_SIZE, INFERENCE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 클래스별 색상 팔레트 생성
        self.color_palette = self.create_color_palette()
        
        # Depth와 RGB 이미지 동기화 구독
        depth_sub = message_filters.Subscriber(
            self, 
            Image, 
            '/camera/camera/depth/image_rect_raw',
            qos_profile=qos_profile
        )
        rgb_sub = message_filters.Subscriber(
            self, 
            Image, 
            '/camera/camera/color/image_raw',
            qos_profile=qos_profile
        )
        
        # TimeSynchronizer로 메시지 동기화 (100ms 허용)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [depth_sub, rgb_sub], 
            queue_size=10,
            slop=0.1  # 100ms
        )
        self.ts.registerCallback(self.synchronized_callback)
        
        # Point Cloud 발행자
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, 
            '/semantic_pointcloud', 
            qos_profile
        )
        
        # TF 변환을 위한 버퍼 및 리스너
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 좌표계 설정
        self.source_frame = 'camera_depth_optical_frame'
        self.target_frame = 'body'
        
        # 카메라 내부 파라미터 - Depth (Intel RealSense D455)
        self.depth_fx = 431.0625
        self.depth_fy = 431.0625
        self.depth_cx = 434.492
        self.depth_cy = 242.764
        
        # 카메라 내부 파라미터 - RGB (Intel RealSense D455)
        self.rgb_fx = 645.4923
        self.rgb_fy = 644.4183
        self.rgb_cx = 653.03259
        self.rgb_cy = 352.28909
        
        # 다운샘플링 비율
        self.downsample_y = 9
        self.downsample_x = 6
        
        # Point Cloud 필드 정의 (x, y, z, rgb, label)
        self.pointcloud_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='label', offset=16, datatype=PointField.UINT32, count=1),
        ]
        
        # Latency 측정
        self.timing_history = {
            'total': deque(maxlen=50),
            'segmentation': deque(maxlen=50),
            'depth_to_pc': deque(maxlen=50),
            'alignment': deque(maxlen=50),
            'tf_transform': deque(maxlen=50),
            'downsample': deque(maxlen=50),
            'msg_create': deque(maxlen=50),
        }
        self.frame_count = 0
        self.last_report_time = time.time()
        
        # GPU에 depth 그리드 미리 생성 (480x848 기준)
        if self.device.type == 'cuda':
            self._init_gpu_parameters()
        
        self.get_logger().info('🚀 Semantic Point Cloud Node initialized')

    def _init_gpu_parameters(self):
        """GPU에서 사용할 파라미터 미리 생성"""
        height, width = 480, 848
        
        v, u = torch.meshgrid(
            torch.arange(height, device=self.device, dtype=torch.float32),
            torch.arange(width, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        
        self.u_grid = u
        self.v_grid = v
        self.depth_fx_tensor = torch.tensor(self.depth_fx, device=self.device, dtype=torch.float32)
        self.depth_fy_tensor = torch.tensor(self.depth_fy, device=self.device, dtype=torch.float32)
        self.depth_cx_tensor = torch.tensor(self.depth_cx, device=self.device, dtype=torch.float32)
        self.depth_cy_tensor = torch.tensor(self.depth_cy, device=self.device, dtype=torch.float32)
        
        self.get_logger().info(f'GPU 파라미터 초기화 완료 ({height}x{width})')

    def load_segformer_model(self):
        """Segformer 모델 로드"""
        model = DirectSegFormer(num_classes=NUM_LABELS)
        
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
            
            # 키 이름 매핑
            new_state_dict = {}
            for key, value in checkpoint.items():
                if key.startswith('segformer.') or key.startswith('decode_head.'):
                    new_key = 'original_model.' + key
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            model.load_state_dict(new_state_dict, strict=False)
            
        except Exception as e:
            self.get_logger().error(f"Model loading failed: {e}")
            self.get_logger().warn("Using model without pretrained weights")
        
        model.to(self.device)
        model.eval()
        return model

    def create_color_palette(self):
        """클래스별 고유 색상 팔레트 생성 (BGR)"""
        cmap = plt.cm.get_cmap('tab20', NUM_LABELS)
        palette = np.zeros((NUM_LABELS, 3), dtype=np.uint8)
        
        for i in range(NUM_LABELS):
            if i == 0:  # 배경은 회색
                palette[i] = [100, 100, 100]
                continue
            rgba = cmap(i % 20)
            bgr = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
            palette[i] = bgr
        return palette

    def synchronized_callback(self, depth_msg, rgb_msg):
        """Depth와 RGB 이미지가 동기화되어 들어오는 콜백"""
        timings = {}
        t_start = time.perf_counter()
        
        try:
            # 1. 이미지 변환
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg, 
                desired_encoding=depth_msg.encoding
            ).astype(np.float32) / 1000.0
            
            rgb_image = self.bridge.imgmsg_to_cv2(
                rgb_msg, 
                desired_encoding='bgr8'
            )
            
            # 2. Segmentation 수행
            t1 = time.perf_counter()
            semantic_mask = self.run_segmentation(rgb_image)
            timings['segmentation'] = (time.perf_counter() - t1) * 1000
            
            # 3. Depth to Point Cloud
            t2 = time.perf_counter()
            if self.device.type == 'cuda':
                point_cloud = self.depth_to_pointcloud_gpu(depth_image)
            else:
                point_cloud = self.depth_to_pointcloud_cpu(depth_image)
            timings['depth_to_pc'] = (time.perf_counter() - t2) * 1000
            
            # 4. RGB-Depth 정렬 및 Semantic Label 매핑
            t3 = time.perf_counter()
            aligned_labels, aligned_colors = self.align_semantic_to_depth(
                semantic_mask, 
                rgb_image, 
                depth_image.shape
            )
            timings['alignment'] = (time.perf_counter() - t3) * 1000
            
            # 5. TF 변환
            t4 = time.perf_counter()
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    self.source_frame,
                    rclpy.time.Time()
                )
                transform_matrix = self.transform_to_matrix(transform)
                
                if self.device.type == 'cuda':
                    transformed_cloud = self.apply_transform_gpu(point_cloud, transform_matrix)
                else:
                    transformed_cloud = self.apply_transform_cpu(point_cloud, transform_matrix)
            except TransformException as e:
                self.get_logger().warn(f'TF 변환 실패: {e}', throttle_duration_sec=1.0)
                return
            timings['tf_transform'] = (time.perf_counter() - t4) * 1000
            
            # 6. 다운샘플링
            t5 = time.perf_counter()
            if self.device.type == 'cuda':
                points, colors, labels = self.process_pointcloud_gpu(
                    transformed_cloud, 
                    aligned_labels, 
                    aligned_colors
                )
            else:
                points, colors, labels = self.process_pointcloud_cpu(
                    transformed_cloud, 
                    aligned_labels, 
                    aligned_colors
                )
            timings['downsample'] = (time.perf_counter() - t5) * 1000
            
            # 7. 메시지 생성 및 발행
            t6 = time.perf_counter()
            pointcloud_msg = self.create_semantic_pointcloud_msg(
                points, 
                colors, 
                labels, 
                self.target_frame
            )
            self.pointcloud_pub.publish(pointcloud_msg)
            timings['msg_create'] = (time.perf_counter() - t6) * 1000
            
            # 전체 시간
            timings['total'] = (time.perf_counter() - t_start) * 1000
            
            # 타이밍 기록
            self.record_timings(timings)
            
        except Exception as e:
            self.get_logger().error(f'처리 오류: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def run_segmentation(self, rgb_image):
        """Segformer를 이용한 Semantic Segmentation"""
        # BGR -> RGB
        rgb_image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        

        pil_image = PILImage.fromarray(rgb_image_rgb)
        # 512x512로 리사이즈 및 전처리
        input_tensor = self.transform(pil_image)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        # 추론
        with torch.no_grad():
            logits = self.model(input_tensor)
        
        # 원본 크기로 업샘플링
        original_h, original_w = rgb_image.shape[:2]
        upsampled_logits = F.interpolate(
            logits,
            size=(original_h, original_w),
            mode='bilinear',
            align_corners=False
        )
        
        # Argmax로 클래스 예측
        pred_mask = torch.argmax(upsampled_logits, dim=1).squeeze()
        pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)
        
        return pred_mask_np

    def align_semantic_to_depth(self, semantic_mask, rgb_image, depth_shape):
        """
        RGB 좌표계의 Semantic Mask를 Depth 좌표계로 정렬
        
        Args:
            semantic_mask: (H_rgb, W_rgb) - RGB 이미지의 semantic label
            rgb_image: (H_rgb, W_rgb, 3) - RGB 이미지
            depth_shape: (H_depth, W_depth) - Depth 이미지 크기
            
        Returns:
            aligned_labels: (H_depth, W_depth) - Depth 좌표계로 정렬된 label
            aligned_colors: (H_depth, W_depth, 3) - Depth 좌표계로 정렬된 색상
        """
        depth_h, depth_w = depth_shape
        rgb_h, rgb_w = semantic_mask.shape
        
        # Depth 픽셀 좌표 생성
        u_depth, v_depth = np.meshgrid(np.arange(depth_w), np.arange(depth_h))
        
        # Depth 좌표 -> RGB 좌표 변환 (간단한 스케일링 + 오프셋)
        # 실제로는 extrinsic calibration이 필요하지만, 
        # RealSense의 경우 aligned 이미지를 사용하면 거의 일치함
        scale_x = rgb_w / depth_w
        scale_y = rgb_h / depth_h
        
        u_rgb = (u_depth * scale_x).astype(np.int32)
        v_rgb = (v_depth * scale_y).astype(np.int32)
        
        # 범위 클리핑
        u_rgb = np.clip(u_rgb, 0, rgb_w - 1)
        v_rgb = np.clip(v_rgb, 0, rgb_h - 1)
        
        # RGB 좌표에서 label과 color 추출
        aligned_labels = semantic_mask[v_rgb, u_rgb]
        aligned_colors = rgb_image[v_rgb, u_rgb]
        
        return aligned_labels, aligned_colors

    def depth_to_pointcloud_gpu(self, depth_map):
        """GPU를 이용한 Depth to Point Cloud 변환"""
        depth_tensor = torch.from_numpy(depth_map).to(self.device)
        
        z = depth_tensor
        x = (self.u_grid - self.depth_cx_tensor) * z / self.depth_fx_tensor
        y = (self.v_grid - self.depth_cy_tensor) * z / self.depth_fy_tensor
        
        pointcloud = torch.stack([x, y, z], dim=-1)
        return pointcloud

    def depth_to_pointcloud_cpu(self, depth_map):
        """CPU NumPy를 이용한 Depth to Point Cloud 변환"""
        height, width = depth_map.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        z = depth_map
        x = (u - self.depth_cx) * z / self.depth_fx
        y = (v - self.depth_cy) * z / self.depth_fy
        
        return np.stack((x, y, z), axis=-1)

    def apply_transform_gpu(self, points, matrix):
        """GPU를 이용한 좌표 변환"""
        original_shape = points.shape
        points_flat = points.reshape(-1, 3)
        
        matrix_tensor = torch.from_numpy(matrix).to(self.device, dtype=torch.float32)
        
        ones = torch.ones((points_flat.shape[0], 1), device=self.device, dtype=torch.float32)
        homogeneous = torch.cat([points_flat, ones], dim=1)
        
        transformed = torch.mm(homogeneous, matrix_tensor.T)
        
        return transformed[:, :3].reshape(original_shape)

    def apply_transform_cpu(self, points, matrix):
        """CPU NumPy를 이용한 좌표 변환"""
        original_shape = points.shape
        points_flat = points.reshape(-1, 3)
        
        ones = np.ones((points_flat.shape[0], 1))
        homogeneous_points = np.hstack((points_flat, ones))
        
        transformed = homogeneous_points @ matrix.T
        
        return transformed[:, :3].reshape(original_shape)

    def process_pointcloud_gpu(self, pointcloud, labels, colors):
        """GPU를 이용한 다운샘플링"""
        # 다운샘플링
        sampled_pc = pointcloud[::self.downsample_y, ::self.downsample_x, :]
        sampled_labels = labels[::self.downsample_y, ::self.downsample_x]
        sampled_colors = colors[::self.downsample_y, ::self.downsample_x]
        
        # Flatten
        points = sampled_pc.reshape(-1, 3)
        points_np = points.cpu().numpy() if isinstance(points, torch.Tensor) else points
        labels_flat = sampled_labels.reshape(-1)
        colors_flat = sampled_colors.reshape(-1, 3)
        
        return points_np, colors_flat, labels_flat

    def process_pointcloud_cpu(self, pointcloud, labels, colors):
        """CPU를 이용한 다운샘플링"""
        sampled_pc = pointcloud[::self.downsample_y, ::self.downsample_x, :]
        sampled_labels = labels[::self.downsample_y, ::self.downsample_x]
        sampled_colors = colors[::self.downsample_y, ::self.downsample_x]
        
        points = sampled_pc.reshape(-1, 3)
        labels_flat = sampled_labels.reshape(-1)
        colors_flat = sampled_colors.reshape(-1, 3)
        
        return points, colors_flat, labels_flat

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

    def create_semantic_pointcloud_msg(self, points, colors, labels, frame_id):
        """Semantic Point Cloud 메시지 생성 (x, y, z, rgb, label)"""
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        
        # RGB 패킹
        rgb_uint32 = (
            (colors[:, 2].astype(np.uint32) << 16) |  # R
            (colors[:, 1].astype(np.uint32) << 8) |   # G
            (colors[:, 0].astype(np.uint32))          # B (BGR input)
        )
        rgb_float32 = rgb_uint32.view(np.float32)
        
        # Label을 uint32로
        labels_uint32 = labels.astype(np.uint32)
        
        # XYZ + RGB + Label 결합
        pointcloud_data = np.hstack([
            points.astype(np.float32), 
            rgb_float32.reshape(-1, 1),
            labels_uint32.reshape(-1, 1).view(np.float32)  # uint32를 float32로 reinterpret
        ])
        
        return PointCloud2(
            header=header,
            height=1,
            width=pointcloud_data.shape[0],
            fields=self.pointcloud_fields,
            is_bigendian=False,
            point_step=20,  # 5 * 4 bytes
            row_step=20 * pointcloud_data.shape[0],
            data=pointcloud_data.tobytes(),
            is_dense=True,
        )

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
            f"\n{'='*70}",
            f"📊 Semantic Point Cloud Performance (최근 {len(self.timing_history['total'])} frames)",
            f"{'='*70}",
            f"Backend: {'GPU (CUDA)' if self.device.type == 'cuda' else 'CPU'}"
        ]
        
        for key in ['total', 'segmentation', 'depth_to_pc', 'alignment', 
                    'tf_transform', 'downsample', 'msg_create']:
            if self.timing_history[key]:
                times = list(self.timing_history[key])
                avg = np.mean(times)
                std = np.std(times)
                min_t = np.min(times)
                max_t = np.max(times)
                
                label_map = {
                    'total': '🔴 전체',
                    'segmentation': '  ├─ Segmentation (Segformer)',
                    'depth_to_pc': '  ├─ Depth→PC 변환',
                    'alignment': '  ├─ RGB-Depth 정렬',
                    'tf_transform': '  ├─ TF 좌표 변환',
                    'downsample': '  ├─ 다운샘플링',
                    'msg_create': '  └─ 메시지 생성',
                }
                
                stats_msg.append(
                    f"{label_map.get(key, key):35} "
                    f"avg: {avg:6.2f}ms  "
                    f"std: {std:5.2f}ms  "
                    f"[{min_t:5.2f} ~ {max_t:6.2f}]ms"
                )
        
        fps = len(self.timing_history['total']) / 2.0
        stats_msg.append(f"{'='*70}")
        stats_msg.append(f"FPS: {fps:.1f} Hz")
        stats_msg.append(f"{'='*70}\n")
        
        self.get_logger().info('\n'.join(stats_msg))


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    node = SemanticPointCloudNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
