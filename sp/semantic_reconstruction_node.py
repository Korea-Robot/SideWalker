#!/usr/bin/env python3
"""
Semantic Point Cloud Reconstruction Node
Generates semantic point clouds from RGB-D images

파일 구조
1. reconstruction_config.py - 설정 파일

모든 파라미터를 한 곳에서 관리
카메라 intrinsic 파라미터 (depth, RGB)
Semantic 모델 on/off 및 타입 선택
다운샘플링, 토픽, 프레임 ID 등

2. semantic_model.py - Semantic 모델 모듈

3가지 모델 지원 (Custom, SegFormer-ADE20k, MaskFormer-COCO)
통합된 인터페이스로 간단하게 사용
predict() 메서드로 semantic mask 반환

3. reconstruction_node.py - ROS2 메인 노드

RGB-D 동기화 및 처리
Semantic point cloud 생성
GPU/CPU 최적화
TF 좌표 변환

🎯 핵심 기능
✅ Semantic 모드: RGB intrinsic으로 semantic segmentation → depth 좌표로 정렬 → semantic point cloud 생성
✅ RGB-only 모드: config.use_semantic = False로 설정하면 semantic 없이 RGB 값만 사용
✅ 최적화: GPU 캐싱, 효율적인 좌표 변환, 불필요한 코드 제거

"""

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
import message_filters
import torch
import time
from collections import deque

# from semantic_reconstruction_config import ReconstructionConfig
# from optimized_config import ReconstructionConfig
from last_config import ReconstructionConfig

# from semantic_reconstruction_model  import SemanticModel
from optimized_model import SemanticModel

class ReconstructionNode(Node):
    """
    ROS2 Node for Semantic Point Cloud Reconstruction
    Combines RGB, Depth, and optional Semantic Segmentation
    """
    
    def __init__(self, config: ReconstructionConfig = None):
        super().__init__('reconstruction_node')
        
        # Configuration
        self.config = config or ReconstructionConfig()
        
        # Core components
        self.bridge = CvBridge()
        self.device = self._setup_device()
        self.semantic_model = SemanticModel(self.config, self.device, self.get_logger())
        
        # Setup ROS2 interfaces
        self._setup_subscribers()
        self._setup_publishers()
        self._setup_tf()
        
        # GPU optimization
        if self.device.type == 'cuda':
            self._init_gpu_cache()
        
        # Performance monitoring
        self.timings = {
            'total': deque(maxlen=50),
            'semantic': deque(maxlen=50),
            'depth_to_pc': deque(maxlen=50),
            'alignment': deque(maxlen=50),
            'transform': deque(maxlen=50),
        }
        self.frame_count = 0
        self.last_report_time = time.time()
        
        mode = "Semantic RGB" if self.config.use_semantic else "RGB-only"
        self.get_logger().info(f'🚀 Reconstruction Node Started ({mode} mode)')
    
    def _setup_device(self):
        """Setup computation device (GPU/CPU)"""
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
            self.get_logger().info('🖥️  Using GPU (CUDA)')
        else:
            device = torch.device('cpu')
            self.get_logger().info('🖥️  Using CPU')
        return device
    
    def _setup_subscribers(self):
        """Setup synchronized RGB-D subscribers"""
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        depth_sub = message_filters.Subscriber(
            self, Image, self.config.depth_topic, qos_profile=qos
        )
        rgb_sub = message_filters.Subscriber(
            self, Image, self.config.rgb_topic, qos_profile=qos
        )
        
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [depth_sub, rgb_sub],
            queue_size=10,
            slop=self.config.sync_slop
        )
        self.sync.registerCallback(self.rgbd_callback)
    
    def _setup_publishers(self):
        """Setup point cloud publisher"""
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.pc_pub = self.create_publisher(
            PointCloud2,
            self.config.pointcloud_topic,
            qos
        )
    
    def _setup_tf(self):
        """Setup TF2 transform listener"""
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
    
    def _init_gpu_cache(self):
        """Pre-allocate GPU tensors for depth unprojection"""
        h, w = 480, 848  # D455 depth resolution
        
        v, u = torch.meshgrid(
            torch.arange(h, device=self.device, dtype=torch.float32),
            torch.arange(w, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        
        self.u_grid = u
        self.v_grid = v
        self.fx = torch.tensor(self.config.depth_intrinsics.fx, device=self.device)
        self.fy = torch.tensor(self.config.depth_intrinsics.fy, device=self.device)
        self.cx = torch.tensor(self.config.depth_intrinsics.cx, device=self.device)
        self.cy = torch.tensor(self.config.depth_intrinsics.cy, device=self.device)
        
        self.get_logger().info(f'GPU cache initialized ({h}x{w})')
    
    def rgbd_callback(self, depth_msg, rgb_msg):
        """Main callback for synchronized RGB-D processing"""
        t_start = time.perf_counter()
        
        try:
            # 1. Convert ROS messages
            depth = self.bridge.imgmsg_to_cv2(
                depth_msg, 
                desired_encoding=depth_msg.encoding
            ).astype(np.float32) / 1000.0  # mm -> m
            
            rgb = self.bridge.imgmsg_to_cv2(
                rgb_msg,
                desired_encoding='bgr8'
            )
            
            # 2. Semantic segmentation (optional)
            t1 = time.perf_counter()
            semantic_mask = None
            if self.config.use_semantic:
                semantic_mask = self.semantic_model.predict(rgb)
            self.timings['semantic'].append((time.perf_counter() - t1) * 1000)
            
            # 3. Depth to point cloud
            t2 = time.perf_counter()
            points = self._depth_to_pointcloud(depth)
            self.timings['depth_to_pc'].append((time.perf_counter() - t2) * 1000)
            
            # 4. Align RGB/Semantic to depth
            t3 = time.perf_counter()
            colors, labels = self._align_to_depth(rgb, semantic_mask, depth.shape)
            self.timings['alignment'].append((time.perf_counter() - t3) * 1000)
            
            # 5. Transform to target frame
            t4 = time.perf_counter()
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.config.target_frame,
                    self.config.source_frame,
                    rclpy.time.Time()
                )
                points = self._apply_transform(points, transform)
            except TransformException as e:
                self.get_logger().warn(f'TF failed: {e}', throttle_duration_sec=1.0)
                return
            self.timings['transform'].append((time.perf_counter() - t4) * 1000)
            
            # 6. Downsample and publish
            points, colors, labels = self._downsample(points, colors, labels)
            pc_msg = self._create_pointcloud_msg(points, colors, labels)
            self.pc_pub.publish(pc_msg)
            
            # 7. Record timing
            self.timings['total'].append((time.perf_counter() - t_start) * 1000)
            self.frame_count += 1
            
            # Periodic stats
            if time.time() - self.last_report_time >= 2.0:
                self._print_stats()
                self.last_report_time = time.time()
                
        except Exception as e:
            self.get_logger().error(f'Processing error: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def _depth_to_pointcloud(self, depth):
        """Convert depth image to 3D point cloud"""
        if self.device.type == 'cuda':
            return self._depth_to_pc_gpu(depth)
        else:
            return self._depth_to_pc_cpu(depth)
    
    def _depth_to_pc_gpu(self, depth):
        """GPU-accelerated depth unprojection"""
        z = torch.from_numpy(depth).to(self.device)
        x = (self.u_grid - self.cx) * z / self.fx
        y = (self.v_grid - self.cy) * z / self.fy
        return torch.stack([x, y, z], dim=-1)
    
    def _depth_to_pc_cpu(self, depth):
        """CPU depth unprojection"""
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        fx, fy = self.config.depth_intrinsics.fx, self.config.depth_intrinsics.fy
        cx, cy = self.config.depth_intrinsics.cx, self.config.depth_intrinsics.cy
        
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        return np.stack((x, y, z), axis=-1)
    
    def _align_to_depth(self, rgb, semantic_mask, depth_shape):
        """
        Align RGB and semantic labels to depth frame
        
        Returns:
            colors: (H_d, W_d, 3) BGR colors
            labels: (H_d, W_d) semantic labels (or None)
        """
        h_d, w_d = depth_shape
        h_r, w_r = rgb.shape[:2]
        
        # Simple scaling-based alignment
        u_d, v_d = np.meshgrid(np.arange(w_d), np.arange(h_d))
        u_r = (u_d * w_r / w_d).astype(np.int32)
        v_r = (v_d * h_r / h_d).astype(np.int32)
        u_r = np.clip(u_r, 0, w_r - 1)
        v_r = np.clip(v_r, 0, h_r - 1)
        
        colors = rgb[v_r, u_r]
        
        if semantic_mask is not None:
            labels = semantic_mask[v_r, u_r]
        else:
            labels = None
        
        return colors, labels
    
    def _apply_transform(self, points, transform):
        """Apply TF transform to point cloud"""
        # Extract translation and rotation
        t = transform.transform.translation
        translation = np.array([t.x, t.y, t.z])
        
        r = transform.transform.rotation
        quat = [r.w, r.x, r.y, r.z]
        rotation = quat2mat(quat)
        
        # Build 4x4 matrix
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        
        # Apply transform
        if isinstance(points, torch.Tensor):
            return self._transform_gpu(points, matrix)
        else:
            return self._transform_cpu(points, matrix)
    
    def _transform_gpu(self, points, matrix):
        """GPU transform"""
        shape = points.shape
        pts = points.reshape(-1, 3)
        
        mat = torch.from_numpy(matrix).to(self.device, dtype=torch.float32)
        ones = torch.ones((pts.shape[0], 1), device=self.device, dtype=torch.float32)
        homogeneous = torch.cat([pts, ones], dim=1)
        
        transformed = torch.mm(homogeneous, mat.T)
        return transformed[:, :3].reshape(shape)
    
    def _transform_cpu(self, points, matrix):
        """CPU transform"""
        shape = points.shape
        pts = points.reshape(-1, 3)
        
        ones = np.ones((pts.shape[0], 1))
        homogeneous = np.hstack((pts, ones))
        
        transformed = homogeneous @ matrix.T
        return transformed[:, :3].reshape(shape)
    
    def _downsample(self, points, colors, labels):
        """Downsample point cloud"""
        dy, dx = self.config.downsample_y, self.config.downsample_x
        
        pts = points[::dy, ::dx].reshape(-1, 3)
        cols = colors[::dy, ::dx].reshape(-1, 3)
        
        if labels is not None:
            labs = labels[::dy, ::dx].reshape(-1)
        else:
            labs = None
        
        # Convert to numpy if needed
        if isinstance(pts, torch.Tensor):
            pts = pts.cpu().numpy()
        
        return pts, cols, labs
    
    def _create_pointcloud_msg(self, points, colors, labels):
        """Create PointCloud2 message"""
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.config.target_frame
        
        # Pack RGB as uint32
        rgb_uint32 = (
            (colors[:, 2].astype(np.uint32) << 16) |  # R
            (colors[:, 1].astype(np.uint32) << 8) |   # G
            (colors[:, 0].astype(np.uint32))          # B
        )
        rgb_float32 = rgb_uint32.view(np.float32)
        
        # Build point cloud data
        if labels is not None:
            # x, y, z, rgb, label
            labels_uint32 = labels.astype(np.uint32)
            data = np.hstack([
                points.astype(np.float32),
                rgb_float32.reshape(-1, 1),
                labels_uint32.reshape(-1, 1).view(np.float32)
            ])
            
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name='label', offset=16, datatype=PointField.UINT32, count=1),
            ]
            point_step = 20
        else:
            # x, y, z, rgb only
            data = np.hstack([
                points.astype(np.float32),
                rgb_float32.reshape(-1, 1)
            ])
            
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            point_step = 16
        
        return PointCloud2(
            header=header,
            height=1,
            width=data.shape[0],
            fields=fields,
            is_bigendian=False,
            point_step=point_step,
            row_step=point_step * data.shape[0],
            data=data.tobytes(),
            is_dense=True,
        )
    
    def _print_stats(self):
        """Print performance statistics"""
        if not self.timings['total']:
            return
        
        mode = "Semantic" if self.config.use_semantic else "RGB-only"
        backend = "GPU" if self.device.type == 'cuda' else "CPU"
        
        msg = [
            f"\n{'='*60}",
            f"📊 Point Cloud Reconstruction Stats",
            f"{'='*60}",
            f"Mode: {mode} | Backend: {backend}"
        ]
        
        labels = {
            'total': '🔴 Total',
            'semantic': '  ├─ Semantic',
            'depth_to_pc': '  ├─ Depth→PC',
            'alignment': '  ├─ Alignment',
            'transform': '  └─ Transform',
        }
        
        for key, label in labels.items():
            if self.timings[key]:
                times = list(self.timings[key])
                avg = np.mean(times)
                std = np.std(times)
                min_t = np.min(times)
                max_t = np.max(times)
                
                msg.append(
                    f"{label:25} avg: {avg:6.2f}ms  "
                    f"std: {std:5.2f}ms  [{min_t:5.2f} ~ {max_t:6.2f}]ms"
                )
        
        fps = len(self.timings['total']) / 2.0
        msg.extend([
            f"{'='*60}",
            f"FPS: {fps:.1f} Hz | Frames: {self.frame_count}",
            f"{'='*60}\n"
        ])
        
        self.get_logger().info('\n'.join(msg))


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    # Create configuration (can be customized here)
    config = ReconstructionConfig()
    
    # Example: Disable semantic segmentation for RGB-only mode
    # config.use_semantic = False
    
    # Example: Change model type
    # config.model_type = "segformer-ade20k"
    
    # Example: Adjust downsampling
    # config.downsample_y = 6
    # config.downsample_x = 4
    
    node = ReconstructionNode(config)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()