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
import message_filters
import torch
import torch.nn.functional as F
import time
from collections import deque
import argparse
from typing import Union, List, Tuple
from pycocotools import mask as maskUtils

# NPU 관련 라이브러리
import maccel
from mblt_infer_original.helper import YoloHelper

# Configuration import
from config import Config, COCO_CLASS_TO_IDX


class NPUSemanticPointCloudNode(Node):
    """
    NPU 기반 YOLOv9 Segmentation => ros2 3D pointcloud
    """

    def __init__(self):
        super().__init__('npu_semantic_pointcloud_node')

        # --- Configuration 로딩 ---
        self.config = Config()

        # --- Argument Parsing ---
        parser = argparse.ArgumentParser()
        parser.add_argument("--base_path", type=str, default=".")
        parser.add_argument("--conf_thres", type=float, default=self.config.model.conf_thres)
        parser.add_argument("--iou_thres", type=float, default=self.config.model.iou_thres)
        args, _ = parser.parse_known_args()

        # Update config with command line arguments
        self.config.model.conf_thres = args.conf_thres
        self.config.model.iou_thres = args.iou_thres
        self.config.model.model_path = args.base_path + "/yolov9c-seg.mxq"

        # --- 기본 모듈 초기화 ---
        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.get_logger().info(f'🚀 Device: {self.device}')

        # --- NPU 모델 로딩 ---
        self.get_logger().info('NPU Model loading...')
        acc1 = maccel.Accelerator()
        mc1 = maccel.ModelConfig()
        mc1.set_global8_core_mode()
        self.mxq_model = maccel.Model(self.config.model.model_path, mc1)
        self.mxq_model.launch(acc1)

        # --- YoloHelper 초기화 ---
        self.helper = self.yolov9c_seg_helper(
            conf_thres=self.config.model.conf_thres, 
            iou_thres=self.config.model.iou_thres, 
            device="aries"
        )
        self.get_logger().info('✅ NPU Model loaded')

        # --- ROS 파라미터 설정 ---
        self._setup_ros_parameters()

        # --- ROS 통신 설정 ---
        self._setup_ros_communication()

        # --- GPU 파라미터 초기화 ---
        self._init_gpu_parameters()
        self._init_semantic_colormap()

        # 성능 모니터링
        self.timings = {
            'total': deque(maxlen=50),
            'semantic': deque(maxlen=50),
            'align_gpu': deque(maxlen=50),
            'depth_to_pc': deque(maxlen=50),
            'transform': deque(maxlen=50),
            'pcl_pub': deque(maxlen=50),
        }
        self.last_report_time = time.time()

        self.get_logger().info('✅ NPU Semantic PointCloud Node initialized')
        self.get_logger().info(f"  RGB Topic: {self.config.ros.rgb_topic}")
        self.get_logger().info(f"  Depth Topic: {self.config.ros.depth_topic}")
        self.get_logger().info(f"  PCL Topic: {self.config.ros.semantic_pointcloud_topic}")

    def _setup_ros_parameters(self):
        """ROS 파라미터 선언 및 설정"""
        # ROS Topics
        self.declare_parameter('depth_topic', self.config.ros.depth_topic)
        self.declare_parameter('rgb_topic', self.config.ros.rgb_topic)
        self.declare_parameter('source_frame', self.config.ros.source_frame)
        self.declare_parameter('target_frame', self.config.ros.target_frame)
        self.declare_parameter('sync_slop', self.config.ros.sync_slop)
        self.declare_parameter('semantic_pointcloud_topic', self.config.ros.semantic_pointcloud_topic)

        # Depth Camera
        self.declare_parameter('depth_cam.fx', self.config.depth_cam.fx)
        self.declare_parameter('depth_cam.fy', self.config.depth_cam.fy)
        self.declare_parameter('depth_cam.cx', self.config.depth_cam.cx)
        self.declare_parameter('depth_cam.cy', self.config.depth_cam.cy)
        self.declare_parameter('depth_cam.height', self.config.depth_cam.height)
        self.declare_parameter('depth_cam.width', self.config.depth_cam.width)

        # RGB Camera
        self.declare_parameter('rgb_cam.fx', self.config.rgb_cam.fx)
        self.declare_parameter('rgb_cam.fy', self.config.rgb_cam.fy)
        self.declare_parameter('rgb_cam.cx', self.config.rgb_cam.cx)
        self.declare_parameter('rgb_cam.cy', self.config.rgb_cam.cy)
        self.declare_parameter('rgb_cam.height', self.config.rgb_cam.height)
        self.declare_parameter('rgb_cam.width', self.config.rgb_cam.width)

        # Point Cloud
        self.declare_parameter('pcl.downsample_y', self.config.pointcloud.downsample_y)
        self.declare_parameter('pcl.downsample_x', self.config.pointcloud.downsample_x)

        # 파라미터 값 가져오기
        self.config.ros.depth_topic = self.get_parameter('depth_topic').value
        self.config.ros.rgb_topic = self.get_parameter('rgb_topic').value
        self.config.ros.source_frame = self.get_parameter('source_frame').value
        self.config.ros.target_frame = self.get_parameter('target_frame').value
        self.config.ros.sync_slop = self.get_parameter('sync_slop').value
        self.config.ros.semantic_pointcloud_topic = self.get_parameter('semantic_pointcloud_topic').value

        # Depth Camera
        self.config.depth_cam.fx = self.get_parameter('depth_cam.fx').value
        self.config.depth_cam.fy = self.get_parameter('depth_cam.fy').value
        self.config.depth_cam.cx = self.get_parameter('depth_cam.cx').value
        self.config.depth_cam.cy = self.get_parameter('depth_cam.cy').value
        self.config.depth_cam.height = self.get_parameter('depth_cam.height').value
        self.config.depth_cam.width = self.get_parameter('depth_cam.width').value

        # RGB Camera
        self.config.rgb_cam.fx = self.get_parameter('rgb_cam.fx').value
        self.config.rgb_cam.fy = self.get_parameter('rgb_cam.fy').value
        self.config.rgb_cam.cx = self.get_parameter('rgb_cam.cx').value
        self.config.rgb_cam.cy = self.get_parameter('rgb_cam.cy').value
        self.config.rgb_cam.height = self.get_parameter('rgb_cam.height').value
        self.config.rgb_cam.width = self.get_parameter('rgb_cam.width').value

        # Point Cloud
        self.config.pointcloud.downsample_y = self.get_parameter('pcl.downsample_y').value
        self.config.pointcloud.downsample_x = self.get_parameter('pcl.downsample_x').value

    def _setup_ros_communication(self):
        """ROS 통신 설정 (Subscribers, Publishers, TF)"""
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, # stable connection
            history=HistoryPolicy.KEEP_LAST,
            depth=1 # latest frame
        )

        depth_sub = message_filters.Subscriber(
            self, Image, self.config.ros.depth_topic, qos_profile=qos_profile
        )
        rgb_sub = message_filters.Subscriber(
            self, Image, self.config.ros.rgb_topic, qos_profile=qos_profile
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [depth_sub, rgb_sub],
            queue_size=10,
            slop=self.config.ros.sync_slop
        )
        self.sync.registerCallback(self.rgbd_callback)

        self.sem_pc_pub = self.create_publisher(
            PointCloud2, self.config.ros.semantic_pointcloud_topic, qos_profile
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Point Cloud 필드 정의
        self.semantic_pointcloud_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='label', offset=16, datatype=PointField.UINT32, count=1),
        ]
        self.point_step_pcl = 20

    def yolov9c_seg_helper(
        self,
        img_size: Union[Tuple[int], List[int]] = None,
        conf_thres: float = None,
        iou_thres: float = None,
        device: str = None,
    ):   
        helper = YoloHelper.make_from_yaml(
            "./mblt_infer_original/model_configs/yolov9c_seg_640.yaml", 
            device
        )
        helper.set_inference_param(
            img_size=img_size, conf_thres=conf_thres, iou_thres=iou_thres
        )
        return helper

    def _init_gpu_parameters(self):
        """GPU에서 사용할 파라미터 미리 생성"""
        # Depth 카메라용 픽셀 그리드
        v, u = torch.meshgrid(
            torch.arange(self.config.depth_cam.height, device=self.device, dtype=torch.float32),
            torch.arange(self.config.depth_cam.width, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        self.u_grid_d = u
        self.v_grid_d = v
        self.fx_d_tensor = torch.tensor(self.config.depth_cam.fx, device=self.device, dtype=torch.float32)
        self.fy_d_tensor = torch.tensor(self.config.depth_cam.fy, device=self.device, dtype=torch.float32)
        self.cx_d_tensor = torch.tensor(self.config.depth_cam.cx, device=self.device, dtype=torch.float32)
        self.cy_d_tensor = torch.tensor(self.config.depth_cam.cy, device=self.device, dtype=torch.float32)

        # RGB 카메라용 파라미터
        self.fx_rgb_tensor = torch.tensor(self.config.rgb_cam.fx, device=self.device, dtype=torch.float32)
        self.fy_rgb_tensor = torch.tensor(self.config.rgb_cam.fy, device=self.device, dtype=torch.float32)
        self.cx_rgb_tensor = torch.tensor(self.config.rgb_cam.cx, device=self.device, dtype=torch.float32)
        self.cy_rgb_tensor = torch.tensor(self.config.rgb_cam.cy, device=self.device, dtype=torch.float32)

        # fixed Extrinsics (Depth -> Color)
        # Homogeneous Coordinates 4x4 
        rotation_flat = [
            0.9999944567680359, 0.0004453109868336469, -0.003304719226434827,
            -0.00045781597145833075, 0.9999927282333374, -0.0037841906305402517,
            0.003303010016679764, 0.0037856826093047857, 0.9999873638153076
        ]
        translation_vec = [-0.05908159539103508, 1.4681237189506646e-05, 0.00048153731040656567]
        
        T_color_from_depth_np = np.eye(4, dtype=np.float32)
        
        # Rotation Matrix
        T_color_from_depth_np[:3, :3] = np.array(rotation_flat).reshape(3, 3)
        # translation Matrix
        T_color_from_depth_np[:3, 3] = np.array(translation_vec)

        self.T_color_from_depth_gpu = torch.from_numpy(T_color_from_depth_np).to(self.device)
        self.get_logger().info('고정 Extrinsics (T_color_from_depth) GPU에 로드 완료')

    def _init_semantic_colormap(self):
        """시맨틱 라벨을 RGB로 변환하기 위한 GPU 컬러맵 생성"""
        np.random.seed(12)
        colors_np = np.random.randint(0, 255, size=(self.config.semantic.num_labels, 3), dtype=np.uint8)
        colors_np[0] = [0, 0, 0]  # background
        
        self.semantic_colormap_gpu = torch.from_numpy(colors_np).to(self.device)
        self.get_logger().info(f'GPU 시맨틱 컬러맵 생성 완료 ({self.config.semantic.num_labels} classes)')
    
    def preprocess_image(self, cv_image):
        """NPU 추론을 위한 이미지 전처리"""
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        processed_image = self.helper.pre_process(image_rgb)
        return np.expand_dims(processed_image, axis=0)

    def npu_inference(self, rgb_image):
        """
        NPU를 이용한 YOLOv9 Segmentation 추론
        Returns: instance masks와 labels (NumPy 배열)
        """
        h_orig, w_orig = rgb_image.shape[:2]
        
        # 1. 전처리
        input_tensor = self.preprocess_image(rgb_image)
        
        # 2. NPU 추론
        out_npu = self.mxq_model.infer(input_tensor)
        
        # 3. 후처리 (NMS)
        processing_shapes_list = [(h_orig, w_orig)]
        nms_outs = self.helper.post_process(out_npu, processing_shapes_list)
        img_shape_processed = input_tensor.shape[-2:]
        results = self.helper.post_process.nmsout2eval(
            nms_outs, img_shape_processed, processing_shapes_list
        )
        
        return results

    def create_semantic_mask_from_instances(self, results, img_shape):
        """
        인스턴스 세그멘테이션 결과를 시맨틱 마스크로 변환
        Returns: (H, W) semantic mask
        """
        h, w = img_shape
        semantic_mask = np.zeros((h, w), dtype=np.uint8)
        
        labels_list, boxes_list, scores_list, extra_list = results
        
        if not labels_list or len(labels_list[0]) == 0:
            return semantic_mask
        
        labels = labels_list[0]
        extras = extra_list[0]
        
        # 각 인스턴스의 마스크를 시맨틱 마스크에 그리기
        for label, extra in zip(labels, extras):
            if extra and 'counts' in extra:
                mask = maskUtils.decode(extra)
                class_idx = int(label)
                semantic_mask[mask > 0] = class_idx
        
        return semantic_mask

    def project_points_to_rgb_grid(self, points_in_color_frame):
        """3D 포인트를 RGB 2D 픽셀 그리드로 프로젝션"""
        X = points_in_color_frame[..., 0]
        Y = points_in_color_frame[..., 1]
        Z = points_in_color_frame[..., 2]

        z_mask = Z > 1e-6
        Z_safe = torch.where(z_mask, Z, 1e-6)

        u = self.fx_rgb_tensor * X / Z_safe + self.cx_rgb_tensor
        v = self.fy_rgb_tensor * Y / Z_safe + self.cy_rgb_tensor

        norm_u = (u / (self.config.rgb_cam.width - 1.0)) * 2.0 - 1.0
        norm_v = (v / (self.config.rgb_cam.height - 1.0)) * 2.0 - 1.0

        normalized_grid = torch.stack([norm_u, norm_v], dim=-1)

        sampling_mask = z_mask & \
                        (norm_u >= -1.0) & (norm_u <= 1.0) & \
                        (norm_v >= -1.0) & (norm_v <= 1.0)

        return normalized_grid, sampling_mask

    def rgbd_callback(self, depth_msg, rgb_msg):
        """Depth, RGB 동시 수신 및 전체 파이프라인 처리"""
        t_start = time.perf_counter()

        try:
            # 1. NPU 추론
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            
            t_sem_start = time.perf_counter()
            results = self.npu_inference(rgb_image)
            semantic_mask_rgb_res = self.create_semantic_mask_from_instances(
                results, rgb_image.shape[:2]
            )
            self.timings['semantic'].append((time.perf_counter() - t_sem_start) * 1000)

            # GPU 업로드
            mask_tensor_rgb_res = torch.from_numpy(semantic_mask_rgb_res).to(self.device)

            # 2. Depth 이미지 GPU 업로드
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding=depth_msg.encoding
            ).astype(np.float32)
            
            depth_tensor = torch.from_numpy(depth_image).to(self.device) / 1000.0
            rgb_tensor_rgb_res = torch.from_numpy(rgb_image).to(self.device)

            # 3. 3D 재구성
            t_depth_start = time.perf_counter()
            pointcloud_cam_depth_frame = self.depth_to_pointcloud_gpu(depth_tensor)
            self.timings['depth_to_pc'].append((time.perf_counter() - t_depth_start) * 1000)

            # 4. GPU 기반 정렬
            t_align_start = time.perf_counter()

            pointcloud_cam_color_frame = self.apply_transform_gpu(
                pointcloud_cam_depth_frame, self.T_color_from_depth_gpu
            )

            normalized_uv_grid, sampling_mask = self.project_points_to_rgb_grid(
                pointcloud_cam_color_frame
            )
            
            rgb_for_interp = rgb_tensor_rgb_res.permute(2, 0, 1).float().unsqueeze(0)
            normalized_uv_for_grid_sample = normalized_uv_grid.unsqueeze(0)

            rgb_aligned_interp = F.grid_sample(
                rgb_for_interp, normalized_uv_for_grid_sample, 
                mode='bilinear', padding_mode='zeros', align_corners=False
            )
            
            mask_for_interp = mask_tensor_rgb_res.float().unsqueeze(0).unsqueeze(0)
            mask_aligned_interp = F.grid_sample(
                mask_for_interp, normalized_uv_for_grid_sample, 
                mode='nearest', padding_mode='zeros', align_corners=False
            )

            rgb_aligned_bgr = rgb_aligned_interp.squeeze().permute(1, 2, 0).to(torch.uint8)
            mask_aligned = mask_aligned_interp.squeeze().long()

            invalid_mask = (depth_tensor <= 0.01) | (~sampling_mask)
            rgb_aligned_bgr[invalid_mask] = 0
            mask_aligned[invalid_mask] = 0
            pointcloud_cam_depth_frame[invalid_mask] = 0.0

            self.timings['align_gpu'].append((time.perf_counter() - t_align_start) * 1000)

            # 5. TF 조회 및 좌표 변환
            t_tf_start = time.perf_counter()
            transform = self.tf_buffer.lookup_transform(
                self.config.ros.target_frame, self.config.ros.source_frame, rclpy.time.Time()
            )
            transform_matrix = self.transform_to_matrix(transform)
            
            transformed_cloud = self.apply_transform_gpu(pointcloud_cam_depth_frame, transform_matrix)
            self.timings['transform'].append((time.perf_counter() - t_tf_start) * 1000)

            # 6. PointCloud 메시지 발행
            stamp = depth_msg.header.stamp

            t_pcl_start = time.perf_counter()
            self.process_and_publish_semantic_pointcloud(
                transformed_cloud, rgb_aligned_bgr, mask_aligned, stamp
            )
            self.timings['pcl_pub'].append((time.perf_counter() - t_pcl_start) * 1000)

            self.timings['total'].append((time.perf_counter() - t_start) * 1000)
            self._report_stats()

        except TransformException as e:
            self.get_logger().warn(f'TF 변환 실패: {e}', throttle_duration_sec=1.0)
        except Exception as e:
            self.get_logger().error(f'처리 오류: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def depth_to_pointcloud_gpu(self, depth_tensor):
        """GPU를 이용한 Depth to Point Cloud 변환"""
        z = depth_tensor
        x = (self.u_grid_d - self.cx_d_tensor) * z / self.fx_d_tensor
        y = (self.v_grid_d - self.cy_d_tensor) * z / self.fy_d_tensor
        return torch.stack([x, y, z], dim=-1)

    def apply_transform_gpu(self, points, matrix):
        """GPU를 이용한 좌표 변환"""
        original_shape = points.shape
        points_flat = points.reshape(-1, 3)
        
        if isinstance(matrix, np.ndarray):
            matrix_tensor = torch.from_numpy(matrix).to(self.device, dtype=torch.float32)
        else:
            matrix_tensor = matrix

        ones = torch.ones((points_flat.shape[0], 1), device=self.device, dtype=torch.float32)
        homogeneous = torch.cat([points_flat, ones], dim=1)
        transformed = torch.mm(homogeneous, matrix_tensor.T)
        return transformed[:, :3].reshape(original_shape)

    def transform_to_matrix(self, transform):
        """ROS Transform 메시지를 4x4 동차 변환 행렬로 변환"""
        t = transform.transform.translation
        translation = np.array([t.x, t.y, t.z])
        r = transform.transform.rotation
        quat = [r.w, r.x, r.y, r.z]
        rotation_matrix = quat2mat(quat)
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = translation
        return matrix

    def process_and_publish_semantic_pointcloud(
        self, transformed_cloud, rgb_aligned_bgr, mask_aligned, stamp
    ):
        """Semantic 3D 포인트 클라우드를 다운샘플링, 패킹 후 발행"""

        # 1. 다운샘플링
        points = transformed_cloud[::self.config.pointcloud.downsample_y, ::self.config.pointcloud.downsample_x, :]
        colors_bgr = rgb_aligned_bgr[::self.config.pointcloud.downsample_y, ::self.config.pointcloud.downsample_x, :]
        labels = mask_aligned[::self.config.pointcloud.downsample_y, ::self.config.pointcloud.downsample_x]

        # 2. Flatten
        points_flat = points.reshape(-1, 3)
        colors_flat_bgr = colors_bgr.reshape(-1, 3)
        labels_flat = labels.reshape(-1)

        # 3. 유효한 포인트 필터링
        valid_mask = points_flat[:, 2] > -10.01
        
        points_valid = points_flat[valid_mask]
        colors_valid_bgr = colors_flat_bgr[valid_mask]
        labels_valid = labels_flat[valid_mask]

        num_points = points_valid.shape[0]
        if num_points == 0:
            return

        # 4. RGB 패킹 (BGR -> RGB 변환)
        r = colors_valid_bgr[:, 2].long()  # R은 BGR의 2번 인덱스
        g = colors_valid_bgr[:, 1].long()  # G는 BGR의 1번 인덱스
        b = colors_valid_bgr[:, 0].long()  # B는 BGR의 0번 인덱스
        
        rgb_packed_gpu = (r << 16) | (g << 8) | b
        rgb_float32_gpu = rgb_packed_gpu.to(torch.uint32).view(torch.float32)

        # 5. Label 패킹
        # labels_float32_gpu = labels_valid.long().to(torch.uint32).view(torch.float32)


        # 최대 레이블 인덱스 (배경 0 제외)
        max_class_idx = self.config.semantic.num_labels - 1 

        # 새로운 레이블 인덱스 계산 (역순 매핑)
        new_labels = labels_valid.clone()

        
        non_background_mask = labels_valid != 0
        new_labels[non_background_mask] = max_class_idx - labels_valid[non_background_mask] + 1
        
        labels_float32_gpu = new_labels.long().to(torch.uint32).view(torch.float32)
        
        
        # 6. 데이터 결합
        data_gpu = torch.stack(
            [
                points_valid[:, 0], 
                points_valid[:, 1], 
                points_valid[:, 2], 
                rgb_float32_gpu, 
                labels_float32_gpu
            ],
            dim=-1
        )

        # 7. GPU -> CPU 전송
        data_np = data_gpu.cpu().numpy()

        # 8. PointCloud2 메시지 생성
        pointcloud_msg = self._create_semantic_cloud_from_data(
            data_np, stamp, self.config.ros.target_frame
        )

        # 9. 발행
        self.sem_pc_pub.publish(pointcloud_msg)
    
    def _create_semantic_cloud_from_data(self, data_np, stamp, frame_id):
        """(N, 5) [x, y, z, rgb_float32, label_float32] NumPy 배열로 PointCloud2 메시지 생성"""
        header = Header(stamp=stamp, frame_id=frame_id)
        num_points = data_np.shape[0]
        
        fields = self.semantic_pointcloud_fields
        point_step = self.point_step_pcl

        return PointCloud2(
            header=header,
            height=1,
            width=num_points,
            fields=fields,
            is_bigendian=False,
            point_step=point_step,
            row_step=point_step * num_points,
            data=data_np.astype(np.float32).tobytes(),
            is_dense=True,
        )

    def _report_stats(self):
        """성능 통계 출력"""
        if time.time() - self.last_report_time < 2.0:
            return
            
        if not self.timings['total']:
            return

        avg_total = np.mean(self.timings['total'])
        fps = 1000.0 / avg_total
        avg_sem = np.mean(self.timings['semantic'])
        avg_align = np.mean(self.timings['align_gpu'])
        avg_depth = np.mean(self.timings['depth_to_pc'])
        avg_tf = np.mean(self.timings['transform'])
        avg_pcl = np.mean(self.timings['pcl_pub'])

        msg = f"\n📊 [NPU-SemanticPCL] FPS: {fps:.1f} Hz (Total: {avg_total:.1f} ms)\n" \
              f"  ├─ NPU Infer : {avg_sem:6.1f} ms\n" \
              f"  ├─ Depth→PC  : {avg_depth:6.1f} ms\n" \
              f"  ├─ Align GPU : {avg_align:6.1f} ms\n" \
              f"  ├─ Transform : {avg_tf:6.1f} ms\n" \
              f"  └─ PCL Pub   : {avg_pcl:6.1f} ms"
        
        self.get_logger().info(msg)
        self.last_report_time = time.time()


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    
    node = NPUSemanticPointCloudNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down...')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()