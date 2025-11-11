#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import torch

class PointCloudBEVNode(Node):
    def __init__(self):
        super().__init__('pointcloud_bev_node')
        self.bridge = CvBridge()
        self.device = torch.device('cuda')
        self.get_logger().info(f'🚀 CUDA GPU 가속 활성화 (PyTorch, {self.device})')

        # --- 기존 파라미터 ---
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('pointcloud_topic', '/pointcloud')
        self.declare_parameter('source_frame', 'camera_depth_optical_frame')
        self.declare_parameter('target_frame', 'camera_link')
        self.declare_parameter('cam.fx', 431.0625)
        self.declare_parameter('cam.fy', 431.0625)
        self.declare_parameter('cam.cx', 434.492)
        self.declare_parameter('cam.cy', 242.764)
        self.declare_parameter('cam.height', 480)
        self.declare_parameter('cam.width', 848)
        self.declare_parameter('pcl.downsample_y', 6)
        self.declare_parameter('pcl.downsample_x', 4)
        self.declare_parameter('bev_topic', '/bev_map')
        self.declare_parameter('bev.z_min', -0.35)
        self.declare_parameter('bev.z_max', 1.0)
        self.declare_parameter('bev.resolution', 0.05)
        self.declare_parameter('bev.size_x', 50.0) # 30
        self.declare_parameter('bev.size_y', 40.0)
        self.declare_parameter('bev.min_points_per_cell', 5) # 기존 밀도 필터

        ### --- [NEW] 시간적 필터 파라미터 --- ###
        # 감쇠율 (0.0 ~ 1.0): 작을수록 과거 데이터가 빨리 사라짐 (반응성↑, 노이즈제거↓)
        self.declare_parameter('bev.temporal.decay', 0.9) 
        # 증가율 (0.0 ~ 1.0): 클수록 새로운 장애물이 빨리 나타남
        self.declare_parameter('bev.temporal.increase', 0.6)
        # 임계값 (0.0 ~ 1.0): 이 확률 이상이어야 실제 장애물로 표시
        self.declare_parameter('bev.temporal.threshold', 0.95)
        ##########################################

        # --- 파라미터 로드 ---
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
        bev_topic = self.get_parameter('bev_topic').value
        self.z_min = self.get_parameter('bev.z_min').value
        self.z_max = self.get_parameter('bev.z_max').value
        self.resolution = self.get_parameter('bev.resolution').value
        self.size_x = self.get_parameter('bev.size_x').value
        self.size_y = self.get_parameter('bev.size_y').value
        self.min_points_per_cell = self.get_parameter('bev.min_points_per_cell').value
        
        ### --- [NEW] --- ###
        self.decay_rate = self.get_parameter('bev.temporal.decay').value
        self.increase_rate = self.get_parameter('bev.temporal.increase').value
        self.occupancy_threshold = self.get_parameter('bev.temporal.threshold').value
        #####################

        self.cells_x = int(self.size_x / self.resolution)
        self.cells_y = int(self.size_y / self.resolution)
        self.grid_origin_x = 0.0
        self.grid_origin_y = -self.size_y / 2.0

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, depth_topic, self.depth_callback, qos_profile)
        self.pointcloud_pub = self.create_publisher(PointCloud2, pointcloud_topic, qos_profile)
        self.bev_pub = self.create_publisher(PointCloud2, bev_topic, qos_profile)

        self.pointcloud_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        self.point_step = 16

        self._init_gpu_parameters()
        self.get_logger().info(f'✅ Temporal Filter Active (Decay: {self.decay_rate}, Thresh: {self.occupancy_threshold})')

    def _init_gpu_parameters(self):
        # GPU params for pointcloud transformation accelrating
        v, u = torch.meshgrid(torch.arange(self.cam_height, device=self.device, dtype=torch.float32), torch.arange(self.cam_width, device=self.device, dtype=torch.float32), indexing='ij')
        self.u_grid = u; self.v_grid = v
        self.fx_tensor = torch.tensor(self.fx, device=self.device, dtype=torch.float32)
        self.fy_tensor = torch.tensor(self.fy, device=self.device, dtype=torch.float32)
        self.cx_tensor = torch.tensor(self.cx, device=self.device, dtype=torch.float32)
        self.cy_tensor = torch.tensor(self.cy, device=self.device, dtype=torch.float32)
        self.z_min_t = torch.tensor(self.z_min, device=self.device, dtype=torch.float32)
        self.z_max_t = torch.tensor(self.z_max, device=self.device, dtype=torch.float32)
        self.z_range_t = self.z_max_t - self.z_min_t
        self.resolution_t = torch.tensor(self.resolution, device=self.device, dtype=torch.float32)
        self.grid_origin_x_t = torch.tensor(self.grid_origin_x, device=self.device, dtype=torch.float32)
        self.grid_origin_y_t = torch.tensor(self.grid_origin_y, device=self.device, dtype=torch.float32)

        # 재사용 텐서들
        self.bev_heights_flat = torch.full((self.cells_y * self.cells_x,), -torch.inf, device=self.device, dtype=torch.float32)
        self.bev_counts_flat = torch.zeros((self.cells_y * self.cells_x,), device=self.device, dtype=torch.int32)

        ### --- [NEW] 시간적 점유 확률 그리드 (History) --- ###
        # 이 텐서는 매 프레임 초기화되지 않고 계속 값을 유지합니다.
        self.bev_occupancy_grid = torch.zeros(
            (self.cells_y * self.cells_x,), 
            device=self.device, 
            dtype=torch.float32
        )
        #####################################################

        # camera frame to robot frame 3D transformation only Rotation : Homegeneuous coordinates
        matrix = np.array(
            [[0.,0.,1.,0.0], 
             [-1.,0.,0.,0.], 
             [0.,-1.,0.,0.], 
             [0.,0.,0.,1.]], dtype=np.float32
        )
        
        self.transform_matrix = torch.from_numpy(matrix).to(self.device, dtype=torch.float32)

    def depth_callback(self, msg):
        # depth callback 
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding).astype(np.float32) / 1000.0
            # numpy.ndarray (480,848)
            
            depth_tensor = torch.from_numpy(depth_image).to(self.device) # using cuda!!!!!
            # torch.Tensor (480,848) : only z value : depth image 
            
            pointcloud_cam = self.depth_to_pointcloud_gpu(depth_tensor)
            # torch.Tensor (480,848,3) : x,y,z value : point cloud
            
            transformed_cloud = self.apply_transform_gpu(pointcloud_cam, self.transform_matrix)
            # torch.Tensor (480,848,3 ) : extrinsic transformation : camera frame to robot frame : homogeneous coordinates transformation
            
            stamp = msg.header.stamp
            
            self.process_and_publish_pointcloud(transformed_cloud, stamp)
            
            self.process_and_publish_bev(transformed_cloud, stamp)
        except Exception as e:
            self.get_logger().error(f'Error: {e}')

    # (depth_to_pointcloud_gpu, apply_transform_gpu, process_and_publish_pointcloud 동일) 
    def depth_to_pointcloud_gpu(self, depth_tensor):
        z = depth_tensor # depth value
        x = (self.u_grid - self.cx_tensor) * z / self.fx_tensor # broadcasting!! + inverse intrinsic matrix 
        y = (self.v_grid - self.cy_tensor) * z / self.fy_tensor
        return torch.stack([x, y, z], dim=-1)

    def apply_transform_gpu(self, points, matrix):
        original_shape = points.shape
        points_flat = points.reshape(-1, 3)
        
        ones = torch.ones((points_flat.shape[0], 1), device=self.device, dtype=torch.float32)
        homogeneous = torch.cat([points_flat, ones], dim=1)
        transformed = torch.mm(homogeneous, matrix.T)
        return transformed[:, :3].reshape(original_shape)

    def process_and_publish_pointcloud(self, transformed_cloud, stamp):
        sampled = transformed_cloud[::self.downsample_y, ::self.downsample_x, :]
        points = sampled.reshape(-1, 3)
        points_np = points.cpu().numpy()
        if points_np.shape[0] == 0: return
        colors = np.zeros((points_np.shape[0], 3), dtype=np.uint8)
        colors[:, 0] = 100; colors[:, 1] = 200; colors[:, 2] = 200
        self.pointcloud_pub.publish(self.create_pointcloud_msg(points_np, colors, stamp, self.target_frame))

    # grid bev map
    def process_and_publish_bev(self, transformed_cloud, stamp):
        x_flat = transformed_cloud[..., 0].ravel()
        y_flat = transformed_cloud[..., 1].ravel()
        z_flat = transformed_cloud[..., 2].ravel()

        mask = (z_flat > self.z_min_t) & (z_flat < self.z_max_t)

        # bev transformation
        grid_c = ((x_flat - self.grid_origin_x_t) / self.resolution_t).long()
        grid_r = ((y_flat - self.grid_origin_y_t) / self.resolution_t).long()
        mask &= (grid_c >= 0) & (grid_c < self.cells_x) & (grid_r >= 0) & (grid_r < self.cells_y)

        valid_z = z_flat[mask]
        if valid_z.shape[0] == 0: 
             # 장애물이 아예 없어도 Decay는 진행되어야 잔상이 사라짐
            self.bev_occupancy_grid.mul_(self.decay_rate)
            return

        valid_r = grid_r[mask]
        valid_c = grid_c[mask]
        linear_indices = valid_r * self.cells_x + valid_c

        # 1. 현재 프레임 분석 (높이 & 밀도)
        self.bev_heights_flat.fill_(-torch.inf)
        self.bev_heights_flat.index_reduce_(dim=0, index=linear_indices, source=valid_z, reduce="amax", include_self=False)
        
        self.bev_counts_flat.fill_(0)
        ones = torch.ones_like(linear_indices, dtype=torch.int32)
        self.bev_counts_flat.index_add_(dim=0, index=linear_indices, source=ones)

        # 2. 현재 프레임에서 유효한 장애물 후보 식별
        current_frame_mask = (self.bev_heights_flat > -torch.inf) & \
                             (self.bev_counts_flat >= self.min_points_per_cell)

        ### --- [NEW] 시간적 필터 업데이트 (핵심 로직) --- ###
        
        # 2.1 Decay: 모든 셀의 확률을 감소시킴 (과거 기록 흐리기)
        self.bev_occupancy_grid.mul_(self.decay_rate)

        # 2.2 Increase: 현재 프레임에서 탐지된 셀의 확률을 증가시킴
        # current_frame_mask가 True인 인덱스만 골라서 increase_rate 더하기
        # where()[0]으로 인덱스 추출
        detected_indices = torch.where(current_frame_mask)[0]
        
        if detected_indices.shape[0] > 0:
             # 해당 인덱스에만 값 더하기
             # (직접 인덱싱하여 더하는 것이 빠름)
             self.bev_occupancy_grid[detected_indices] += self.increase_rate

        # 2.3 Clamp: 확률은 0.0 ~ 1.0 사이 유지
        self.bev_occupancy_grid.clamp_(0.0, 1.0)

        # 3. 최종 출력 마스크: 누적 확률이 임계값을 넘는 셀만 표시
        final_mask = self.bev_occupancy_grid >= self.occupancy_threshold
        #######################################################

        valid_indices_flat = torch.where(final_mask)[0]
        if valid_indices_flat.shape[0] == 0: return

        # 색상 표시를 위한 높이값은 현재 프레임 값 사용 (없으면 과거값이라도 쓰기 위해 약간의 트릭 필요하나, 일단 현재값 사용)
        # 만약 현재 프레임에 없는데 과거 잔상때문에 표시된다면 높이가 -inf일 수 있음.
        # 이를 방지하기 위해 높이 맵도 약간의 persistence를 주거나, 안전하게 0.0 등으로 처리
        height_values = self.bev_heights_flat[valid_indices_flat]
        # 혹시 높이 정보가 없는(잔상만 남은) 셀이 있다면 안전한 값으로 대체
        height_values = torch.where(height_values == -torch.inf, self.z_min_t, height_values)

        r_idx_bev = torch.div(valid_indices_flat, self.cells_x, rounding_mode='floor')
        c_idx_bev = valid_indices_flat % self.cells_x
        x_world = self.grid_origin_x_t + (c_idx_bev.float() + 0.5) * self.resolution_t
        y_world = self.grid_origin_y_t + (r_idx_bev.float() + 0.5) * self.resolution_t
        z_world = torch.zeros_like(x_world)

        rgb_float32_gpu = self._height_to_color_gpu(height_values)
        bev_data_gpu = torch.stack([x_world, y_world, z_world, rgb_float32_gpu], dim=-1)
        self.bev_pub.publish(self._create_cloud_from_data(bev_data_gpu.cpu().numpy(), stamp, self.target_frame))

    # ... (_height_to_color_gpu, create_pointcloud_msg, _create_cloud_from_data, main 동일) ...
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
        r_val = (r * 255).long(); g_val = (g * 255).long(); b_val = (b * 255).long()
        rgb_packed_gpu = (r_val * 65536) + (g_val * 256) + b_val
        return rgb_packed_gpu.to(torch.uint32).view(torch.float32)

    def create_pointcloud_msg(self, points_np, colors_np, stamp, frame_id):
        header = Header(stamp=stamp, frame_id=frame_id)
        rgb_uint32 = ((colors_np[:, 0].astype(np.uint32) << 16) | (colors_np[:, 1].astype(np.uint32) << 8) | (colors_np[:, 2].astype(np.uint32)))
        pointcloud_data = np.hstack([points_np.astype(np.float32), rgb_uint32.view(np.float32).reshape(-1, 1)])
        return PointCloud2(header=header, height=1, width=pointcloud_data.shape[0], fields=self.pointcloud_fields, is_bigendian=False, point_step=self.point_step, row_step=self.point_step * pointcloud_data.shape[0], data=pointcloud_data.tobytes(), is_dense=True)

    def _create_cloud_from_data(self, point_data_np, stamp, frame_id):
        header = Header(stamp=stamp, frame_id=frame_id)
        num_points = point_data_np.shape[0]
        return PointCloud2(header=header, height=1, width=num_points, fields=self.pointcloud_fields, is_bigendian=False, point_step=self.point_step, row_step=self.point_step * num_points, data=point_data_np.astype(np.float32).tobytes(), is_dense=True)

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudBEVNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
