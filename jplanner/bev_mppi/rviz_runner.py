#!/usr/bin/env python3
# runner.py

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist, Point, PoseStamped, PointStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header

import cv2
import numpy as np
import math
import traceback
import time
import threading

# BEV Map 처리를 위해
import sensor_msgs_py.point_cloud2 as pc2

# --- MPPI 핵심 라이브러리 ---
import torch
# -------------------------

# --- 모듈화된 코드 임포트 ---
from optimized_controller import MPPIController
# -----------------------------

class MPPIBevPlanner(Node):
    """
    MPPI 플래너 메인 노드 (RViz2 시각화 지원 버전)
    """

    def quaternion_to_yaw_from_parts(self, w, x, y, z):
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def __init__(self):
        super().__init__('mppi_bev_planner_rviz_node')

        # --- 1. ROS 2 파라미터 ---
        self.declare_parameter('grid_resolution', 0.1)
        self.declare_parameter('grid_size_x', 50.0)
        self.declare_parameter('grid_size_y', 30.0)
        self.declare_parameter('inflation_radius', 0.1)
        self.declare_parameter('max_linear_velocity', 0.9)
        self.declare_parameter('min_linear_velocity', 0.15)
        self.declare_parameter('max_angular_velocity', 1.0)
        self.declare_parameter('goal_threshold', 0.5)
        self.declare_parameter('yaw_threshold', 0.4)
        self.declare_parameter('yaw_p_gain', 0.5)
        self.declare_parameter('min_align_angular_velocity', 0.1)
        self.declare_parameter('mppi_k', 5000)
        self.declare_parameter('mppi_t', 40)
        self.declare_parameter('mppi_dt', 0.1)
        self.declare_parameter('mppi_lambda', 1.0)
        self.declare_parameter('mppi_sigma_v', 0.1)
        self.declare_parameter('mppi_sigma_w', 0.3)
        self.declare_parameter('goal_cost_weight', 95.0)
        self.declare_parameter('obstacle_cost_weight', 244.0)
        self.declare_parameter('control_cost_weight', 0.1)
        self.declare_parameter('num_samples_to_plot', 50) # RViz에 표시할 샘플 개수
        self.declare_parameter('collision_check_distance', 0.0)
        self.declare_parameter('collision_check_width', 0.25)
        self.declare_parameter('collision_cost_threshold', 250.0)
        self.declare_parameter('robot_frame', 'base_link') # 로봇 기준 좌표계 이름

        # --- 파라미터 가져오기 ---
        self.grid_res = self.get_parameter('grid_resolution').value
        self.size_x = self.get_parameter('grid_size_x').value
        self.size_y = self.get_parameter('grid_size_y').value
        self.inflation_radius = self.get_parameter('inflation_radius').value
        self.max_v = self.get_parameter('max_linear_velocity').value
        self.min_v = self.get_parameter('min_linear_velocity').value
        self.max_w = self.get_parameter('max_angular_velocity').value
        self.goal_threshold = self.get_parameter('goal_threshold').value
        self.yaw_threshold = self.get_parameter('yaw_threshold').value
        self.yaw_p_gain = self.get_parameter('yaw_p_gain').value
        self.min_align_w = self.get_parameter('min_align_angular_velocity').value
        self.K = self.get_parameter('mppi_k').value
        self.T = self.get_parameter('mppi_t').value
        self.dt = self.get_parameter('mppi_dt').value
        self.lambda_ = self.get_parameter('mppi_lambda').value
        self.goal_cost_w = self.get_parameter('goal_cost_weight').value
        self.obstacle_cost_w = self.get_parameter('obstacle_cost_weight').value
        self.control_cost_w = self.get_parameter('control_cost_weight').value
        self.num_samples_to_plot = self.get_parameter('num_samples_to_plot').value
        self.collision_cost_th = self.get_parameter('collision_cost_threshold').value
        self.robot_frame = self.get_parameter('robot_frame').value

        sigma_v = self.get_parameter('mppi_sigma_v').value
        sigma_w = self.get_parameter('mppi_sigma_w').value

        # --- Grid 설정 ---
        self.cells_x = int(self.size_x / self.grid_res)
        self.cells_y = int(self.size_y / self.grid_res)
        self.grid_origin_x = -self.size_x / 2.0
        self.grid_origin_y = -self.size_y / 2.0
        inflation_cells = int(self.inflation_radius / self.grid_res)
        self.inflation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inflation_cells + 1, 2 * inflation_cells + 1))

        # 충돌 감지 ROI
        check_dist_cells = int(self.get_parameter('collision_check_distance').value / self.grid_res)
        check_width_cells = int(self.get_parameter('collision_check_width').value / self.grid_res)
        self.robot_grid_c = int((0.0 - self.grid_origin_x) / self.grid_res)
        self.robot_grid_r = int((0.0 - self.grid_origin_y) / self.grid_res)
        self.roi_r_start = max(0, self.robot_grid_r - check_width_cells // 2)
        self.roi_r_end = min(self.cells_y, self.robot_grid_r + check_width_cells // 2)
        self.roi_c_start = max(0, self.robot_grid_c)
        self.roi_c_end = min(self.cells_x, self.robot_grid_c + check_dist_cells)

        # --- 2. ROS Pub/Sub ---
        # QoS 설정 (Visualization은 Best Effort가 나을 수 있음)
        # viz_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        viz_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        reliable_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.bev_sub = self.create_subscription(PointCloud2, '/bev_map', self.bev_map_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/krm_auto_localization/odom', self.odom_callback, reliable_qos)
        self.cmd_pub = self.create_publisher(Twist, '/mcu/command/manual_twist', 10)

        # ★ RViz 시각화용 Publisher
        self.pub_opt_path = self.create_publisher(Path, '/mppi/optimal_path', viz_qos)
        self.pub_samples = self.create_publisher(Marker, '/mppi/sampled_paths', viz_qos)
        self.pub_local_goal = self.create_publisher(Marker, '/mppi/local_goal', viz_qos)
        self.pub_global_wps = self.create_publisher(MarkerArray, '/mppi/global_waypoints', reliable_qos)

        # --- 3. 상태 변수 ---
        self.current_pose = None    # [x, y, yaw] (Global)
        self.costmap_tensor = None  # (GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_task = "NAVIGATING"
        self.pause_start_time = None
        self.collision_detected_last_step = False

        # --- 4. 웨이포인트 로드 ---
        wp_data = [
            {'w': 0.6999738, 'x': 0.005913, 'y': 0.007110, 'z': 0.020897, 'pos_x': 0.0254884, 'pos_y': -0.0148898},
            {'w': 0.6999738, 'x': 0.005913, 'y': 0.007110, 'z': -0.720897, 'pos_x': 0.0254884, 'pos_y': -0.0148898},
            {'w': 0.9999738, 'x': 0.005913, 'y': 0.007110, 'z': 0.720897, 'pos_x': 0.0254884, 'pos_y': -0.0148898},
            {'w': 1.392549, 'x': 0.001612, 'y': 0.005855, 'z': 0.019710, 'pos_x': 6.337625, 'pos_y': -0.486741},
            {'w': 0.716120, 'x': -0.017303, 'y': -0.007357, 'z': -0.697722, 'pos_x': 7.698485, 'pos_y': -31.281488},
            {'w': 0.999245, 'x': -0.005607, 'y': 0.003672, 'z': -0.038267, 'pos_x': 23.286015, 'pos_y': -33.319051},
            {'w': 0.729621, 'x': -0.014739, 'y': -0.009012, 'z': -0.683632, 'pos_x': 26.809536, 'pos_y': -67.154194},
            {'w': -0.361488, 'x': 0.010711, 'y': 0.006852, 'z': 0.932289, 'pos_x': 24.974758, 'pos_y': -75.537039},
            {'w': 0.819278, 'x': -0.043234, 'y': -0.054672, 'z': -0.569142, 'pos_x': 27.027404, 'pos_y': -99.656741},
            {'w': -0.097394, 'x': -0.007766, 'y': 0.010799, 'z': 0.995156, 'pos_x': 26.906550, 'pos_y': -99.646736},
            {'w': 0.815350, 'x': -0.005566, 'y': 0.006831, 'z': 0.578900, 'pos_x': 33.534878, 'pos_y': -54.895761},
            {'w': 0.774516, 'x': 0.000336, 'y': 0.004612, 'z': 0.632536, 'pos_x': 41.289568, 'pos_y': -28.024376}
        ]
        d = []
        for wp in wp_data:
            yaw = self.quaternion_to_yaw_from_parts(wp['w'], wp['x'], wp['y'], wp['z'])
            d.append((wp['pos_x'], wp['pos_y'], yaw))

        self.waypoints = [d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9]]
        self.waypoint_index = 0
        self.get_logger().info(f"✅ Loaded {len(self.waypoints)} waypoints.")

        # --- 5. MPPI 컨트롤러 초기화 ---
        self.controller = MPPIController(
            logger=self.get_logger(), device=self.device,
            K=self.K, T=self.T, dt=self.dt, lambda_=self.lambda_,
            sigma_v=sigma_v, sigma_w=sigma_w,
            min_v=self.min_v, max_v=self.max_v, max_w=self.max_w,
            goal_cost_w=self.goal_cost_w, obstacle_cost_w=self.obstacle_cost_w, control_cost_w=self.control_cost_w,
            grid_resolution=self.grid_res, grid_origin_x=self.grid_origin_x, grid_origin_y=self.grid_origin_y,
            cells_x=self.cells_x, cells_y=self.cells_y,
            num_samples_to_plot=self.num_samples_to_plot
        )

        # --- 6. 타이머 ---
        self.control_timer = self.create_timer(self.dt, self.control_callback)
        self.viz_timer = self.create_timer(1.0, self.publish_global_waypoints) # 1초마다 글로벌 WP 발행

        self.get_logger().info("✅ MPPI BEV Planner (RViz Ready) started.")

    # --- Helper Functions ---
    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    def quaternion_to_yaw_from_msg(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def world_to_grid_idx_numpy(self, x, y):
        grid_c = int((x - self.grid_origin_x) / self.grid_res)
        grid_r = int((y - self.grid_origin_y) / self.grid_res)
        return grid_c, grid_r

    # --- Callbacks ---
    def odom_callback(self, msg: Odometry):
        self.current_pose = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            self.quaternion_to_yaw_from_msg(msg.pose.pose.orientation)
        ]

    def bev_map_callback(self, msg: PointCloud2):
        try:
            grid = np.zeros((self.cells_y, self.cells_x), dtype=np.uint8)
            for point in pc2.read_points(msg, field_names=('x', 'y'), skip_nans=True):
                c, r = self.world_to_grid_idx_numpy(point[0], point[1])
                if 0 <= r < self.cells_y and 0 <= c < self.cells_x:
                    grid[r, c] = 255
            inflated = cv2.dilate(grid, self.inflation_kernel)
            self.costmap_tensor = torch.from_numpy(inflated).to(self.device).float()
        except Exception as e:
            self.get_logger().error(f"BEV map error: {e}")

    # --- Visualization Publishers ---
    def publish_global_waypoints(self):
        """전체 글로벌 웨이포인트를 MarkerArray로 발행"""
        marker_array = MarkerArray()
        for i, (wx, wy, wyaw) in enumerate(self.waypoints):
            marker = Marker()
            marker.header.frame_id = "map" # 글로벌 좌표계 (odom 또는 map)
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "global_waypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = wx
            marker.pose.position.y = wy
            marker.pose.position.z = 0.0
            marker.scale.x = 0.5; marker.scale.y = 0.5; marker.scale.z = 0.5


            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 0 

            # 현재 목표는 초록색, 나머지는 파란색
            if i == self.waypoint_index:
                 marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            else:
                 marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.5)
            
            marker_array.markers.append(marker)
        self.pub_global_wps.publish(marker_array)

    def viz_publish_optimal(self, opt_traj_gpu):
        """최적 경로(nav_msgs/Path) 발행 (Robot Frame)"""
        path_msg = Path()
        path_msg.header.frame_id = self.robot_frame
        path_msg.header.stamp = self.get_clock().now().to_msg()

        opt_traj_cpu = opt_traj_gpu.cpu().numpy() # (T, 3) [x, y, yaw] or [x, y]
        for i in range(opt_traj_cpu.shape[0]):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(opt_traj_cpu[i, 0])
            pose.pose.position.y = float(opt_traj_cpu[i, 1])
            # Orientation은 시각화에 덜 중요하므로 identity로 둠
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.pub_opt_path.publish(path_msg)

    def viz_publish_samples(self, sampled_trajs_gpu):
        """샘플링된 경로들(Marker LINE_LIST) 발행 (Robot Frame)"""
        # sampled_trajs_gpu: (K_subset, T, state_dim)
        samples_cpu = sampled_trajs_gpu.cpu().numpy()
        K_sub, T, _ = samples_cpu.shape

        marker = Marker()
        marker.header.frame_id = self.robot_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "mppi_samples"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.05 # 선 두께
        marker.color = ColorRGBA(r=0.7, g=0.7, b=0.7, a=0.2) # 회색, 반투명
        marker.pose.orientation.w = 1.0



        points = []
        for k in range(K_sub):
            for t in range(T - 1):
                p1 = Point(x=float(samples_cpu[k, t, 0]), y=float(samples_cpu[k, t, 1]), z=0.0)
                p2 = Point(x=float(samples_cpu[k, t+1, 0]), y=float(samples_cpu[k, t+1, 1]), z=0.0)
                points.append(p1)
                points.append(p2)
        
        marker.points = points
        self.pub_samples.publish(marker)

    def viz_publish_goal(self, local_goal_tensor):
        """로컬 목표 지점(Marker SPHERE) 발행 (Robot Frame)"""
        goal_cpu = local_goal_tensor.cpu().numpy()
        
        marker = Marker()
        marker.header.frame_id = self.robot_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "local_goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(goal_cpu[0])
        marker.pose.position.y = float(goal_cpu[1])
        marker.pose.position.z = 0.0
        marker.scale.x = 0.3; marker.scale.y = 0.3; marker.scale.z = 0.3
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0) # 빨간색
        marker.pose.orientation.w = 1.0

        self.pub_local_goal.publish(marker)

    # --- Main Control Loop ---
    def stop_robot(self):
        self.cmd_pub.publish(Twist())
        self.controller.reset()

    def check_collision(self):
        if self.costmap_tensor is None: return False
        danger = self.costmap_tensor[self.roi_r_start:self.roi_r_end, self.roi_c_start:self.roi_c_end]
        return torch.any(danger >= self.collision_cost_th).item()

    def control_callback(self):
        if self.current_pose is None: return
        if self.current_task == "DONE": self.stop_robot(); return

        # 비상 정지 확인
        if self.check_collision():
            if not self.collision_detected_last_step:
                self.get_logger().warn("🛑 IMMINENT COLLISION!")
            self.stop_robot()
            self.collision_detected_last_step = True
            return
        self.collision_detected_last_step = False

        # 목표 계산
        cur_x, cur_y, cur_yaw = self.current_pose
        tx, ty, tyaw = self.waypoints[self.waypoint_index]
        dist = math.hypot(tx - cur_x, ty - cur_y)
        yaw_err = self.normalize_angle(tyaw - cur_yaw)

        # 상태 머신
        if self.current_task == "NAVIGATING":
            if dist < self.goal_threshold:
                self.get_logger().info(f"WP {self.waypoint_index} Reached. Aligning...")
                self.current_task = "ALIGNING"
                self.stop_robot()
                return

            # 로컬 목표 변환 및 MPPI 실행
            dx, dy = tx - cur_x, ty - cur_y
            lx = dx * math.cos(cur_yaw) + dy * math.sin(cur_yaw)
            ly = -dx * math.sin(cur_yaw) + dy * math.cos(cur_yaw)
            local_goal = torch.tensor([lx, ly], device=self.device)

            ctrl, opt_traj, samples = self.controller.run_mppi(local_goal, self.costmap_tensor)
            
            if ctrl:
                # ★ RViz 시각화 발행
                self.viz_publish_optimal(opt_traj)
                self.viz_publish_samples(samples)
                self.viz_publish_goal(local_goal)

                twist = Twist()
                twist.linear.x = ctrl[0]
                twist.angular.z = ctrl[1]
                self.cmd_pub.publish(twist)
            else:
                self.stop_robot()

        elif self.current_task == "ALIGNING":
            if abs(yaw_err) < self.yaw_threshold:
                self.get_logger().info("Aligned. Pausing...")
                self.current_task = "PAUSING"
                self.pause_start_time = self.get_clock().now()
                self.stop_robot()
                return
            
            w = np.clip(self.yaw_p_gain * yaw_err, -self.max_w, self.max_w)
            if abs(w) < self.min_align_w: w = math.copysign(self.min_align_w, w)
            twist = Twist(); twist.angular.z = w
            self.cmd_pub.publish(twist)

        elif self.current_task == "PAUSING":
            if (self.get_clock().now() - self.pause_start_time).nanoseconds / 1e9 >= 1.0:
                self.waypoint_index += 1
                if self.waypoint_index >= len(self.waypoints):
                    self.get_logger().info("🎉 All Done!")
                    self.current_task = "DONE"
                else:
                    self.get_logger().info(f"Moving to WP {self.waypoint_index}...")
                    self.current_task = "NAVIGATING"
            else:
                self.stop_robot()

def main(args=None):
    rclpy.init(args=args)
    node = MPPIBevPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
