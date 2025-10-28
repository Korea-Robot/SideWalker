#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import math

class MatplotlibVisualizer:
    """
    Matplotlib을 사용하여 로봇의 궤적, 경로, 장애물 등을 실시간 시각화
    """
    def __init__(self, node):
        """
        :param node: 데이터를 참조할 메인 ROS 노드 (PlannerNode)
        """
        self.node = node 
        self.fig, self.ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
        self._setup_plot()

    def _setup_plot(self):
        """ 플롯의 기본 설정 (축, 제목, 범례 등) """
        self.ax.set_title('Real-time A* BEV Planner', fontsize=14)
        self.ax.set_xlabel('-Y Position (m)')
        self.ax.set_ylabel('X Position (m)')
        self.ax.grid(True)
        self.ax.set_aspect('equal', adjustable='box')
        
        # 웨이포인트 기준으로 축 범위 설정
        wps_array = np.array(self.node.waypoints)
        x_min, y_min = wps_array.min(axis=0) - 1.5
        x_max, y_max = wps_array.max(axis=0) + 1.5
        self.ax.set_ylim(x_min, x_max)
        self.ax.set_xlim(-y_max, -y_min)
        
        # 플롯 요소들
        self.traj_line, = self.ax.plot([], [], 'b-', lw=2, label='Trajectory')
        self.current_point, = self.ax.plot([], [], 'go', markersize=10, label='Current Position')
        self.heading_line, = self.ax.plot([], [], 'g--', lw=2, label='Heading')
        self.waypoints_line, = self.ax.plot([], [], 'c.-', lw=2, label='Local Path (A*)')
        self.goal_point, = self.ax.plot([], [], 'm*', markersize=15, label='Lookahead Goal')
        self.reached_wps_plot, = self.ax.plot([], [], 'rx', markersize=10, mew=2, label='Reached Waypoints')
        self.pending_wps_plot, = self.ax.plot([], [], 'o', color='lime', markersize=10, mfc='none', mew=2, label='Pending Waypoints')
        self.obstacle_scatter = self.ax.scatter([], [], c='red', s=2, alpha=0.4, label='BEV Obstacles')
        
        self.ax.legend(loc='upper right', fontsize=9)

    def _update_plot(self, frame):
        """ FuncAnimation에 의해 호출되는 업데이트 함수 """
        
        # --- 1. 노드에서 데이터 복사 (Thread-safe) ---
        with self.node.plot_data_lock:
            traj = list(self.node.trajectory_data)
            pose = self.node.current_pose
            waypoints_local = self.node.latest_waypoints.copy()
            lookahead_local = self.node.latest_lookahead_point.copy()
            obstacles_local = self.node.obstacle_points.copy()
            all_wps = np.array(self.node.waypoints)
            wp_idx = self.node.waypoint_index

        if not traj or pose is None:
            return []

        current_x, current_y, current_yaw = pose

        # --- 2. 글로벌 웨이포인트 업데이트 ---
        reached_wps, pending_wps = all_wps[:wp_idx], all_wps[wp_idx:]
        if reached_wps.size > 0: self.reached_wps_plot.set_data(-reached_wps[:, 1], reached_wps[:, 0])
        else: self.reached_wps_plot.set_data([], [])
        if pending_wps.size > 0: self.pending_wps_plot.set_data(-pending_wps[:, 1], pending_wps[:, 0])
        else: self.pending_wps_plot.set_data([], [])

        # --- 3. 로봇 궤적 및 자세 업데이트 ---
        traj_arr = np.array(traj)
        self.traj_line.set_data(-traj_arr[:, 1], traj_arr[:, 0])
        self.current_point.set_data([-current_y], [current_x])
        heading_len = 0.5
        heading_end_x = current_x + heading_len * math.cos(current_yaw)
        heading_end_y = current_y + heading_len * math.sin(current_yaw)
        self.heading_line.set_data([-current_y, -heading_end_y], [current_x, heading_end_x])

        # --- 4. 로컬 플랜/장애물 -> 글로벌 변환 및 업데이트 ---
        if waypoints_local.size > 0 and lookahead_local.size > 0:
            rot_matrix = np.array([[math.cos(current_yaw), -math.sin(current_yaw)],
                                   [math.sin(current_yaw),  math.cos(current_yaw)]])
            
            # A* 경로 (로컬 -> 글로벌)
            waypoints_global = (rot_matrix @ waypoints_local.T).T + np.array([current_x, current_y])
            # Lookahead 지점 (로컬 -> 글로벌)
            lookahead_global = rot_matrix @ lookahead_local + np.array([current_x, current_y])
            
            # 장애물 포인트 (로컬 -> 글로벌)
            if obstacles_local.size > 0:
                obstacles_global = (rot_matrix @ obstacles_local.T).T + np.array([current_x, current_y])
                self.obstacle_scatter.set_offsets(np.c_[-obstacles_global[:, 1], obstacles_global[:, 0]])
            else:
                self.obstacle_scatter.set_offsets(np.empty((0, 2)))
            
            self.waypoints_line.set_data(-waypoints_global[:, 1], waypoints_global[:, 0])
            self.goal_point.set_data([-lookahead_global[1]], [lookahead_global[0]])

        # 업데이트할 플롯 요소 반환 (Blitting)
        return [self.traj_line, self.waypoints_line, self.current_point, 
                self.heading_line, self.goal_point, self.reached_wps_plot, 
                self.pending_wps_plot, self.obstacle_scatter]

    def run(self):
        """ Matplotlib 애니메이션 시작 (plt.show() 호출) """
        ani = FuncAnimation(
            self.fig, self._update_plot,
            interval=100,  # 100ms 마다 업데이트
            blit=True
        )
        try:
            plt.show()
        except Exception as e:
            print(f"Matplotlib visualizer stopped: {e}")
