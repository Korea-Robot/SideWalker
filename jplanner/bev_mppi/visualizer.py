# visualizer.py

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import math


"""
역할: Matplotlib 시각화 설정(setup_visualization)과 업데이트 루프(update_plot)를 담당합니다.

ROS에 독립적이며, runner 노드 객체를 인자로 받아 plot_data_lock을 통해 데이터에 접근합니다.

"""


def update_plot(frame, node, ax, traj_line,
                current_point, heading_line, goal_point,
                reached_wps_plot, pending_wps_plot, obstacle_scatter,
                optimal_traj_line, sampled_traj_lines):
    """Matplotlib 애니메이션의 매 프레임마다 호출되는 함수"""
    
    # --- 1. Runner로부터 스레드 안전하게 데이터 복사 ---
    with node.plot_data_lock:
        traj = list(node.trajectory_data)
        pose = node.current_pose
        # MPPI 데이터 (로컬)
        optimal_traj_local = node.latest_optimal_trajectory_local.copy()
        sampled_trajs_local = node.latest_sampled_trajectories_local.copy()
        goal_local = node.latest_local_goal.copy()
        obstacles_local = node.obstacle_points_local.copy()
        # 글로벌 웨이포인트
        all_wps = np.array(node.waypoints)
        wp_idx = node.waypoint_index

    if not traj or pose is None:
        return []

    # --- 2. 글로벌 웨이포인트 업데이트 ---
    reached_wps, pending_wps = all_wps[:wp_idx], all_wps[wp_idx:]
    if reached_wps.size > 0:
        reached_wps_plot.set_data(-reached_wps[:, 1], reached_wps[:, 0])
    else:
        reached_wps_plot.set_data([], [])
    if pending_wps.size > 0:
        pending_wps_plot.set_data(-pending_wps[:, 1], pending_wps[:, 0])
    else:
        pending_wps_plot.set_data([], [])

    # --- 3. 로봇 궤적 및 자세 업데이트 ---
    traj_arr = np.array(traj)
    traj_line.set_data(-traj_arr[:, 1], traj_arr[:, 0]) # (X, Y) -> (-Y, X)로 플로팅

    current_x, current_y, current_yaw = pose
    current_point.set_data([-current_y], [current_x])
    heading_len = 0.5
    heading_end_x = current_x + heading_len * math.cos(current_yaw)
    heading_end_y = current_y + heading_len * math.sin(current_yaw)
    heading_line.set_data([-current_y, -heading_end_y], [current_x, heading_end_x])

    # --- 4. 로컬 -> 글로벌 좌표 변환 매트릭스 ---
    rot_matrix = np.array([[math.cos(current_yaw), -math.sin(current_yaw)],
                            [math.sin(current_yaw),  math.cos(current_yaw)]])
    
    # --- 5. 장애물, 로컬골, MPPI 궤적 플로팅 ---
    
    # 장애물 포인트 (로컬 -> 글로벌)
    if obstacles_local.size > 0:
        obstacles_global = (rot_matrix @ obstacles_local.T).T + np.array([current_x, current_y])
        obstacle_scatter.set_offsets(np.c_[-obstacles_global[:, 1], obstacles_global[:, 0]])
    else:
        obstacle_scatter.set_offsets(np.empty((0, 2)))
    
    # 로컬 골 (로컬 -> 글로벌)
    if goal_local.size > 0:
        goal_global = rot_matrix @ goal_local + np.array([current_x, current_y])
        goal_point.set_data([-goal_global[1]], [goal_global[0]])
    else:
        goal_point.set_data([], [])

    # 최적 궤적 (로컬 -> 글로벌)
    if optimal_traj_local.size > 0:
        optimal_traj_global = (rot_matrix @ optimal_traj_local[:, :2].T).T + np.array([current_x, current_y])
        optimal_traj_line.set_data(-optimal_traj_global[:, 1], optimal_traj_global[:, 0])
    else:
        optimal_traj_line.set_data([], [])

    # 샘플링된 궤적 다발 (로컬 -> 글로벌)
    if sampled_trajs_local.size > 0:
        for i, line in enumerate(sampled_traj_lines):
            if i < len(sampled_trajs_local):
                traj_local = sampled_trajs_local[i] # (T, 3)
                traj_global = (rot_matrix @ traj_local[:, :2].T).T + np.array([current_x, current_y])
                line.set_data(-traj_global[:, 1], traj_global[:, 0])
            else:
                line.set_data([], []) # 남는 라인 아티스트 클리어
    else:
        for line in sampled_traj_lines:
            line.set_data([], [])

    # 업데이트된 아티스트 리스트 반환 (blit=True를 위해)
    artists = [traj_line, current_point, heading_line, goal_point,
               reached_wps_plot, pending_wps_plot, obstacle_scatter, optimal_traj_line]
    artists.extend(sampled_traj_lines)
    
    return artists

def setup_visualization(node):
    """
    Matplotlib Figure와 Artist를 설정하고 애니메이션을 시작합니다.
    이 함수는 plt.show()로 인해 메인 스레드를 블로킹합니다.
    
    Args:
        node (MPPIBevPlanner): 데이터에 접근하기 위한 ROS 2 노드 인스턴스
    """
    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    ax.set_title('Real-time MPPI BEV Planner', fontsize=14)
    ax.set_xlabel('-Y Position (m)')
    ax.set_ylabel('X Position (m)')
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    
    # 웨이포인트 기준으로 뷰 범위 설정
    wps_array = np.array(node.waypoints)
    x_min, y_min = wps_array.min(axis=0) - 1.5
    x_max, y_max = wps_array.max(axis=0) + 1.5
    ax.set_ylim(x_min, x_max)
    ax.set_xlim(-y_max, -y_min)
    
    # --- 플롯 아티스트 생성 ---
    # (update_plot 함수에서 사용할 객체들을 미리 생성)
    traj_line, = ax.plot([], [], 'b-', lw=2, label='Trajectory')
    current_point, = ax.plot([], [], 'go', markersize=10, label='Current Position')
    heading_line, = ax.plot([], [], 'g--', lw=2, label='Heading')
    
    goal_point, = ax.plot([], [], 'm*', markersize=15, label='Local Goal')
    reached_wps_plot, = ax.plot([], [], 'rx', markersize=10, mew=2, label='Reached Waypoints')
    pending_wps_plot, = ax.plot([], [], 'o', color='lime', markersize=10, mfc='none', mew=2, label='Pending Waypoints')
    
    obstacle_scatter = ax.scatter([], [], c='red', s=2, alpha=0.4, label='BEV Obstacles')
    
    # --- MPPI 전용 아티스트 ---
    optimal_traj_line, = ax.plot([], [], 'm-', lw=2.5, zorder=10,
                                 label=f'Optimal Trajectory (U)')
    
    sampled_traj_lines = []
    for i in range(node.num_samples_to_plot):
        label = 'Sampled Trajectories (K)' if i == 0 else None
        line, = ax.plot([], [], 'c-', lw=0.5, alpha=0.2, zorder=5, label=label)
        sampled_traj_lines.append(line)
    
    ax.legend(loc='upper right', fontsize=9)
    
    # --- 애니메이션 시작 ---
    ani = FuncAnimation(
        fig, update_plot, 
        fargs=(node, ax, traj_line,
               current_point, heading_line, goal_point,
               reached_wps_plot, pending_wps_plot, obstacle_scatter,
               optimal_traj_line, sampled_traj_lines),
        interval=100, # 100ms (10Hz)
        blit=True
    )

    try:
        plt.show() # 메인 스레드 블로킹
    except Exception as e:
        node.get_logger().info(f"Matplotlib window closed: {e}")
