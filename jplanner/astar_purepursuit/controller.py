#!/usr/bin/env python3

import numpy as np
import math
from geometry_msgs.msg import Twist

# path following & twist  
# not see all planning, just look ahead dist point  
# calculate lookahead point  angle 

class PurePursuitController:
    """
    경로를 따라가는 제어기 (Pure Pursuit과 유사)
    """
    def __init__(self, max_linear, max_angular, look_ahead, turn_damping, angular_gain):
        self.max_linear_velocity = max_linear
        self.max_angular_velocity = max_angular
        self.look_ahead_dist = look_ahead
        self.turn_damping_factor = turn_damping
        self.angular_gain = angular_gain

    def calculate_command(self, path_world_np):
        """
        주어진 로컬 경로(path_world_np)를 따라가는 Twist 명령을 계산합니다.
        경로는 로봇 기준 좌표계 (x 전방, y 좌측)입니다.
        
        :param path_world_np: (N, 2) numpy array. 로봇 기준 (x, y) 좌표.
        :return: (Twist, lookahead_point)
        """
        
        if path_world_np is None or len(path_world_np) == 0:
            return Twist(), np.array([0.0, 0.0])

        # 1. Lookahead 지점 찾기
        # 로봇(0,0)으로부터 path_world_np 상의 점들까지의 거리 계산
        distances = np.linalg.norm(path_world_np, axis=1)
        
        # look_ahead_dist보다 멀리 있는 첫 번째 점 찾기
        lookahead_idx = np.argmax(distances >= self.look_ahead_dist)
        
        if lookahead_idx == 0: # 경로가 lookahead 거리보다 짧은 경우
            # 그냥 마지막 점을 사용
            lookahead_idx = len(path_world_np) - 1
        
        la_x, la_y = path_world_np[lookahead_idx]
        lookahead_point = np.array([la_x, la_y])


        # 2. 제어 명령 계산
        # 로봇 전방 기준 lookahead 지점까지의 각도
        target_angle = math.atan2(la_y, la_x)
        
        # P 제어기 (각도 오차에 비례)
        angular_z = self.angular_gain * target_angle
        angular_z = np.clip(angular_z, -self.max_angular_velocity, self.max_angular_velocity)
        
        if abs(angular_z) < 0.1:
            angular_z = 0.0

        # 각도가 클수록 속도 감속
        linear_x = self.max_linear_velocity / (1.0 + self.turn_damping_factor * abs(angular_z))
        
        # 3. Twist 메시지 생성
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        
        return twist, lookahead_point
