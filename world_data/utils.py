# utils.py

import numpy as np 

# --- 유틸리티 함수 ---
def convert_to_egocentric(global_target_pos, agent_pos, agent_heading):
    """
    월드 좌표계의 목표 지점을 에이전트 중심의 자기(egocentric) 좌표계로 변환합니다.

    :param global_target_pos: 월드 좌표계에서의 목표 지점 [x, y]
    :param agent_pos: 월드 좌표계에서의 에이전트 위치 [x, y]
    :param agent_heading: 에이전트의 현재 진행 방향 (라디안)
    :return: 에이전트 기준 상대 위치 [x, y]. x: 좌/우, y: 전/후
    """
    # 1. 월드 좌표계에서 에이전트로부터 목표 지점까지의 벡터 계산
    vec_in_world = global_target_pos - agent_pos

    # 2. 에이전트의 heading의 "음수" 각도를 사용하여 회전 변환
    # 월드 좌표계에서 에이전트 좌표계로 바꾸려면, 에이전트의 heading만큼 반대로 회전해야 함
    theta = -agent_heading
    cos_h = np.cos(theta)
    sin_h = np.sin(theta)
    
    rotation_matrix = np.array([
        [cos_h, -sin_h],
        [sin_h,  cos_h]
    ])

    # 3. 회전 행렬을 적용하여 에이전트 중심 좌표계의 벡터를 얻음
    ego_vector = rotation_matrix @ vec_in_world
    
    return ego_vector




######################## PD Controller #############################

import time 

class PD_Controller:
    def __init__(self,kp=0.3,kd=0.1,min_dt=0.1):
        """
        PID 제어기 초기화
            kp: 비례 상수
            ki: 적분 상수
            kd: 미분 상수
            setpoint: 목표치
            output_limit: 제어 출력의 최대/최소 한계 (anti-windup 적용)
            min_dt: 최소 시간 간격 (너무 작은 dt로 인한 미분 항 폭주 방지)
        """
        self.kp = kp 
        self.kd = kd 
        self.min_dt = min_dt
        self.last_error = 0.0 
        self.last_time = time.time()
    
    def update(self,measurement):
        """
        측정값(현재 오차)을 기반으로 제어 신호를 계산하고 상태를 업데이트합니다.
        :param measurement: 제어할 값 (에이전트 중심 좌표계에서의 목표 지점 y값, 즉 횡방향 오차)
        :return: 제어 신호 (조향값)
        """
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt < self.min_dt:
            # derivative explode 방지
            dt = self.min_dt
        
        error = measurement # goal position of y (~=yaw)
        
        derivative = (error-self.last_error)/(dt+1e-9)
        
        pd_control = self.kp * error + self.kd * derivative
        
        print(' derivative',derivative)
        self.last_error = error 
        self.last_time = current_time
        
        # min max cut 
        pd_control = min(max(-1,pd_control),1)

        default_throttle =0.4
        
        return [pd_control,default_throttle]