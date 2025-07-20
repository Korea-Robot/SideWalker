import numpy as np 


def compute_reward(ego_goal_position, info, last_distance_to_goal):
    """
    사용자 정의 보상 함수. 목표 지점과의 거리, 충돌, 경로 이탈 여부를 종합하여 보상을 계산합니다.

    :param ego_goal_position: 에이전트 중심 좌표계의 목표 위치
    :param info: env.step()에서 반환된 정보 딕셔너리
    :param last_distance_to_goal: 이전 스텝에서의 목표까지의 거리
    :return: (계산된 커스텀 보상, 현재 목표까지의 거리)
    """
    # 1. 목표 지점까지의 거리 기반 보상
    current_distance = np.linalg.norm(ego_goal_position)
    # 이전 스텝보다 가까워졌으면 양수 보상, 멀어졌으면 음수 보상(페널티)
    distance_reward = (last_distance_to_goal - current_distance) * 10.0 # 스케일링 팩터
    
    reward = distance_reward

    # 2. 목표 도착 보상
    if current_distance < 0.5:  # 목표 지점 0.5m 이내로 들어오면 큰 보상
        reward += 50.0

    # 3. 충돌/사고 페널티
    # info 딕셔너리에서 충돌 관련 키들을 확인하여 하나라도 True이면 페널티 부과
    is_crash = info.get('crash_vehicle', False) or \
               info.get('crash_object', False) or \
               info.get('crash_building', False) or \
               info.get('crash_sidewalk', False)
    if is_crash:
        reward -= 10.0

    # 4. 경로(인도) 이탈 페널티
    if info.get('out_of_road', False):
        reward -= 5.0
        
    return reward, current_distance