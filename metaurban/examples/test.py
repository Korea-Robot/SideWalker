def quaternion_to_yaw(self, q):
    """쿼터니언을 Yaw 각도로 변환하는 헬퍼 함수"""
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def odom_callback(self, msg: Odometry):
    """Odometry 메시지를 받아 현재 위치와 방향(yaw)을 업데이트합니다."""
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    orientation_q = msg.pose.pose.orientation
    yaw = self.quaternion_to_yaw(orientation_q)
    
    self.current_pose = [x, y, yaw]
    # self.get_logger().info(f"Odom updated: x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.2f}")


def control_callback(self):
    """Main control loop for inference and command publishing."""
    # Odometry 데이터가 아직 수신되지 않았거나, 카메라 이미지가 없으면 실행하지 않음
    if self.current_depth_tensor is None or self.current_pose is None:
        return
        
    try:
        # 1. 현재 목표 지점(Global Goal) 설정
        if self.waypoint_index >= len(self.waypoints):
            self.get_logger().info("All waypoints reached. Stopping.")
            self.cmd_pub.publish(Twist()) # 모든 목표 도달 시 정지
            return
            
        target_wp = self.waypoints[self.waypoint_index]
        current_x, current_y, current_yaw = self.current_pose

        # 2. 목표 지점 도착 여부 확인
        distance_to_goal = math.sqrt((target_wp[0] - current_x)**2 + (target_wp[1] - current_y)**2)
        
        if distance_to_goal < self.goal_threshold:
            self.get_logger().info(f"Waypoint {self.waypoint_index} reached!")
            self.waypoint_index += 1
            return # 다음 루프에서 새로운 목표로 계산 시작

        # 3. Global Goal -> Local Goal 변환 (가장 중요한 부분)
        # 전역 좌표계에서의 목표와 내 위치의 차이 계산
        dx_global = target_wp[0] - current_x
        dy_global = target_wp[1] - current_y
        
        # 로봇의 현재 방향(yaw)을 기준으로 좌표계 회전
        local_x = dx_global * math.cos(current_yaw) + dy_global * math.sin(current_yaw)
        local_y = -dx_global * math.sin(current_yaw) + dy_global * math.cos(current_yaw)
        
        # AI 모델에 입력할 로컬 목표 텐서 생성
        local_goal_tensor = torch.tensor([local_x, local_y, 0.0], dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # PlannerNet 추론 시 고정된 self.goal 대신 변환된 local_goal_tensor 사용
            preds, fear = self.net(self.current_depth_tensor, local_goal_tensor)
            
            # ... 나머지 추론, 경로 생성, 속도 계산, 시각화, 명령 발행 코드는 거의 동일 ...
            waypoints = self.traj_cost.opt.TrajGeneratorFromPFreeRot(preds, step=0.1)
            cmd_vels = self.waypoints_to_cmd_vel(waypoints)
            
            linear_x = torch.clamp(cmd_vels[0, 0, 0], -1.0, 1.0).item()
            angular_z = torch.clamp(cmd_vels[0, 0, 1], -1.0, 1.0).item()
            fear_val = fear.cpu().item()

            # ... 안전 로직 ...

        # ... 시각화 업데이트 및 Twist 메시지 발행 ...
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_pub.publish(twist)
        
        self.get_logger().info(f"Target WP[{self.waypoint_index}]:({target_wp[0]:.1f}, {target_wp[1]:.1f}) | Local Goal:({local_x:.2f}, {local_y:.2f}) | Cmd: v={linear_x:.2f}, w={angular_z:.2f}")

    except Exception as e:
        self.get_logger().error(f"Control loop error: {e}\n{traceback.format_exc()}")
        self.cmd_pub.publish(Twist())