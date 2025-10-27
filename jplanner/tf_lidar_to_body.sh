# ros2 run tf2_ros static_transform_publisher [x y z] [yaw pitch roll] [parent_frame] [child_frame]
# 참고: ROS2의 static_transform_publisher는 [x y z] [yaw pitch roll] 순서를 사용합니다! (Roll, Pitch, Yaw 순서가 아님)

ros2 run tf2_ros static_transform_publisher -0.3 0 0 3.14159 0 0 body_lidar body
