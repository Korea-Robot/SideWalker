sudo snap ack snapd_24724.assert
sudo snap install snapd_24724.snap

docker exec -it krm_jazzy_0918 /bin/bash

// localization
ros2 launch krm_rko_localization odometry.launch.py \
    config_file:=/root/krm_ws/src/krm_rko_localization/ros/config/default.yaml \
    rviz:=true

// os sensor 
ros2 launch ouster_ros sensor.composite.launch.py viz:=False

check : ros2 topic list | grep oust

SLAM - dlio
ros2 launch direct_lidar_inertial_odometry dlio.launch.py rviz:=true pointcloud_topic:=/ouster/points imu_topic:=/ouster/imu

// save
ros2 service call /save_pcd direct_lidar_inertial_odometry/srv/SavePCD "{leaf_size: 0.05, save_path: '/root/krm_data'}"

// viewer cd krm_data
pcl_viewer dlio_map_20251023_094444.pcd 



docker run --name krm_jazzy_0918 -it --runtime=nvidia --gpus all --network=host --privileged -e DISPLAY=$DISPLAY --restart always -v /home/krm/krm_data:/root/krm_data -v /tmp/.X11-unix:/tmp/.X11-unix -p 6666:6666 -v /dev/shm:/dev/shm jazzy_sol_img:latest /bin/bash
