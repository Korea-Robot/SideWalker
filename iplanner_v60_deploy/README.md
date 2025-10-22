# iplanner + rko_lio_odometry(based on map)

rko_lio (localization)   : odometry estimation 

iplanner  (planner)      :  
segformer (segmentation) : 
halo      (reward estimation) : 








rclpy now
cv2 time 

ros2 topic hz /mcu/state/rs2/depth
ros2 topic hz /pcl2_jh


sudo snap ack snapd_24724.assert
sudo snap install snapd_24724.snap




# xhost display setting
xhost +local:docker


docker exec -it krm_jazzy_0918 /bin/bash

// localization

ros2 launch krm_rko_localization odometry.launch.py \
    config_file:=/root/krm_ws/src/krm_rko_localization/ros/config/default.yaml \
    rviz:=true

// os sensor 
termianl 1 : runner // server
terminal 2 : ros2 service call /argus_mission_runner/activate_sensor_ouster_lidar std_srvs/srv/Empty  // ouster 

check : ros2 topic list | grep oust



cker run --name krm_jazzy_0918 -it --runtime=nvidia --gpus all --network=host --privileged -e DISPLAY=$DISPLAY --restart always -v /home/krm/krm_data:/root/krm_data -v /tmp/.X11-unix:/tmp/.X11-unix -p 6666:6666 -v /dev/shm:/dev/shm jazzy_sol_img:latest /bin/bash


start point

    position:
      x: 8.442714461178333
      y: -12.599239470947854
      z: 0.38310569112421133
    orientation:
      x: 0.003065831716701782
      y: 0.0012753685666226243
      z: -0.992085823485534
      w: 0.12551769974762209
  covariance: '<array type: double[36]>'
twist:
  twist:
    linear:
      x: 0.4269209359248972
      y: -0.04189631523303917
      z: 0.24233376448096425
    angular:
      x: 0.03426362666377817
      y: -0.022886597202030904
      z: -0.008059704799043627
  covariance: '<array type: double[36]>'
---


