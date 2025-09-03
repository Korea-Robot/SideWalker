#!/bin/bash

# ROS 2 환경에서 내비게이션 데이터 기록을 위한 스크립트
# 폴더 이름에 현재 날짜와 시간을 포함하여 생성합니다.
BAG_DIRECTORY="nav_bag_$(date +%Y-%m-%d_%H-%M-%S)"
echo "Recording to directory: $BAG_DIRECTORY"

ros2 bag record -o $BAG_DIRECTORY \
	/tf \
	/argus/ar0234_front_left/image_raw \
	/argus/ar0234_front_right/image_raw \
	/argus/ar0234_rear/image_raw \
	/argus/ar0234_side_left/image_raw \
	/argus/ar0234_side_right/image_raw \
	/camera/camera/color/image_raw \
	/camera/camera/depth/image_rect_raw \
	/mcu/state/jointURDF \
	/gx5/gnss1/fix \
	/gx5/gnss2/fix \
	/mcu/state/battery

echo "Recording stopped."
