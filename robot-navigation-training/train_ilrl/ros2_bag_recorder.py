#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rosbag2_py
from rclpy.serialization import serialize_message
import os
import datetime

class ROS2BagRecorder(Node):
    def __init__(self):
        super().__init__('ros2_bag_recorder')
        
        # Bag writer 설정
        self.setup_bag_writer()
        
        # 구독자 설정
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw', 
            self.depth_callback, 10
        )
        self.rgb_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw',
            self.rgb_callback, 10
        )
        self.cmd_sub = self.create_subscription(
            Twist, '/mcu/command/manual_twist',
            self.cmd_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/command_odom',
            self.odom_callback, 10
        )
        
        self.get_logger().info("ROS2 Bag Recorder started. Recording topics...")
        
    def setup_bag_writer(self):
        # 현재 시간으로 bag 파일명 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bag_path = f"./bags/navigation_data_{timestamp}"
        
        # 디렉토리 생성
        os.makedirs(bag_path, exist_ok=True)
        
        # Bag writer 초기화
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        
        self.writer = rosbag2_py.SequentialWriter()
        self.writer.open(storage_options, converter_options)
        
        # 토픽 정보 생성
        topics = [
            ('/camera/camera/depth/image_rect_raw', 'sensor_msgs/msg/Image'),
            ('/camera/camera/color/image_raw', 'sensor_msgs/msg/Image'),
            ('/mcu/command/manual_twist', 'geometry_msgs/msg/Twist'),
            ('/command_odom', 'nav_msgs/msg/Odometry')
        ]
        
        for topic_name, topic_type in topics:
            topic_info = rosbag2_py.TopicMetadata(
                name=topic_name,
                type=topic_type,
                serialization_format='cdr'
            )
            self.writer.create_topic(topic_info)
            
        self.get_logger().info(f"Bag file created: {bag_path}")
    
    def depth_callback(self, msg):
        self.writer.write(
            '/camera/camera/depth/image_rect_raw',
            serialize_message(msg),
            self.get_clock().now().nanoseconds
        )
    
    def rgb_callback(self, msg):
        self.writer.write(
            '/camera/camera/color/image_raw',
            serialize_message(msg),
            self.get_clock().now().nanoseconds
        )
    
    def cmd_callback(self, msg):
        self.writer.write(
            '/mcu/command/manual_twist',
            serialize_message(msg),
            self.get_clock().now().nanoseconds
        )
    
    def odom_callback(self, msg):
        self.writer.write(
            '/command_odom',
            serialize_message(msg),
            self.get_clock().now().nanoseconds
        )
    
    def destroy_node(self):
        if hasattr(self, 'writer'):
            self.writer.close()
        self.get_logger().info("Bag recording stopped.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    recorder = ROS2BagRecorder()
    
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
