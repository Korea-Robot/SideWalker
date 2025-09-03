#!/usr/bin/env python3

import os
import numpy as np
import cv2
import torch
import pickle
from pathlib import Path
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import json
from tqdm import tqdm

class BagToDataset:
    def __init__(self, bag_path, output_path):
        self.bag_path = bag_path
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.bridge = CvBridge()
        self.data_samples = []
        
        # 데이터 동기화를 위한 버퍼
        self.depth_buffer = {}
        self.rgb_buffer = {}
        self.cmd_buffer = {}
        self.odom_buffer = {}
        
    def read_bag(self):
        """Bag 파일에서 데이터 읽기"""
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}
        
        print("Reading bag file...")
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            
            if topic == '/camera/camera/depth/image_rect_raw':
                msg = deserialize_message(data, Image)
                self.depth_buffer[timestamp] = self.process_depth_image(msg)
                
            elif topic == '/camera/camera/color/image_raw':
                msg = deserialize_message(data, Image)
                self.rgb_buffer[timestamp] = self.process_rgb_image(msg)
                
            elif topic == '/mcu/command/manual_twist':
                msg = deserialize_message(data, Twist)
                self.cmd_buffer[timestamp] = [msg.linear.x, msg.angular.z]
                
            elif topic == '/command_odom':
                msg = deserialize_message(data, Odometry)
                self.odom_buffer[timestamp] = self.process_odom(msg)
        
        reader.close()
        print(f"Read {len(self.depth_buffer)} depth images, {len(self.rgb_buffer)} RGB images")
        
    def process_depth_image(self, msg):
        """Depth 이미지 처리"""
        depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        max_depth_value = 10.0
        depth_cv = (np.clip(depth_cv, 0, max_depth_value*1000) / 1000.0).astype(np.float32)
        depth_cv[depth_cv > max_depth_value] = 0
        return depth_cv
    
    def process_rgb_image(self, msg):
        """RGB 이미지 처리"""
        rgb_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        return cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)
    
    def process_odom(self, msg):
        """Odometry 데이터 처리"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        # Quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return [x, y, yaw]
    
    def synchronize_data(self, time_threshold=50000000):  # 50ms
        """시간 기준으로 데이터 동기화"""
        print("Synchronizing data...")
        
        # 모든 타임스탬프 수집
        all_timestamps = set()
        all_timestamps.update(self.depth_buffer.keys())
        all_timestamps.update(self.rgb_buffer.keys())
        all_timestamps.update(self.cmd_buffer.keys())
        all_timestamps.update(self.odom_buffer.keys())
        
        synchronized_data = []
        
        for timestamp in tqdm(sorted(all_timestamps)):
            # 각 타입별로 가장 가까운 시간의 데이터 찾기
            depth_data = self.find_closest_data(self.depth_buffer, timestamp, time_threshold)
            rgb_data = self.find_closest_data(self.rgb_buffer, timestamp, time_threshold)
            cmd_data = self.find_closest_data(self.cmd_buffer, timestamp, time_threshold)
            odom_data = self.find_closest_data(self.odom_buffer, timestamp, time_threshold)
            
            # 모든 데이터가 있는 경우만 저장
            if all([depth_data is not None, rgb_data is not None, 
                   cmd_data is not None, odom_data is not None]):
                synchronized_data.append({
                    'timestamp': timestamp,
                    'depth': depth_data,
                    'rgb': rgb_data,
                    'cmd': cmd_data,
                    'odom': odom_data
                })
        
        print(f"Synchronized {len(synchronized_data)} samples")
        return synchronized_data
    
    def find_closest_data(self, buffer, target_timestamp, threshold):
        """가장 가까운 시간의 데이터 찾기"""
        if not buffer:
            return None
            
        closest_timestamp = min(buffer.keys(), 
                              key=lambda x: abs(x - target_timestamp))
        
        if abs(closest_timestamp - target_timestamp) <= threshold:
            return buffer[closest_timestamp]
        return None
    
    def create_dataset(self):
        """데이터셋 생성"""
        self.read_bag()
        synchronized_data = self.synchronize_data()
        
        if not synchronized_data:
            print("No synchronized data found!")
            return
        
        # 데이터 저장
        dataset = {
            'depth_images': [],
            'rgb_images': [],
            'commands': [],
            'odometry': [],
            'timestamps': []
        }
        
        print("Creating dataset...")
        for i, sample in enumerate(tqdm(synchronized_data)):
            # 이미지 파일로 저장
            depth_path = self.output_path / f"depth_{i:06d}.npy"
            rgb_path = self.output_path / f"rgb_{i:06d}.jpg"
            
            np.save(depth_path, sample['depth'])
            cv2.imwrite(str(rgb_path), cv2.cvtColor(sample['rgb'], cv2.COLOR_RGB2BGR))
            
            dataset['depth_images'].append(str(depth_path))
            dataset['rgb_images'].append(str(rgb_path))
            dataset['commands'].append(sample['cmd'])
            dataset['odometry'].append(sample['odom'])
            dataset['timestamps'].append(sample['timestamp'])
        
        # 메타데이터 저장
        metadata_path = self.output_path / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Dataset created with {len(dataset['commands'])} samples")
        print(f"Saved to: {self.output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_path', required=True, help='Path to bag file')
    parser.add_argument('--output_path', required=True, help='Output dataset path')
    args = parser.parse_args()
    
    converter = BagToDataset(args.bag_path, args.output_path)
    converter.create_dataset()

if __name__ == '__main__':
    main()
