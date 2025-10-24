import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt

class RealSenseViewer(Node):
    def __init__(self):
        super().__init__('realsense_viewer')
        self.bridge = CvBridge()

        qos_profile = rclpy.qos.QoSProfile(depth=10)

        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.rgb_callback,
            qos_profile
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            qos_profile
        )

        self.rgb_image = None
        self.depth_image = None

    def rgb_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.show_images()

    def depth_callback(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        # boolean indexing
        depth[depth>16000] = 16000
        # Normalize for display
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        self.depth_image = cv2.convertScaleAbs(depth_normalized)
        self.show_images()
        self.image_statistics(depth)

    @staticmethod
    def image_statistics(image):
        # Only include valid depth range
        # valid_pixels = image[(image > 0) & (image < 65535)].flatten()
        valid_pixels= image.flatten()
        if valid_pixels.size == 0:
            print("No valid depth pixels.")
            return
        plt.clf()  # Clear the current figure
        plt.hist(valid_pixels, bins=100, range=(0, 20000), color='blue', alpha=0.7)
        plt.title("Depth Value Histogram")
        plt.xlabel("Depth value")
        plt.ylabel("Frequency")
        plt.pause(0.001)  # Pause for a short moment to update the plot

    def show_images(self):
        if self.rgb_image is not None and self.depth_image is not None:
            cv2.imshow('RGB Image', self.rgb_image)
            cv2.imshow('Depth Image', self.depth_image)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    plt.ion()  # Turn on interactive mode for matplotlib
    plt.figure(figsize=(8, 4))
    viewer = RealSenseViewer()
    try:
        rclpy.spin(viewer)
    except KeyboardInterrupt:
        pass
    viewer.destroy_node()
    cv2.destroyAllWindows()
    plt.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
