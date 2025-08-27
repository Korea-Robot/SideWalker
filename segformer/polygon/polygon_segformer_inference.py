#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ROS2
import rclpy
from rclpy.node import Node

# Common
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from torchvision import transforms

# --- Configuration Variables ---
# 1. Realsense camera image topic name (Modify this part)
REALSENSE_TOPIC = "/camera/camera/color/image_raw"

# 2. Trained model weights file path
MODEL_PATH = "ckpts/best_seg_model.pth"
# In your polygon_segformer_inference.py file
MODEL_PATH = "ckpts/best_seg_model.safetensors"

# 3. Device setting for inference
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Class information from training script ---
CLASS_TO_IDX = {
    'background': 0, 'barricade': 1, 'bench': 2, 'bicycle': 3, 'bollard': 4,
    'bus': 5, 'car': 6, 'carrier': 7, 'cat': 8, 'chair': 9, 'dog': 10,
    'fire_hydrant': 11, 'kiosk': 12, 'motorcycle': 13, 'movable_signage': 14,
    'parking_meter': 15, 'person': 16, 'pole': 17, 'potted_plant': 18,
    'power_controller': 19, 'scooter': 20, 'stop': 21, 'stroller': 22, 'table': 23,
    'traffic_light': 24, 'traffic_light_controller': 25, 'traffic_sign': 26,
    'tree_trunk': 27, 'truck': 28, 'wheelchair': 29
}
NUM_LABELS = len(CLASS_TO_IDX)

def create_color_palette():
    """Create a unique color palette for each class (OpenCV BGR format)"""
    palette = np.zeros((NUM_LABELS, 3), dtype=np.uint8)
    for i in range(NUM_LABELS):
        if i == 0:  # background
            palette[i] = [0, 0, 0]
            continue
        # Generate colors in HSV color space and convert to BGR
        hue = int(i * (180 / (NUM_LABELS - 1)))
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)
        palette[i] = color_bgr[0, 0, :]
    return palette

# --- ROS2 Node Class ---
# Inherits from rclpy.node.Node
class SemanticInferenceNode(Node):
    def __init__(self):
        """Initialize ROS2 node and model"""
        # --- ROS2 Change: Node initialization ---
        super().__init__('realsense_inference_node')
        self.get_logger().info("Realsense Inference node has started.")

        self.bridge = CvBridge()
        self.color_palette = create_color_palette()

        # Load model
        self.model = self.load_model()
        self.get_logger().info(f"Model loaded from '{MODEL_PATH}'. Running inference on {DEVICE}.")

        # Image preprocessor
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # --- ROS2 Change: Subscriber creation ---
        # Use create_subscription() instead of rospy.Subscriber
        # QoS (Quality of Service) profile is added for message handling reliability
        self.image_sub = self.create_subscription(
            Image,
            REALSENSE_TOPIC,
            self.image_callback,
            10  # QoS profile depth
        )

    def load_model(self):
        """Load a pre-trained SegFormer model and apply the state_dict"""
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()  # Set to inference mode
        return model

    def preprocess_image(self, cv_image):
        """Convert an OpenCV image to a model input tensor"""
        # BGR -> RGB conversion
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Convert to tensor and normalize
        input_tensor = self.transform(rgb_image)

        # Add batch dimension and send to device
        return input_tensor.unsqueeze(0).to(DEVICE)

    def image_callback(self, msg):
        """Callback function to receive image topics and perform inference and visualization"""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            original_h, original_w, _ = cv_image.shape
        except CvBridgeError as e:
            # --- ROS2 Change: Logging ---
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # 1. Preprocess image
        input_tensor = self.preprocess_image(cv_image)

        # 2. Perform inference (disable gradient calculation)
        with torch.no_grad():
            outputs = self.model(pixel_values=input_tensor)
            logits = outputs.logits

        # 3. Post-process results
        # Upsample logits to original image size
        upsampled_logits = F.interpolate(
            logits,
            size=(original_h, original_w),
            mode='bilinear',
            align_corners=False
        )
        # Create prediction mask from the class with the highest probability
        pred_mask = torch.argmax(upsampled_logits, dim=1).squeeze()

        # Convert to NumPy array
        pred_mask_np = pred_mask.cpu().numpy().astype(np.uint8)

        # 4. Visualization
        # Colorize the segmentation mask
        segmentation_image = self.color_palette[pred_mask_np]

        # Overlay the mask on the original image
        overlay_image = cv2.addWeighted(cv_image, 0.6, segmentation_image, 0.4, 0)

        # 5. Display results on screen
        cv2.imshow("Segment-only Result", segmentation_image)
        cv2.imshow("Overlay Result", overlay_image)

        # Wait 1ms for the window to update
        cv2.waitKey(1)

def main(args=None):
    # --- ROS2 Change: Main execution block ---
    rclpy.init(args=args)
    
    node = SemanticInferenceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt, shutting down.")
    finally:
        # Destroy the node explicitly
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        print("Node shutdown complete, all windows closed.")

if __name__ == '__main__':
    main()
