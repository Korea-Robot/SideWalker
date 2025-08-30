#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
from erfpspnet import Net  # erfpspnet.py is required
from collections import OrderedDict

class PSPNetNode(Node):
    def __init__(self):
        super().__init__('pspsp_net_viewer_node')
        
        # 1) Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # 2) Load the custom PSPNet model
        try:
            self.model = Net(22)  # Adjust to the number of classes
            state_dict = torch.load("./model_best.pth", map_location=self.device)
            # Remove 'module.' prefix if the model was trained with DataParallel
            new_state = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state[name] = v
            self.model.load_state_dict(new_state)
            self.model.to(self.device).eval()
            self.get_logger().info("Model loaded successfully.")
        except FileNotFoundError:
            self.get_logger().error("Error: model_best.pth or erfpspnet.py not found.")
            self.get_logger().error("Please ensure the model file and definition file are in the same directory.")
            rclpy.shutdown()
            return
        except Exception as e:
            self.get_logger().error(f"An error occurred while loading the model: {e}")
            rclpy.shutdown()
            return

        # 3) Define the class color palette
        palette_base = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1], dtype=torch.int64)
        self.colors = (torch.arange(256, dtype=torch.int64)[:, None] * palette_base) % 255
        self.colors = self.colors.numpy().astype('uint8')  # RGB palette

        # 4) ROS 2 setup
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # Change this to your input image topic
            self.image_callback,
            10)
        
        self.get_logger().info('PSPNet viewer node has been started. Waiting for images...')

    def image_callback(self, msg):
        """Callback function to process and display the segmented image."""
        try:
            # Convert ROS Image message to an OpenCV image (BGR format)
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert ROS image: {e}")
            return

        # 5) Preprocessing: BGR -> RGB, numpy -> tensor, normalization
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor_img = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True).float() / 255.0

        # 6) Inference
        with torch.no_grad():
            logits = self.model(tensor_img)

        # 7) Create the prediction mask
        preds = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)  # HxW

        # 8) Colorize the mask and overlay it on the original image
        mask_rgb = self.colors[preds]
        overlay_rgb = cv2.addWeighted(img_rgb, 0.5, mask_rgb, 0.5, 0)
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV display

        # 9) Display results: show original and segmented images side-by-side
        combined_display = np.hstack((img_bgr, overlay_bgr))
        cv2.imshow("Original | PSPNet Segmentation", combined_display)

        # Check for key press to close the window
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press ESC to exit
            self.get_logger().info("ESC key pressed. Shutting down.")
            # This will cause rclpy.spin() to return
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    pspnet_node = PSPNetNode()
    
    # Use a try-finally block to ensure cv2 windows are closed
    try:
        rclpy.spin(pspnet_node)
    finally:
        # Cleanup
        pspnet_node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
