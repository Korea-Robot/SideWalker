#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

class SegformerViewerNode(Node):
    def __init__(self):
        super().__init__('segformer_viewer_node')
        
        # 1) Set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # 2) Load SegFormer model from a checkpoint
        checkpoint = "smp-hub/segformer-b2-1024x1024-city-160k"
        self.model = smp.Segformer.from_pretrained(checkpoint).eval().to(self.device)

        # 3) Image preprocessing pipeline
        img_size = 512
        self.preprocessing = A.Compose([
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        # 4) Create a color palette
        palette_base = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1], dtype=torch.int64)
        self.colors = (torch.arange(256, dtype=torch.int64)[:, None] * palette_base) % 255
        self.colors = self.colors.numpy().astype('uint8')  # RGB palette

        # ROS 2 setup
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # Change this to your input image topic
            self.image_callback,
            10)
        
        self.get_logger().info('Segformer viewer node has been started. Waiting for images...')

    def image_callback(self, msg):
        """Callback function for processing image messages and displaying the result."""
        try:
            # Convert ROS Image message to an OpenCV image
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert ROS image: {e}")
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 6) Preprocessing
        augmented = self.preprocessing(image=img_rgb)
        tensor_img = augmented['image'].unsqueeze(0).to(self.device)  # 1xCxHxW

        # 7) Inference and upsampling
        with torch.no_grad():
            logits = self.model(tensor_img)
        if isinstance(logits, dict):
            logits = logits['out']
        logits = F.interpolate(logits, size=img_rgb.shape[:2], mode='bilinear', align_corners=False)
        preds = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)  # HxW

        # 8) Colorize the mask and overlay it on the image
        mask_rgb = self.colors[preds]
        mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.5, mask_bgr, 0.5, 0)
        
        # Stack the original and segmented images side-by-side for comparison
        images_to_show = np.hstack((img_bgr, overlay))

        # 9) Display the images
        cv2.imshow("Segformer Segmentation Viewer", images_to_show)
        
        # Check for key press to close the window
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press ESC to exit
            rclpy.shutdown()
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    segformer_viewer_node = SegformerViewerNode()
    
    # Use a try-finally block to ensure cv2 windows are closed
    try:
        rclpy.spin(segformer_viewer_node)
    finally:
        # Cleanup
        segformer_viewer_node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
