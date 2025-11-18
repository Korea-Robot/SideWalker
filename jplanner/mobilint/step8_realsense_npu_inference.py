#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
import cv2
import argparse
import time
import threading
import queue
import traceback
from pycocotools import mask as maskUtils
from typing import Union, List, Tuple


# Numba JIT
from numba import njit, uint8

import maccel
from mblt_infer_original.helper import YoloHelper

# numba accel function
# call by ref
@njit(fastmath=True, cache=True, nogil=True)
def apply_mask_numba(image, mask, color, alpha=0.4):
    """
    image: (H, W, 3) uint8 배열 (원본 이미지)
    mask: (H, W) boolean/uint8 배열 (마스크)
    color: (3,) uint8 배열 (RGB 색상)
    alpha: 투명도
    """  
    h, w = mask.shape
    inv_alpha = 1.0 - alpha
    
    for i in range(h):
        for j in range(w):
            if mask[i, j]:
                b_val = color[0]
                g_val = color[1]
                r_val = color[2]
                
                image[i, j, 0] = uint8(image[i, j, 0] * inv_alpha + b_val * alpha)
                image[i, j, 1] = uint8(image[i, j, 1] * inv_alpha + g_val * alpha)
                image[i, j, 2] = uint8(image[i, j, 2] * inv_alpha + r_val * alpha)

# --- 설정 ---
parser = argparse.ArgumentParser()
parser.add_argument("--base_path", type=str, default=".")
args = parser.parse_args()

REALSENSE_TOPIC = "/camera/camera/color/image_raw"
CONF_THRES = 0.55
IOU_THRES = 0.45

VIS_SCALE =1 # 0.5 

# --- 모델 로딩 ---
print('Model loading...')
model_path = args.base_path + "/yolov9c-seg.mxq"
acc1 = maccel.Accelerator()
mc1 = maccel.ModelConfig()
mc1.set_global8_core_mode()
mxq_model1 = maccel.Model(model_path, mc1)
mxq_model1.launch(acc1)

def yolov9c_seg_helper(
    img_size: Union[Tuple[int], List[int]] = None,
    conf_thres: float = None,
    iou_thres: float = None,
    device: str = None,
):    
    helper = YoloHelper.make_from_yaml("./mblt_infer_original/model_configs/yolov9c_seg_640.yaml", device)
    helper.set_inference_param(img_size=img_size, conf_thres=conf_thres, iou_thres=iou_thres)
    return helper

helper = yolov9c_seg_helper(conf_thres=CONF_THRES, iou_thres=IOU_THRES,device="aries")

# COCO 클래스 정의
COCO_CLASS_TO_IDX = {
    0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 12: 'street sign', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    26: 'hat', 27: 'backpack', 28: 'umbrella', 29: 'shoe', 30: 'eye glasses',
    31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis',
    36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 45: 'plate',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
    66: 'mirror', 67: 'dining table', 68: 'window', 69: 'desk', 70: 'toilet',
    71: 'door', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
    76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster',
    81: 'sink', 82: 'refrigerator', 83: 'blender', 84: 'book', 85: 'clock',
    86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush',
    91: 'hair brush'
}
NUM_LABELS = len(COCO_CLASS_TO_IDX)


class NPU_Node(Node):
    def __init__(self):
        super().__init__('npu_segmentation_optimized_node')
        
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(NUM_LABELS, 3), dtype=np.uint8)
        self.colors[0] = [0, 0, 0]
        self.bridge = CvBridge()

        # 3 Queue
        self.infer_queue = queue.Queue(maxsize=1)
        self.vis_queue = queue.Queue(maxsize=1)
        self.display_queue = queue.Queue(maxsize=1)

        self.running = True

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, # must be
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.subscription = self.create_subscription(
            Image, REALSENSE_TOPIC, self.image_callback, qos_profile)

        # worker thread
        self.thread_infer = threading.Thread(target=self.thread_inference_worker, daemon=True)
        self.thread_vis = threading.Thread(target=self.thread_visualization_worker, daemon=True)
        
        self.thread_infer.start()
        self.thread_vis.start()

        self.get_logger().info('Hybrid Optimized Node Started.')

    def image_callback(self, msg):
        """
        별도 쓰레드(Spin)에서 호출되므로, 여기서 딜레이가 생겨도 메인 화면은 멈추지 않음.
        하지만 가능한 빨리 리턴해주는 것이 좋음.
        """
        if not self.infer_queue.full():
            try:
                img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.infer_queue.put(img_bgr, block=False)
            except Exception:
                pass # 큐 가득 참 or 변환 실패 -> 스킵 (Real-time 유지)

    def preprocess_image(self, cv_image):
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        processed_image = helper.pre_process(image_rgb)
        return np.expand_dims(processed_image, axis=0)


    ####################################
    ####### Thread 1
    ####################################
    def thread_inference_worker(self):
        """ NPU 추론"""
        while self.running:
            try:
                org_img = self.infer_queue.get(timeout=0.1)
                
                vis_img = cv2.resize(org_img, (0, 0), fx=VIS_SCALE, fy=VIS_SCALE, interpolation=cv2.INTER_LINEAR)
                vis_h, vis_w = vis_img.shape[:2]
                processing_shapes_list = [(vis_h, vis_w)]

                input_tensor = self.preprocess_image(vis_img)
                out_npu = mxq_model1.infer(input_tensor)

                nms_outs = helper.post_process(out_npu, processing_shapes_list)
                img_shape_processed = input_tensor.shape[-2:]
                results = helper.post_process.nmsout2eval(
                    nms_outs, img_shape_processed, processing_shapes_list
                )
                
                if not self.vis_queue.full():
                    self.vis_queue.put((vis_img, results), block=False)
                
            except queue.Empty:
                continue
            except Exception:
                # Error logging
                traceback.print_exc()

                
                
    ####################################
    ####### Thread 2
    ####################################
    def thread_visualization_worker(self):
        """ Viz (CPU/Numba)"""
        while self.running:
            try:
                vis_img, (labels_list, boxes_list, scores_list, extra_list) = self.vis_queue.get(timeout=0.1)
                
                labels = labels_list[0] if labels_list else []
                boxes = boxes_list[0] if boxes_list else []
                scores = scores_list[0] if scores_list else []
                extras = extra_list[0] if extra_list else []

                if len(labels) > 0:
                    display_img = vis_img 
                    
                    for box, label, score, extra in zip(boxes, labels, scores, extras):
                        # if score < 0.4: continue 

                        class_idx = int(label)
                        color_np = self.colors[class_idx % 92] # NUM_LABELS
                        color_tuple = tuple(color_np.tolist())
                        label_name  = COCO_CLASS_TO_IDX.get(class_idx, str(class_idx))
                        
                        
                        if extra and 'counts' in extra:
                            mask = maskUtils.decode(extra)
                            mask = np.ascontiguousarray(mask)
                            
                            # nogil 
                            apply_mask_numba(display_img, mask, color_np, alpha=0.4)

                        x1, y1, x2, y2 = map(int, [box[0], box[1], box[0]+box[2], box[1]+box[3]])
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), color_tuple, 2)
                        cv2.putText(display_img, label_name, (x1, y1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    final_show = display_img
                else:
                    final_show = vis_img

                if not self.display_queue.full():
                    self.display_queue.put(final_show, block=False)

            except queue.Empty:
                continue
            except Exception:
                traceback.print_exc()

    def get_latest_frame(self):
        try:
            return self.display_queue.get(block=False)
        except queue.Empty:
            return None

def main(args=None):
    rclpy.init(args=args)
    npu_node = NPU_Node()

    # ROS 통신을 별도 데몬 쓰레드로 분리하여 GUI 부하와 상관없이 수신 보장
    spin_thread = threading.Thread(target=rclpy.spin, args=(npu_node,), daemon=True)
    spin_thread.start()

    cv2.namedWindow("NPU Segmentation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("NPU Segmentation", 1280,720)

    prev_time = time.time()
    frame_cnt = 0

    try:
        while rclpy.ok():
            frame = npu_node.get_latest_frame()
            
            if frame is not None:
                frame_cnt += 1
                curr_time = time.time()
                elapsed = curr_time - prev_time
                if elapsed >= 1.0:
                    fps = frame_cnt / elapsed
                    print(f"FPS: {fps:.2f}")
                    frame_cnt = 0
                    prev_time = curr_time

                cv2.imshow("NPU Segmentation", frame)
            
            # rclpy.spin_once() 제거

            key = cv2.waitKey(1) & 0xFF
            if key == 27: # ESC
                break

    except KeyboardInterrupt:
        pass
    finally:
        npu_node.running = False
        # spin_thread는 daemon=True라 메인 종료 시 자동 종료
        npu_node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()