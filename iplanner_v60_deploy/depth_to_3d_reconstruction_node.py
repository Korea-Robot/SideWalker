import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import cv2
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, TransformException
from transforms3d.quaternions import quat2mat
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class krm_traject_planningNode(Node):
    def __init__(self):
        super().__init__('pointcloud_pub')
        self.bridge = CvBridge()
        self.camera_info = None
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # subscribe topic 
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, qos_profile)
        self.create_subscription(CameraInfo, '/camera/camera/depth/camera_info', self.info_callback, qos_profile)

        # pusblish topic 
        #self.pcl_pub = self.create_publisher(PointCloud2, '/pcl/output', qos_profile)
        self.robot_pcl_pub = self.create_publisher(PointCloud2, '/pointcloud', qos_profile)
        #self.path_pcl_pub = self.create_publisher(PointCloud2, '/pcl/fil_output', qos_profile)

        # self.target_frame_id='camera_depth_optical_frame'
        self.target_frame_id='body'
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.s_digity= 3 
        self.s_digitx= 2
        # 360 2 2 -> 57600
        # 640 3 3 -> 38400

    def info_callback(self, msg):
        self.camera_info = msg
    def make_array(self,selected_image,transformed_cloud):
        
        transformed_cloud_x=transformed_cloud[::self.s_digity,::self.s_digitx,0]
        transformed_cloud_y=transformed_cloud[::self.s_digity,::self.s_digitx,1]
        transformed_cloud_z=transformed_cloud[::self.s_digity,::self.s_digitx,2]
        selected_image=selected_image[::self.s_digity,::self.s_digitx]
        
        '''
        transformed_cloud_x_f=transformed_cloud_x.copy()
        transformed_cloud_z_f=transformed_cloud_z.copy()
        #mask_arr=np.logical_and(transformed_cloud_x_f<1.2,transformed_cloud_x_f>0.2)
        mask_arr=((transformed_cloud_x_f<2.5) &(transformed_cloud_x_f>0.2) )
        transformed_cloud_x=transformed_cloud_x[mask_arr]
        transformed_cloud_y=transformed_cloud_y[mask_arr]
        transformed_cloud_z=transformed_cloud_z[mask_arr]
        selected_image=selected_image[mask_arr]
        
        
        mask_colo_arr=(np.logical_and(transformed_cloud_z<1.65,transformed_cloud_z>-0.1)).reshape(-1,)
        
        rgb_base_arr= np.zeros_like(selected_image)
        #colors_arr=np.zeros_like(transformed_cloud)

        r_arr=rgb_base_arr+mask_colo_arr*255
        g_arr=rgb_base_arr+mask_colo_arr*0+100
        b_arr=rgb_base_arr+mask_colo_arr*0+200
        '''
        
        mask_colo_arr=(np.logical_and(transformed_cloud_x<1.285,transformed_cloud_x>1.065))
        


        rgb_base_arr= np.zeros_like(selected_image)
        '''
        r_arr=rgb_base_arr+255
        g_arr=rgb_base_arr+100
        b_arr=rgb_base_arr+200
        '''
        r_arr=rgb_base_arr+mask_colo_arr*0+200
        g_arr=rgb_base_arr+mask_colo_arr*148+100
        b_arr=rgb_base_arr+mask_colo_arr*208

        points_arr=np.stack((transformed_cloud_x, transformed_cloud_y, transformed_cloud_z), axis=-1)
        colors_arr=np.stack((r_arr, g_arr, b_arr), axis=-1)
        return points_arr.reshape(-1,3),colors_arr.reshape(-1,3)
    def depth_callback(self, msg):
        #self.get_logger().info("Elapsed time: ms")
        if self.camera_info is None:
            return
        #self.get_logger().info(msg) 
        depth_cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding).astype(np.float32) / 1000.0
        #self.get_logger().info("{:}".format(depth_cv)) 
        point_cloud = self.depth_to_point_cloud_with_camera_info(depth_cv, self.camera_info)
        #point_cloud_modi = point_cloud[point_cloud[:, 2] < 2.0]
        #360 640 3 에서 -1,3으로
        point_cloud_modi=point_cloud.reshape(-1,3)

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame_id,
                'camera_depth_optical_frame',
                rclpy.time.Time()
            )
            hmt = self.transform_to_hmt(transform)
            transformed_cloud = self.apply_transform_to_point_cloud(point_cloud_modi, hmt)
            transformed_cloud=transformed_cloud.reshape(480,848,3)
            filtered_arr,colors_arr=self.make_array(depth_cv,transformed_cloud)
            #filtered_arr=filtered_arr.reshape(-1,3) 
            #colors_arr=colors_arr.reshape(-1,3)    
            robot_msg = self.numpy_to_pointcloud2(filtered_arr,colors_arr, frame_id=self.target_frame_id)
            self.robot_pcl_pub.publish(robot_msg)

        except TransformException as e:
            self.get_logger().error(f'TF Error: {e}')

    def depth_to_point_cloud_with_camera_info(self, depth_map, camera_info):
        K = np.array(camera_info.k).reshape(3, 3)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        height, width = depth_map.shape
        #mask = (depth_map > 0.2) & (depth_map < 2.0)
        #v, u = np.where(mask)
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        z = depth_map
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        

        return np.stack((x, y, z), axis=-1)

    def numpy_to_pointcloud2(self, points,colors, frame_id):
        """
        points: (N, 3) float32 np.ndarray (x, y, z)
        colors: (N, 3) uint8 np.ndarray (r, g, b)
        """
        #rgb_base_arr=np.zeros_like(points)
        #rgb_base_arr=rgb_base_arr.astype(np.uint8)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
            
        # RGB를 uint32로 packed -> float32 reinterpret cast
        rgb_uint32 = (
            (colors[:,0].astype(np.uint32) << 16) |
            (colors[:,1].astype(np.uint32) << 8) |
            (colors[:,2].astype(np.uint32))
        )
        rgb_float32 = rgb_uint32.view(np.float32)

        # XYZ + RGB 붙이기
        pc_with_rgb = np.hstack([points.astype(np.float32), rgb_float32.reshape(-1, 1)])

        data = pc_with_rgb.tobytes()

        pointcloud_msg = PointCloud2(
            header=header,
            height=1,
            width=pc_with_rgb.shape[0],
            fields=fields,
            is_bigendian=False,
            point_step=16,  # 4 * float32 (x,y,z,rgb)
            row_step=16 * pc_with_rgb.shape[0],
            data=data,
            is_dense=True,
        )

        return pointcloud_msg

    def transform_to_hmt(self, transform):
        t = transform.transform.translation
        r = transform.transform.rotation
        trans = [t.x, t.y, t.z]
        quat = [r.x, r.y, r.z, r.w]
        #rot_matrix = quaternion_matrix(quat)[:3, :3]
        rot_matrix = quat2mat([quat[3], quat[0], quat[1], quat[2]])
        hmt = np.eye(4)
        hmt[:3, :3] = rot_matrix
        hmt[:3, 3] = trans
        return hmt

    def apply_transform_to_point_cloud(self, points, hmt):
        ones = np.ones((points.shape[0], 1))
        homo_points = np.hstack((points, ones))
        transformed = homo_points @ hmt.T
        return transformed[:, :3]


def main(args=None):
    rclpy.init(args=args)
    node = krm_traject_planningNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
