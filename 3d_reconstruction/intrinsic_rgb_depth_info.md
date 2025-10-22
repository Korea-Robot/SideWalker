# RGB info

header:
  stamp:
    sec: 1761121262
    nanosec: 603631592
  frame_id: camera_color_optical_frame
height: 720
width: 1280
distortion_model: plumb_bob
d:
- -0.05555006489157677
- 0.06587371975183487
- 5.665919161401689e-05
- 0.0014403886161744595
- -0.02127622440457344
k:
- 645.4923095703125
- 0.0
- 653.0325927734375
- 0.0
- 644.4183349609375
- 352.2890930175781
- 0.0
- 0.0
- 1.0
r:
- 1.0
- 0.0
- 0.0
- 0.0
- 1.0
- 0.0
- 0.0
- 0.0
- 1.0
p:
- 645.4923095703125
- 0.0
- 653.0325927734375
- 0.0
- 0.0
- 644.4183349609375
- 352.2890930175781
- 0.0
- 0.0
- 0.0
- 1.0
- 0.0
binning_x: 0
binning_y: 0
roi:
  x_offset: 0
  y_offset: 0
  height: 0
  width: 0
  do_rectify: false
---


# Depth info 
header:
  stamp:
    sec: 1761121196
    nanosec: 169050293
  frame_id: camera_depth_optical_frame
height: 480
width: 848
distortion_model: plumb_bob
d:
- 0.0
- 0.0
- 0.0
- 0.0
- 0.0
k:
- 431.0625305175781
- 0.0
- 434.49224853515625
- 0.0
- 431.0625305175781
- 242.76425170898438
- 0.0
- 0.0
- 1.0
r:
- 1.0
- 0.0
- 0.0
- 0.0
- 1.0
- 0.0
- 0.0
- 0.0
- 1.0
p:
- 431.0625305175781
- 0.0
- 434.49224853515625
- 0.0
- 0.0
- 431.0625305175781
- 242.76425170898438
- 0.0
- 0.0
- 0.0
- 1.0
- 0.0
binning_x: 0
binning_y: 0
roi:
  x_offset: 0
  y_offset: 0
  height: 0
  width: 0
  do_rectify: false
---


ros2 topic echo /camera/camera/extrinsics/depth_to_color
1761122179.085108 [123]       ros2: config: //CycloneDDS/Domain/General: 'NetworkInterfaceAddress': deprecated element (file:///home/krm/.cyclonedds.xml line 8)
rotation:
- 0.999993622303009
- -0.0006787928286939859
- -0.0035094195045530796
- 0.0006689286092296243
- 0.9999958276748657
- -0.0028111860156059265
- 0.0035113131161779165
- 0.002808820456266403
- 0.9999898672103882
translation:
- -0.05926649272441864
- 1.0062108231068123e-05
- 0.0007165665156207979




krm@ubuntu:~$ rviz
bash: rviz: command not found
krm@ubuntu:~$ ros2 topic echo /camera/camera/depth/color/points


---
header:
  stamp:
    sec: 1761122539
    nanosec: 465128418
  frame_id: camera_depth_optical_frame
height: 1
width: 352346
fields:
- name: x
  offset: 0
  datatype: 7
  count: 1
- name: y
  offset: 4
  datatype: 7
  count: 1
- name: z
  offset: 8
  datatype: 7
  count: 1
- name: rgb
  offset: 16
  datatype: 7
  count: 1
is_bigendian: false
point_step: 20
row_step: 7046920
data:
