python3 optimized_semantic_pointcloud_node.py
1761379862.132970 [123] pt_main_th: config: //CycloneDDS/Domain/General: 'NetworkInterfaceAddress': deprecated element (file:///home/krm/.cyclonedds.xml line 8)

>>> [rcutils|error_handling.c:108] rcutils_set_error_state()
This error state is being overwritten:

  'string data is not null-terminated, at ./src/serdata.cpp:384'

with this new error message:

  'invalid data size, at ./src/serdata.cpp:384'

rcutils_reset_error() should be called after error handling to avoid this.
<<<
[INFO] [1761379862.218100785] [semantic_pointcloud_bev_node]: 🚀 CUDA GPU 가속 활성화 (PyTorch)
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
/home/krm/.local/lib/python3.10/site-packages/transformers/image_processing_base.py:412: UserWarning: The following named arguments are not valid for `Mask2FormerImageProcessor.__init__` and were ignored: '_max_size', 'reduce_labels'
  image_processor = cls(**image_processor_dict)
[INFO] [1761379864.648432929] [semantic_pointcloud_bev_node]: ✅ MaskFormer-COCO model loaded
[INFO] [1761379864.649242861] [semantic_pointcloud_bev_node]: ⚡ Half Precision (FP16) enabled for inference
Traceback (most recent call last):
  File "/home/krm/SideWalker/jplanner/optimized_semantic_pointcloud_node.py", line 513, in <module>
    main()
  File "/home/krm/SideWalker/jplanner/optimized_semantic_pointcloud_node.py", line 500, in main
    node = SemanticPointCloudBEVNode()
  File "/home/krm/SideWalker/jplanner/optimized_semantic_pointcloud_node.py", line 126, in __init__
    self._init_gpu_parameters()
  File "/home/krm/SideWalker/jplanner/optimized_semantic_pointcloud_node.py", line 139, in _init_gpu_parameters
    h_d = self.config.depth_intrinsics.height
AttributeError: 'CameraIntrinsics' object has no attribute 'height'
