python3 semantic_bev_node.py
1761051774.925436 [123] pt_main_th: config: //CycloneDDS/Domain/General: 'NetworkInterfaceAddress': deprecated element (file:///home/krm/.cyclonedds.xml line 8)
[INFO] [1761051775.037210940] [realtime_semantic_bev_node]: Using device: cuda
Some weights of SegformerForSemanticSegmentation were not initialized from the model checkpoint at nvidia/mit-b0 and are newly initialized: ['decode_head.batch_norm.bias', 'decode_head.batch_norm.num_batches_tracked', 'decode_head.batch_norm.running_mean', 'decode_head.batch_norm.running_var', 'decode_head.batch_norm.weight', 'decode_head.classifier.bias', 'decode_head.classifier.weight', 'decode_head.linear_c.0.proj.bias', 'decode_head.linear_c.0.proj.weight', 'decode_head.linear_c.1.proj.bias', 'decode_head.linear_c.1.proj.weight', 'decode_head.linear_c.2.proj.bias', 'decode_head.linear_c.2.proj.weight', 'decode_head.linear_c.3.proj.bias', 'decode_head.linear_c.3.proj.weight', 'decode_head.linear_fuse.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
[INFO] [1761051778.636358071] [realtime_semantic_bev_node]: ✅ Successfully loaded model weights from 'best_model.pth'.
Traceback (most recent call last):
  File "/home/krm/SideWalker/segformer/polygon/semantic_bev_node.py", line 309, in <module>
    main()
  File "/home/krm/SideWalker/segformer/polygon/semantic_bev_node.py", line 301, in main
    rclpy.spin(node)
  File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/__init__.py", line 229, in spin
    executor.spin_once()
  File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py", line 751, in spin_once
    self._spin_once_impl(timeout_sec)
  File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py", line 748, in _spin_once_impl
    raise handler.exception()
  File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/task.py", line 254, in __call__
    self._handler.send(None)
  File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py", line 447, in handler
    await call_coroutine(entity, arg)
  File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py", line 361, in _execute_timer
    await await_or_execute(tmr.callback)
  File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/executors.py", line 107, in await_or_execute
    return callback(*args)
  File "/home/krm/SideWalker/segformer/polygon/semantic_bev_node.py", line 197, in process_and_visualize
    bev_colored = cv2.LUT(semantic_bev,self.bev_colormap_cv)
cv2.error: OpenCV(4.12.0) /io/opencv/modules/core/src/lut.cpp:159: error: (-215:Assertion failed) (lutcn == cn || lutcn == 1) && _lut.total() == 256 && _lut.isContinuous() && (depth == CV_8U || depth == CV_8S) in function 'LUT'