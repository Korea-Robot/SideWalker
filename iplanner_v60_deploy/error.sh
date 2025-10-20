python navigation_iplanner_viz.py 
1755784030.190047 [123] pt_main_th: config: //CycloneDDS/Domain/General: 'NetworkInterfaceAddress': deprecated element (file:///home/krm/.cyclonedds.xml line 8)
[INFO] [1755784030.769941950] [realsense_planner_control_viz]: PlannerNet model loaded successfully on cuda
[INFO] [1755784030.772733449] [realsense_planner_control_viz]: Starting visualization thread.
[INFO] [1755784030.773134629] [realsense_planner_control_viz]: ✅ RealSense PlannerNet Control with Visualization has started.
[INFO] [1755784031.263214435] [realsense_planner_control_viz]: Shutting down...
[INFO] [1755784031.266169868] [realsense_planner_control_viz]: Visualization thread stopped.
Traceback (most recent call last):
  File "/home/krm/World/iplanner/navigation_iplanner_viz.py", line 211, in control_callback
    final_img = self.draw_path_and_direction(img_to_draw, waypoints, angular_z)
  File "/home/krm/World/iplanner/navigation_iplanner_viz.py", line 146, in draw_path_and_direction
    for i, (wp_x, wp_y) in enumerate(waypoints):
ValueError: too many values to unpack (expected 2)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/krm/World/iplanner/navigation_iplanner_viz.py", line 250, in <module>
    main()
  File "/home/krm/World/iplanner/navigation_iplanner_viz.py", line 241, in main
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
  File "/home/krm/World/iplanner/navigation_iplanner_viz.py", line 224, in control_callback
    self.get_logger().error(f"Control loop error: {e}", exc_info=True)
  File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/impl/rcutils_logger.py", line 345, in error
    return self.log(message, LoggingSeverity.ERROR, **kwargs)
  File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/impl/rcutils_logger.py", line 284, in log
    detected_filters = get_filters_from_kwargs(**kwargs)
  File "/opt/ros/humble/local/lib/python3.10/dist-packages/rclpy/impl/rcutils_logger.py", line 207, in get_filters_from_kwargs
    raise TypeError(
TypeError: parameter "exc_info" is not one of the recognized logging options "['throttle_duration_sec', 'throttle_time_source_type', 'skip_first', 'once']"