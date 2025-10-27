rko_lio_online_node:
  ros__parameters:
    base_frame: os_sensor
    body_frame: body_lidar
    body_odom_topic: /krm_auto_localization/odom_body
    convergence_criterion: 0.001
    deskew: true
    double_downsample: true
    extrinsic_imu2base_quat_xyzw_xyz:
    - 0.0
    - 0.0
    - 0.0
    - 1.0
    - 0.006
    - -0.012
    - 0.008
    extrinsic_lidar2base_quat_xyzw_xyz:
    - 0.0
    - 0.0
    - 0.0
    - 1.0
    - 0.0
    - 0.0
    - 0.0
    global_map_topic: /rko_lio/global_map
    imu_frame: os_imu
    imu_topic: /ouster/imu
    initialization_phase: false
    invert_odom_tf: false
    lidar_frame: os_sensor
    lidar_topic: /ouster/points
    map_frame: map
    map_path: /root/krm_data/sample.pcd
    map_voxel_size: 0.2
    viz_map_voxel_size: 0.3
    max_correspondance_distance: 0.5
    max_expected_jerk: 3.0
    max_iterations: 100
    max_num_threads: 0
    max_points_per_voxel: 20
    max_range: 100.0
    min_beta: 200.0
    min_range: 1.0
    odom_frame: lidar_tracking_link
    odom_topic: /krm_auto_localization/odom
    publish_deskewed_scan: true
    publish_lidar_acceleration: false
    qos_overrides:
      /parameter_events:
        publisher:
          depth: 1000
          durability: volatile
          history: keep_last
          reliability: reliable
      /tf:
        publisher:
          depth: 100
          durability: volatile
          history: keep_last
          reliability: reliable
    results_dir: results
    run_name: rko_lio_run
    save_results_on_destroy: false
    start_type_description_service: true
    use_global_map: true 
    use_sim_time: false
    voxel_size: 1.0

