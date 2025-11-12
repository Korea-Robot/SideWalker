# calculation logging code 

```python
   def log_mission_summary(self):
        """
        미션 종료 시(성공, 중단, 실패) 최종 통계를 로깅합니다.
        """
        # 중복 로깅 방지
        if self.summary_logged:
            return
        self.summary_logged = True
        
        completed_count = self.waypoint_index
        total_count = self.total_waypoints

        if total_count == 0:
            self.get_logger().info("Mission summary: No waypoints were loaded.")
            return

        # (e.g., 3 / 10) * 100.0 = 30.0
        percentage = (completed_count / total_count) * 100.0

        # --- (신규) 완료된 웨이포인트 목록 생성 ---
        completed_list_str = ""
        if completed_count > 0:
            # 1-based 인덱스(i+1)와 좌표를 함께 기록
            # 사용자가 요청한 대로 for-loop을 사용하여 목록 생성
            waypoint_lines = []
            for i in range(completed_count):
                # (수정) "waypoint success"와 1-based index 사용
                line = f"    > Waypoint {i + 1}번 성공 (좌표: {self.waypoints[i]})"
                waypoint_lines.append(line)
            
            completed_list_str = "\n".join(waypoint_lines)
        # ------------------------------------

        summary_msg = (
            f"\n--- 🏁 Mission Summary 🏁 ---\n"
            f"  Waypoints Completed: {completed_count} / {total_count}\n"
            f"  Completion Rate:     {percentage:.1f}%\n"
        )
        
        # (신규) 완료된 목록이 있으면 요약에 추가
        if completed_list_str:
            summary_msg += "  --- Completed List ---\n"
            summary_msg += f"{completed_list_str}\n"
            summary_msg += "  ----------------------\n"

        # 상태 확인
        if completed_count == total_count:
            # 100% 성공
            summary_msg += "  Status:              SUCCESS (All waypoints reached)"
            self.get_logger().info(summary_msg)
        else:
            # 100% 미만 (중단 또는 실패)
            with self.plot_data_lock:
                status = self.current_status
            
            # 상태가 '충돌'이나 '에러'가 아닌데 종료된 경우 (e.g. Ctrl+C)
            if "COLLISION" not in status and "ERROR" not in status:
                 status = "Interrupted (e.g., Ctrl+C or Viz close)"
            
            summary_msg += f"  Status:              STOPPED ({status})"
            # 실패/중단은 WARN 레벨로 로깅
            self.get_logger().warn(summary_msg)
```

# logs  

## test1

```bash
[INFO] [1762843837.915871661] [mppi_bev_planner_viz_node]: 
--- 🏁 Mission Summary 🏁 ---
  Waypoints Completed: 60 / 60
  Completion Rate:     100.0%
  --- Completed List ---
    > Waypoint 1번 성공 (좌표: (12.75, -30.78))
    > Waypoint 2번 성공 (좌표: (4.46, 0.26))
    > Waypoint 3번 성공 (좌표: (1.0, 1.0))
    > Waypoint 4번 성공 (좌표: (4.46, 0.26))
    > Waypoint 5번 성공 (좌표: (12.75, -30.78))
    > Waypoint 6번 성공 (좌표: (24.16, -30.74))
    > Waypoint 7번 성공 (좌표: (29.65, -97.64))
    > Waypoint 8번 성공 (좌표: (32.42, -96.53))
    > Waypoint 9번 성공 (좌표: (61.57, -101.34))
    > Waypoint 10번 성공 (좌표: (60.59, -67.95))
    > Waypoint 11번 성공 (좌표: (53.99, -22.33))
    > Waypoint 12번 성공 (좌표: (32.87, -28.13))
    > Waypoint 13번 성공 (좌표: (12.75, -30.78))
    > Waypoint 14번 성공 (좌표: (4.46, 0.26))
    > Waypoint 15번 성공 (좌표: (1.0, 1.0))
    > Waypoint 16번 성공 (좌표: (4.46, 0.26))
    > Waypoint 17번 성공 (좌표: (12.75, -30.78))
    > Waypoint 18번 성공 (좌표: (24.16, -30.74))
    > Waypoint 19번 성공 (좌표: (29.65, -97.64))
    > Waypoint 20번 성공 (좌표: (32.42, -96.53))
    > Waypoint 21번 성공 (좌표: (61.57, -101.34))
    > Waypoint 22번 성공 (좌표: (60.59, -67.95))
    > Waypoint 23번 성공 (좌표: (53.99, -22.33))
    > Waypoint 24번 성공 (좌표: (32.87, -28.13))
    > Waypoint 25번 성공 (좌표: (12.75, -30.78))
    > Waypoint 26번 성공 (좌표: (4.46, 0.26))
    > Waypoint 27번 성공 (좌표: (1.0, 1.0))
    > Waypoint 28번 성공 (좌표: (4.46, 0.26))
    > Waypoint 29번 성공 (좌표: (12.75, -30.78))
    > Waypoint 30번 성공 (좌표: (24.16, -30.74))
    > Waypoint 31번 성공 (좌표: (29.65, -97.64))
    > Waypoint 32번 성공 (좌표: (32.42, -96.53))
    > Waypoint 33번 성공 (좌표: (61.57, -101.34))
    > Waypoint 34번 성공 (좌표: (60.59, -67.95))
    > Waypoint 35번 성공 (좌표: (53.99, -22.33))
    > Waypoint 36번 성공 (좌표: (32.87, -28.13))
    > Waypoint 37번 성공 (좌표: (12.75, -30.78))
    > Waypoint 38번 성공 (좌표: (4.46, 0.26))
    > Waypoint 39번 성공 (좌표: (1.0, 1.0))
    > Waypoint 40번 성공 (좌표: (4.46, 0.26))
    > Waypoint 41번 성공 (좌표: (12.75, -30.78))
    > Waypoint 42번 성공 (좌표: (24.16, -30.74))
    > Waypoint 43번 성공 (좌표: (29.65, -97.64))
    > Waypoint 44번 성공 (좌표: (32.42, -96.53))
    > Waypoint 45번 성공 (좌표: (61.57, -101.34))
    > Waypoint 46번 성공 (좌표: (60.59, -67.95))
    > Waypoint 47번 성공 (좌표: (53.99, -22.33))
    > Waypoint 48번 성공 (좌표: (32.87, -28.13))
    > Waypoint 49번 성공 (좌표: (12.75, -30.78))
    > Waypoint 50번 성공 (좌표: (4.46, 0.26))
    > Waypoint 51번 성공 (좌표: (1.0, 1.0))
    > Waypoint 52번 성공 (좌표: (4.46, 0.26))
    > Waypoint 53번 성공 (좌표: (12.75, -30.78))
    > Waypoint 54번 성공 (좌표: (24.16, -30.74))
    > Waypoint 55번 성공 (좌표: (29.65, -97.64))
    > Waypoint 56번 성공 (좌표: (32.42, -96.53))
    > Waypoint 57번 성공 (좌표: (61.57, -101.34))
    > Waypoint 58번 성공 (좌표: (60.59, -67.95))
    > Waypoint 59번 성공 (좌표: (53.99, -22.33))
    > Waypoint 60번 성공 (좌표: (32.87, -28.13))
  ----------------------
  Status:              SUCCESS (All waypoints reached)
^[$
```


## test2

```bash
--- 🏁 Mission Summary 🏁 ---
  Waypoints Completed: 36 / 36
  Completion Rate:     100.0%
  --- Completed List ---
    > Waypoint 1번 성공 (좌표: (12.75, -30.78))
    > Waypoint 2번 성공 (좌표: (4.46, 0.26))
    > Waypoint 3번 성공 (좌표: (1.0, 1.0))
    > Waypoint 4번 성공 (좌표: (4.46, 0.26))
    > Waypoint 5번 성공 (좌표: (12.75, -30.78))
    > Waypoint 6번 성공 (좌표: (24.16, -30.74))
    > Waypoint 7번 성공 (좌표: (29.65, -97.64))
    > Waypoint 8번 성공 (좌표: (32.42, -96.53))
    > Waypoint 9번 성공 (좌표: (61.57, -101.34))
    > Waypoint 10번 성공 (좌표: (60.59, -67.95))
    > Waypoint 11번 성공 (좌표: (53.99, -22.33))
    > Waypoint 12번 성공 (좌표: (32.87, -28.13))
    > Waypoint 13번 성공 (좌표: (12.75, -30.78))
    > Waypoint 14번 성공 (좌표: (4.46, 0.26))
    > Waypoint 15번 성공 (좌표: (1.0, 1.0))
    > Waypoint 16번 성공 (좌표: (4.46, 0.26))
    > Waypoint 17번 성공 (좌표: (12.75, -30.78))
    > Waypoint 18번 성공 (좌표: (24.16, -30.74))
    > Waypoint 19번 성공 (좌표: (29.65, -97.64))
    > Waypoint 20번 성공 (좌표: (32.42, -96.53))
    > Waypoint 21번 성공 (좌표: (61.57, -101.34))
    > Waypoint 22번 성공 (좌표: (60.59, -67.95))
    > Waypoint 23번 성공 (좌표: (53.99, -22.33))
    > Waypoint 24번 성공 (좌표: (32.87, -28.13))
    > Waypoint 25번 성공 (좌표: (12.75, -30.78))
    > Waypoint 26번 성공 (좌표: (4.46, 0.26))
    > Waypoint 27번 성공 (좌표: (1.0, 1.0))
    > Waypoint 28번 성공 (좌표: (4.46, 0.26))
    > Waypoint 29번 성공 (좌표: (12.75, -30.78))
    > Waypoint 30번 성공 (좌표: (24.16, -30.74))
    > Waypoint 31번 성공 (좌표: (29.65, -97.64))
    > Waypoint 32번 성공 (좌표: (32.42, -96.53))
    > Waypoint 33번 성공 (좌표: (61.57, -101.34))
    > Waypoint 34번 성공 (좌표: (60.59, -67.95))
    > Waypoint 35번 성공 (좌표: (53.99, -22.33))
    > Waypoint 36번 성공 (좌표: (32.87, -28.13))
  ----------------------
  Status:              SUCCESS (All waypoints reached)
==^_^_[INFO] [1762847211.651538121] [mppi_bev_planner_viz_node]: Matplotlib closed, shutting down ROS node.
[INFO] [1762847211.652557210] [mppi_bev_planner_viz_node]: Shutting down... Stopping robot.

```

## test3

```bash
iz_node]: ✅ Waypoint 20번 성공! (좌표: (32.87, -28.13))
[INFO] [1762849585.693819556] [mppi_bev_planner_viz_node]: 
--- 🏁 Mission Summary 🏁 ---
  Waypoints Completed: 20 / 20
  Completion Rate:     100.0%
  --- Completed List ---
    > Waypoint 1번 성공 (좌표: (24.16, -30.74))
    > Waypoint 2번 성공 (좌표: (33.65, -77.64))
    > Waypoint 3번 성공 (좌표: (45.77, -22.33))
    > Waypoint 4번 성공 (좌표: (32.87, -28.13))
    > Waypoint 5번 성공 (좌표: (24.16, -30.74))
    > Waypoint 6번 성공 (좌표: (33.65, -77.64))
    > Waypoint 7번 성공 (좌표: (45.77, -22.33))
    > Waypoint 8번 성공 (좌표: (32.87, -28.13))
    > Waypoint 9번 성공 (좌표: (24.16, -30.74))
    > Waypoint 10번 성공 (좌표: (33.65, -77.64))
    > Waypoint 11번 성공 (좌표: (45.77, -22.33))
    > Waypoint 12번 성공 (좌표: (32.87, -28.13))
    > Waypoint 13번 성공 (좌표: (24.16, -30.74))
    > Waypoint 14번 성공 (좌표: (33.65, -77.64))
    > Waypoint 15번 성공 (좌표: (45.77, -22.33))
    > Waypoint 16번 성공 (좌표: (32.87, -28.13))
    > Waypoint 17번 성공 (좌표: (24.16, -30.74))
    > Waypoint 18번 성공 (좌표: (33.65, -77.64))
    > Waypoint 19번 성공 (좌표: (45.77, -22.33))
    > Waypoint 20번 성공 (좌표: (32.87, -28.13))
  ----------------------
  Status:              SUCCESS (All waypoints reached)


```