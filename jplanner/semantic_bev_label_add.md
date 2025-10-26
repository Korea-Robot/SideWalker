def process_and_publish_semantic_bev(
        self, transformed_cloud, mask_aligned, stamp
    ):
        # ... (이전 코드 동일) ...
        # 13. 라벨 -> RGB 색상 변환 (GPU)
        rgb_float32_gpu = self._label_to_color_gpu(label_values)

        # --- ⬇️ 추가된 부분 ⬇️ ---
        # 13.5. 라벨 -> Float32 변환 (GPU)
        # (N,) long -> (N,) uint32 -> (N,) float32
        labels_float32_gpu = label_values.long().to(torch.uint32).view(torch.float32)
        # --- ⬆️ 추가된 부분 ⬆️ ---

        # 14. (X, Y, Z, RGB, Label) 데이터 결합 (GPU)
        # --- ⬇️ 수정된 부분 ⬇️ ---
        bev_data_gpu = torch.stack(
            [x_world, y_world, z_world, rgb_float32_gpu, labels_float32_gpu],
            dim=-1 # (N, 5)
        )
        # --- ⬆️ 수정된 부분 ⬆️ ---

        # 15. GPU -> CPU 전송
        bev_data_np = bev_data_gpu.cpu().numpy()
        
        # ... (이후 코드 계속) ...


# ... (이전 코드 동일) ...
        # 15. GPU -> CPU 전송
        bev_data_np = bev_data_gpu.cpu().numpy()

        # 16. PointCloud2 메시지 생성 (CPU)
        # --- ⬇️ 수정된 부분 ⬇️ ---
        bev_msg = self._create_semantic_cloud_from_data(
            bev_data_np, stamp, self.target_frame
        )
        # --- ⬆️ 수정된 부분 ⬆️ ---

        # 17. 발행
        self.sem_bev_pub.publish(bev_msg)

def _create_bev_cloud_from_data(self, point_data_np, stamp, frame_id):
        """
        (N, 5) [x, y, z, rgb_float32, label_float32] NumPy 배열로
        BEV PointCloud2 메시지를 생성합니다.
        (PCL 생성 함수와 동일해짐)
        """
        header = Header(stamp=stamp, frame_id=frame_id)
        num_points = point_data_np.shape[0]

        # --- ⬇️ 수정된 부분 ⬇️ ---
        return PointCloud2(
            header=header,
            height=1,
            width=num_points,
            fields=self.semantic_bev_fields, # 1번에서 5개 필드로 수정됨
            is_bigendian=False,
            point_step=self.point_step_bev, # 1번에서 20으로 수정됨
            row_step=self.point_step_bev * num_points,
            data=point_data_np.astype(np.float32).tobytes(),
            is_dense=True,
        )
        # --- ⬆️ 수정된 부분 ⬆️ ---
