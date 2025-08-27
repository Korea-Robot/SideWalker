🎯 주요 기능
1. 실시간 OpenCV 시각화

3개 이미지 (원본, 세그멘테이션, 오버레이)를 하나의 창에 나란히 표시
실시간 FPS, 추론시간, 프레임 카운트 표시
클래스별 색상 정보 창 (옵션)

2. 사용자 친화적 인터페이스

ESC: 프로그램 종료
S: 스크린샷 저장
실시간 성능 모니터링

3. 자동 크기 조정

이미지가 너무 크면 자동으로 400px 너비로 조정
3개 이미지를 가로로 배치하여 한 눈에 비교 가능

🚀 실행 방법
1. 기본 실행
bashpython3 realsense_segmentation_inference.py
2. 클래스 정보 창과 함께 실행
bashpython3 realsense_segmentation_inference.py --show_class_info
3. 결과 저장하며 실행
bashpython3 realsense_segmentation_inference.py --save_results --save_interval 10
4. 특정 모델로 실행
bashpython3 realsense_segmentation_inference.py --model_path ckpts/my_model.pth --show_class_info
📊 시각화 구성
┌─────────────┬─────────────┬─────────────┐
│   Original  │Segmentation │   Overlay   │
│    Image    │   Result    │             │
├─────────────┼─────────────┼─────────────┤
│ Inference Time: 25.3ms  FPS: 35.2      │
│ Device: cuda:0          Frame: 1247     │
└─────────────────────────────────────────┘
🎨 색상 매핑

30개 클래스 각각에 고유한 HSV 기반 색상 할당
배경(클래스 0)은 검은색
--show_class_info 옵션으로 클래스-색상 매핑 정보 표시

🔧 설정 옵션

--model_name: 모델 이름 (기본값: nvidia/mit-b0)
--model_path: 학습된 모델 경로
--realsense_topic: ROS2 이미지 토픽
--show_class_info: 클래스 정보 창 표시
--save_results: 결과 자동 저장
--output_dir: 저장 디렉토리

이제 matplotlib 스타일이 아닌 OpenCV로 깔끔하게 실시간 시각화가 됩니다!
