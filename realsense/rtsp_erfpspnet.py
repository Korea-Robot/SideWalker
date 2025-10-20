#!/usr/bin/env python3
import numpy as np
import cv2
import torch
from erfpspnet import Net  # erfpspnet.py 파일이 필요합니다.
from collections import OrderedDict

def main():
    # 1) 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2) 사용자 정의 PSPNet 모델 로드
    try:
        model = Net(22)  # 클래스 개수에 맞게 조정
        # 모델 가중치 로드 (CPU/GPU 자동 매핑)
        state_dict = torch.load("./model_best.pth", map_location=device)
        # 'module.' prefix 제거 (DataParallel로 학습된 모델을 로드할 때 필요)
        new_state = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state[name] = v
        model.load_state_dict(new_state)
        model.to(device).eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: model_best.pth or erfpspnet.py not found.")
        print("Please ensure the model file and definition file are in the same directory.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    # 3) 클래스 컬러 팔레트 정의
    palette_base = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1], dtype=torch.int64)
    colors = (torch.arange(256, dtype=torch.int64)[:, None] * palette_base) % 255
    colors = colors.numpy().astype('uint8')  # RGB 팔레트

    # 4) RTSP 스트림 설정
    rtsp_url = "rtsp://192.168.168.105:4001/front_left" # RTSP 주소 형식에 맞게 수정
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream at {rtsp_url}")
        return

    print("RTSP stream opened successfully. Starting inference...")

    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                print("Stream end or error. Exiting...")
                break

            img_bgr = frame  # cap.read()는 이미 numpy 배열(BGR)을 반환

            # 5) 전처리: BGR -> RGB, numpy -> tensor, 정규화
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor_img = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True).float() / 255.0

            # 6) 추론
            with torch.no_grad():
                logits = model(tensor_img)

            # 7) 예측 마스크 생성
            preds = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)  # HxW

            # 8) 마스크 컬러화 & 원본 이미지와 오버레이
            mask_rgb = colors[preds]
            overlay = cv2.addWeighted(img_rgb, 0.5, mask_rgb, 0.5, 0)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR) # OpenCV 표시는 BGR로

            # 9) 결과 디스플레이: 원본 영상과 세그멘테이션 결과 나란히 표시
            combined_display = np.hstack((img_bgr, overlay_bgr))

            cv2.imshow("Original | PSPNet Segmentation", combined_display)

            # 'ESC' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == 27:
                print("ESC key pressed. Exiting.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # 10) 자원 해제
        print("Releasing resources.")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
