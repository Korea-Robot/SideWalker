# MetaUrban RL 추론 및 비디오 녹화

학습된 PPO 모델을 사용하여 MetaUrban 환경에서 자율주행을 수행하고 RGB 동영상으로 저장하는 추론 코드입니다.

## 필요한 파일들

추론을 실행하기 위해서는 다음 파일들이 필요합니다:

1. **inference.py** - 메인 추론 코드
2. **model.py** - Actor/Critic 모델 정의
3. **config.py** - 설정 파일
4. **env_config.py** - 환경 설정 파일
5. **utils.py** - 유틸리티 함수들
6. **perceptnet.py** - PerceptNet 모델 (depth encoder용)
7. **학습된 모델 파일** (예: `metaurban_discrete_actor_multimodal_final.pt`)

## 사용법

### 기본 사용법

```bash
python inference.py --actor_path path/to/your/model.pt --output_video output.mp4
```

### 모든 옵션 사용

```bash
python inference.py \
    --actor_path metaurban_discrete_actor_multimodal_final.pt \
    --output_video videos/inference_episode.mp4 \
    --max_steps 1000 \
    --fps 30 \
    --device cuda:0 \
    --seed 42
```

### 스크립트 사용 (권장)

```bash
# 실행 권한 부여
chmod +x run_inference.sh

# 기본 설정으로 실행
./run_inference.sh

# 사용자 정의 설정으로 실행
./run_inference.sh model.pt output.mp4 1500 cuda:1
```

## 매개변수 설명

- `--actor_path`: 학습된 actor 모델 파일 경로 (.pt 파일)
- `--output_video`: 저장할 비디오 파일 경로 (기본값: inference_episode.mp4)
- `--max_steps`: 최대 스텝 수 (기본값: 1000)
- `--fps`: 비디오 프레임 레이트 (기본값: 30)
- `--device`: 실행할 디바이스 (기본값: cuda:0)
- `--seed`: 재현성을 위한 랜덤 시드 (기본값: 42)

## 출력

실행하면 다음과 같은 정보가 출력됩니다:

1. **실시간 진행 상황**: 100스텝마다 현재 상태 출력
2. **에피소드 요약**: 성공/실패 여부, 총 스텝 수, 보상 등
3. **비디오 정보**: 저장된 프레임 수, 영상 길이 등

### 예시 출력

```
Using device: cuda:0
Loading actor model from: metaurban_discrete_actor_multimodal_final.pt
Inference agent initialized successfully!

Starting inference episode...
Starting episode with 45 waypoints
Goal position: [234.5, 123.2]

Step 100: Distance to goal: 45.2, Speed: 0.85, Checkpoint: 5
Step 200: Distance to goal: 38.1, Speed: 0.92, Checkpoint: 8
...

SUCCESS! Reached destination in 456 steps

Episode Summary:
  Success: True
  Crash: False
  Out of road: False
  Steps: 456
  Total reward: 125.4
  Final distance to goal: 2.1
  Checkpoints passed: 28

Saving video...
Video saved to: inference_episode.mp4
Total frames: 456
Duration: 15.20 seconds
```

## 주요 기능

### InferenceAgent 클래스
- 학습된 모델 로드 및 초기화
- 관찰 데이터로부터 행동 선택
- PD 컨트롤러를 통한 최종 steering 조정

### VideoRecorder 클래스
- RGB 프레임 수집 및 저장
- OpenCV를 사용한 MP4 비디오 생성
- 프레임 레이트 및 해상도 설정

### 추론 프로세스
1. 환경 초기화 (충분한 waypoint가 있는 환경 선택)
2. 각 스텝에서 센서 데이터 수집 (RGB, Depth, Semantic)
3. 목표 지점을 ego-centric 좌표로 변환
4. 모델을 통한 행동 선택
5. PD 컨트롤러로 steering 조정
6. 환경 실행 및 결과 저장
7. RGB 프레임을 비디오에 추가

## 문제 해결

### 모델 로드 오류
- 모델 파일 경로가 올바른지 확인
- 모델 파일이 손상되지 않았는지 확인
- GPU 메모리가 충분한지 확인

### 환경 초기화 문제
- MetaUrban이 올바르게 설치되었는지 확인
- 필요한 의존성들이 모두 설치되었는지 확인

### 비디오 저장 오류
- 출력 디렉토리에 쓰기 권한이 있는지 확인
- 디스크 공간이 충분한지 확인
- OpenCV가 올바르게 설치되었는지 확인

## 의존성

```bash
pip install torch torchvision
pip install opencv-python
pip install numpy matplotlib
pip install transformers
pip install metaurban  # 또는 해당 환경에 맞는 설치 방법
```

## 성능 최적화

- GPU 사용을 위해 CUDA 환경 설정
- 배치 크기 1로 추론하므로 GPU 메모리 사용량은 낮음
- 실시간 추론을 위해 모델을 `.eval()` 모드로 설정
- `torch.no_grad()` 컨텍스트로 그래디언트 계산 비활성화