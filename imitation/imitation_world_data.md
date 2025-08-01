
# 🚘 World Model & Imitation Learning Dataset Summary

이 문서는 RGB, Depth, Semantic 이미지 기반 시뮬레이션 데이터를 이용하여 World Model 및 Imitation Learning 학습을 위한 데이터셋 구조와 구성 요소를 정의합니다.

---

## 📦 데이터 저장 구조 (에피소드 단위) 
#### 가변 에피소드임 (reset index 기준) - crash, terminated 등 끝나면 에피소드 종료.

```

data/
├── episode\_0001/
│   ├── rgb/0000.png, 0001.png, ...
│   ├── depth/0000.png, ...
│   ├── semantic/0000.png, ...
│   ├── action\_reward.json
│   ├── ego\_state.json
│   ├── waypoints.json
├── episode\_0002/
...

````

---

## 🧩 각 파일 설명

- `rgb/`, `depth/`, `semantic/`: 시각 센서 이미지 (PNG)
- `action_reward_goal_egostate.json`:

```json
  [
    {"step": 0, "action": [steer, throttle], "reward": r, "done": false,"goal":goal_position,"position": [x, y], "heading": θ},
    ...
  ]
```

- `waypoints.json`: 전체 경로 좌표 리스트

---

## 📌 필수 요소 (per step)

| 요소               | 설명                       |
| ---------------- | ------------------------ |
| `obs_t`          | RGB, Depth, Semantic 이미지 |
| `action_t`       | 조향, 스로틀  [-1~1]         |
| `reward_t`       | 리워드                      |
| `done_t`         | 종료 여부                    |
| `obs_{t+1}`      | 다음 이미지 세트               |
| `position_t`     | 현재 위치                    |
| `position_{t+1}` | 다음 위치                    |
| `heading_t`      | 현재 heading angle         |
| `heading_{t+1}`  | 다음 heading angle         |

---

## 🧠 학습 목적에 따른 활용

### 🔁 Imitation Learning (Behavior Cloning)

* **Input**: `obs_t` (이미지) , `goal` (목표 지점)
* **Target**: `action_t`

### 🌍 World Model (Dynamics Prediction)

* **Input**: `obs_t + action_t`
* **Target**: `obs_{t+1}` 또는 latent encoding

### 🎯 Reward Model

* **Input**: `obs_t`, `action_t`,  `goal`
* **Target**: `reward_t`

### 🗺 Position/Heading Prediction

* **Input**: `obs_t`, `action_t`
* **Target**: `position_{t+1}`, `heading_{t+1}`

---

## ⚙️ PyTorch Dataset 구성 방향

* 에피소드 기반 Dataset
* transition 단위 Dataset (N-step 가능)
* 가변 길이 에피소드 → collate\_fn으로 처리
* 전처리 transform (e.g., Resize, Normalize) 포함
* GRU - world model구조를 위해서 최소 각각 H step의 데이터셋이 필요함.
* 현재 정한걸로는 H - step 이하로 생성된 데이터는 버리고 H이상일때만 모은다.
* 또한 어디에 계속 끼어서 못가는 경우가 생길수있으므로 30스텝동안 일정 거리이상 안움직이면 그 데이터도 버린다.

---

## ✅ 확장 고려 사항

* `next_obs`를 명시적으로 저장 또는 인덱스 기반 접근
* 이미지 → tensor 변환 후 `.pt` 저장 고려 (속도 ↑)
* 데이터 압축 및 LMDB/HDF5로 저장 고려 가능

---
