# from metaurban.envs import SidewalkStaticMetaUrbanEnv
# from metaurban.component.sensors.semantic_camera import SemanticCamera
# import matplotlib.pyplot as plt
# import os

# size = (256, 128) if not os.getenv('TEST_DOC') else (16, 16) # for github CI

# env = SidewalkStaticMetaUrbanEnv(dict(
#     object_density=0.1,
#     log_level=50, # suppress log
#     image_observation=True,
#     show_terrain=not os.getenv('TEST_DOC'),
#     sensors={"sementic_camera": [SemanticCamera, *size]},
#     vehicle_config={"image_source": "sementic_camera"},
#     stack_size=3,
# ))
# obs, info = env.reset()
# for _ in range(5):
#     obs, r, d, t, i = env.step((0, 1))

# env.close()

# print({k: v.shape for k, v in obs.items()})  # Image is in shape (H, W, C, num_stacks)

# plt.subplot(131)
# plt.imshow(obs["image"][:, :, :, 0])
# plt.axis('off')
# plt.subplot(132)
# plt.imshow(obs["image"][:, :, :, 1])
# plt.axis('off')
# plt.subplot(133)
# plt.imshow(obs["image"][:, :, :, 2])
# plt.axis('off')
# plt.show()


from metaurban.envs import SidewalkStaticMetaUrbanEnv
from metaurban.component.sensors.semantic_camera import SemanticCamera
import matplotlib.pyplot as plt
import os
import numpy as np # numpy를 임포트하여 사용할 수 있습니다.

# TEST_DOC 환경 변수가 설정되지 않았을 때 (일반 실행 시) 사용할 이미지 크기
# 설정되었다면 (Github CI 등 테스트 환경) 작은 크기 사용
size = (256, 128) if not os.getenv('TEST_DOC') else (16, 16) # (너비, 높이)

env = SidewalkStaticMetaUrbanEnv(dict(
    render_mode=True,
    object_density=0.1,
    log_level=50,  # 로그 메시지 억제
    image_observation=True,
    show_terrain=not os.getenv('TEST_DOC'),
    # 수정 1: 'sementic_camera' -> 'semantic_camera' 오타 수정
    sensors={"semantic_camera": [SemanticCamera, *size]},
    vehicle_config={"image_source": "semantic_camera"}, # 여기도 동일하게 수정
    stack_size=3,
))
obs, info = env.reset()
for _ in range(5):
    obs, r, d, t, i = env.step((0, 1))

env.close()

# 관찰된 데이터의 형태(shape) 출력
print({k: v.shape for k, v in obs.items()})

# 시각화
plt.figure(figsize=(15, 5)) # 전체 그림 크기 조절

# 수정 2: 4차원 이미지 데이터를 올바르게 슬라이싱
# obs["image"]의 shape은 (높이, 너비, 채널, 스택 수), 즉 (128, 256, 1, 3) 입니다.
# plt.imshow는 (높이, 너비) 형태의 2D 배열을 받아야 하므로, 채널 차원(세 번째 차원)을 명시적으로 선택해야 합니다.
# cmap='gray'를 추가하여 흑백으로 명확하게 표시합니다.
plt.subplot(131)
plt.title("Frame t-2")
plt.imshow(obs["image"][:, :, 0, 0], cmap='gray') # 첫 번째 스택 이미지
plt.axis('off')

plt.subplot(132)
plt.title("Frame t-1")
plt.imshow(obs["image"][:, :, 0, 1], cmap='gray') # 두 번째 스택 이미지
plt.axis('off')

plt.subplot(133)
plt.title("Frame t")
plt.imshow(obs["image"][:, :, 0, 2], cmap='gray') # 세 번째 스택 이미지 (가장 최신)
plt.axis('off')

plt.tight_layout()
plt.show()