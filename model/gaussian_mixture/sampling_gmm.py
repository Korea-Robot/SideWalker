import torch

# 파라미터 (위에서 쓰던 것과 동일)
policy_mean = 0.2
policy_std  = 0.1
prior_mean  = -0.3
prior_std   = 0.2
expl_low    = -1.0
expl_high   = 1.0
weights     = torch.tensor([0.5, 0.3, 0.2])  # [policy, prior, exploration]

# def sample_actions(num_samples: int = 1) -> torch.Tensor:
"""
혼합분포에서 num_samples개의 action을 샘플링.
반환값: shape=(num_samples,)
"""

num_samples = 1
# 1) 컴포넌트 선택 분포
cat = torch.distributions.Categorical(probs=weights)
comps = cat.sample((num_samples,))  # 0,1,2 인덱스
# Categorical에서 뽑힌 0,1,2 인덱스들이 들어있는 정수 텐서


# 2) 각 분포에서 미리 샘플 생성
policy_samps = torch.normal(policy_mean, policy_std, size=(num_samples,))
prior_samps  = torch.normal(prior_mean,  prior_std,  size=(num_samples,))
expl_samps   = torch.rand(num_samples) * (expl_high - expl_low) + expl_low


# comps == 0 처럼 비교 연산을 하면, 같은 모양의 Boolean 텐서가 생성됩니다.
# 예를 들어 comps = tensor([0,2,1,0]) 라면
# comps == 0 ⇒ tensor([True, False, False, True])
# comps == 1 ⇒ tensor([False, False, True, False])
# comps == 2 ⇒ tensor([False, True, False, False])

# 3) 컴포넌트별 샘플 매핑
samples = torch.empty(num_samples)
samples[comps == 0] = policy_samps[comps == 0]
samples[comps == 1] = prior_samps[ comps == 1]
samples[comps == 2] = expl_samps[  comps == 2]


# --- 사용 예시 ---
# 단일 action
# a = sample_actions(1)
# print("sampled action:", a.item())

# # 배치 (예: 5개)
# batch = sample_actions(5)
# print("sampled batch:", batch)
