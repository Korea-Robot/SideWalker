import numpy as np
import matplotlib.pyplot as plt

# --- 파라미터 설정 ---
policy_mean, policy_std = 0.2, 0.1
prior_mean,  prior_std  = -0.3, 0.2
expl_low,    expl_high  = -1.0, 1.0
weights = np.array([0.5, 0.3, 0.2])  # [policy, prior, exploration]

# --- 샘플링 ---
N = 300000
components = np.random.choice(3, size=N, p=weights)
samples = np.empty(N)
samples[components == 0] = np.random.normal(policy_mean, policy_std, size=(components == 0).sum())
samples[components == 1] = np.random.normal(prior_mean, prior_std, size=(components == 1).sum())
samples[components == 2] = np.random.uniform(expl_low, expl_high, size=(components == 2).sum())

# --- 이론적 PDF 계산 --- 
x = np.linspace(-1.5, 1.5, 1000)

# 1) policy Gaussian PDF:  p₁(x) = 1/(σ₁√2π) · exp[-½((x-μ₁)/σ₁)²]
pdf_policy = weights[0] * (
    1/(policy_std * np.sqrt(2*np.pi))
    * np.exp(-0.5 * ((x - policy_mean)/policy_std)**2)
)

# 2) prior  Gaussian PDF:  p₂(x) = 1/(σ₂√2π) · exp[-½((x-μ₂)/σ₂)²]
pdf_prior = weights[1] * (
    1/(prior_std * np.sqrt(2*np.pi))
    * np.exp(-0.5 * ((x - prior_mean)/prior_std)**2)
)

# 3) exploration Uniform PDF: p₃(x) = 1/(b-a)   for a ≤ x ≤ b, else 0
pdf_expl = weights[2] * (
    1/(expl_high - expl_low)
    * ((x >= expl_low) & (x <= expl_high)).astype(float)
)

# 4) Mixture PDF: p_mix(x) = w₁·p₁(x) + w₂·p₂(x) + w₃·p₃(x)
pdf_mix = pdf_policy + pdf_prior + pdf_expl

# --- 시각화 (크기 조정 및 레이블 추가) ---
plt.figure(figsize=(12, 7))
plt.hist(samples, bins=200, density=True, alpha=0.3, label='Samples')

plt.plot(x, pdf_policy, linestyle='--', linewidth=2, label='Policy (Gaussian)')
plt.plot(x, pdf_prior,  linestyle='--', linewidth=2, label='Prior (Gaussian)')
plt.plot(x, pdf_expl,   linestyle='--', linewidth=2, label='Exploration (Uniform)')
plt.plot(x, pdf_mix,    linestyle='-',  linewidth=3, label='Mixture')

plt.title("Mixture Distribution Visualization", fontsize=16)
plt.xlabel("Action value", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig("Distribution_Mixture.png")
plt.show()

# --- 로그 확률 계산 예시 --- 
# 임의의 액션 a₀에 대해: log p_mix(a₀) = log[ w₁·p₁(a₀) + w₂·p₂(a₀) + w₃·p₃(a₀) ]

def mixture_log_prob(a):
    # a: scalar or numpy array of action values
    # 재사용을 위해 PDF 함수로 분리
    p1 = (1/(policy_std * np.sqrt(2*np.pi))
          * np.exp(-0.5 * ((a - policy_mean)/policy_std)**2))
    p2 = (1/(prior_std  * np.sqrt(2*np.pi))
          * np.exp(-0.5 * ((a - prior_mean)/prior_std)**2))
    p3 = (1/(expl_high - expl_low)) * ((a >= expl_low) & (a <= expl_high))
    
    pmix = weights[0]*p1 + weights[1]*p2 + weights[2]*p3
    return np.log(pmix + 1e-12),pmix  # underflow 방지 작은 상수 추가

# 예: 몇 개 액션에 대한 로그 확률
actions = np.array([-1.0, 0.0, 0.2, 0.5])
log_probs,probs = mixture_log_prob(actions)
print("Actions:", actions)
print("probabilities:", probs)
print("Log probabilities:", log_probs)
