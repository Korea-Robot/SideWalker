import torch
import matplotlib.pyplot as plt

# --- 파라미터 설정 ---
policy_mean = 0.2
policy_std  = 0.1
prior_mean  = -0.3
prior_std   = 0.2
expl_low    = -1.0
expl_high   = 1.0
weights     = torch.tensor([0.5, 0.3, 0.2])  # [policy, prior, exploration], 합 = 1

# --- 샘플링 ---
N = 300_000
# 컴포넌트 인덱스 샘플링 (0:policy, 1:prior, 2:expl)
components = torch.multinomial(weights, num_samples=N, replacement=True)
samples = torch.empty(N)

# 각 컴포넌트별로 샘플 할당
mask_p  = components == 0
mask_pr = components == 1
mask_e  = components == 2

# Normal 샘플 (policy, prior)과 Uniform 샘플 (exploration)
samples[mask_p]  = torch.normal(policy_mean, policy_std,  size=(mask_p.sum().item(),))
samples[mask_pr] = torch.normal(prior_mean,  prior_std,   size=(mask_pr.sum().item(),))
samples[mask_e]  = torch.distributions.Uniform(expl_low, expl_high).sample((mask_e.sum().item(),))

# --- 이론적 PDF 계산 ---
x = torch.linspace(-1.5, 1.5, 1000)

# Gaussian PDF: 1/(σ√2π) * exp[-½((x-μ)/σ)²]
const_p  = 1.0/(policy_std * torch.sqrt(torch.tensor(2*torch.pi)))
const_pr = 1.0/(prior_std  * torch.sqrt(torch.tensor(2*torch.pi)))
pdf_p  = weights[0] * const_p  * torch.exp(-0.5 * ((x - policy_mean)/policy_std)**2)
pdf_pr = weights[1] * const_pr * torch.exp(-0.5 * ((x - prior_mean)/prior_std )**2)

# Uniform PDF: 1/(b-a) on [a,b]
pdf_e  = weights[2] * (1.0/(expl_high - expl_low)) * ((x>=expl_low) & (x<=expl_high)).float()

pdf_mix = pdf_p + pdf_pr + pdf_e

# --- 시각화 ---
plt.figure(figsize=(12,7))
# 히스토그램 (토치 → 넘파이 변환)
plt.hist(samples.numpy(), bins=200, density=True, alpha=0.3, label='Samples')
# 개별 분포 곡선
plt.plot(x.numpy(), pdf_p.numpy(),  '--', linewidth=2, label='Policy (Gaussian)')
plt.plot(x.numpy(), pdf_pr.numpy(), '--', linewidth=2, label='Prior (Gaussian)')
plt.plot(x.numpy(), pdf_e.numpy(),  '--', linewidth=2, label='Exploration (Uniform)')
# 혼합 분포
plt.plot(x.numpy(), pdf_mix.numpy(), '-',  linewidth=3, label='Mixture')

plt.title("Mixture Distribution Visualization", fontsize=16)
plt.xlabel("Action value", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig("Distribution_Mixture_torch.png")
plt.show()

# --- 로그 확률 계산 함수 ---
def mixture_log_prob(a):
    """
    a: torch.Tensor of action values
    returns: (log_probs, probs) 각 요소별 mixture 확률과 로그확률
    """
    # compute component PDFs at a
    p1 = const_p * torch.exp(-0.5 * ((a - policy_mean)/policy_std)**2)
    p2 = const_pr * torch.exp(-0.5 * ((a - prior_mean)/prior_std )**2)
    p3 = (1.0/(expl_high - expl_low)) * ((a>=expl_low)&(a<=expl_high)).float()
    
    # mixture 확률
    pmix = weights[0]*p1 + weights[1]*p2 + weights[2]*p3
    log_pmix = torch.log(pmix + 1e-12)  # underflow 방지
    return log_pmix, pmix

# 예시: 몇 개 액션에 대한 로그확률
actions = torch.tensor([-1.0, 0.0, 0.2, 0.5])
log_probs, probs = mixture_log_prob(actions)
print("Actions:", actions)
print("Probabilities:", probs)
print("Log probabilities:", log_probs)
