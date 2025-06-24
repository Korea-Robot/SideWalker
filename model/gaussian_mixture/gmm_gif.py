import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# --- 파라미터 설정 ---
policy_mean_start, policy_std_start = 0.5, 1
prior_mean,  prior_std  = -0.3, 0.2
expl_low,    expl_high  = -1.0, 1.0
exploration_weight = 0.2

initial_policy_weight = 0.1
final_policy_weight   = 0.7
initial_prior_weight  = 0.7
final_prior_weight    = 0.1

n_frames = 50
x = np.linspace(-1.5, 1.5, 1000)

# 기초 PDF
pdf_policy_base = (1/(policy_std_start * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - policy_mean_start)/policy_std_start)**2)
pdf_prior_base  = (1/(prior_std  * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - prior_mean) /prior_std )**2)
pdf_expl_base   = (1/(expl_high - expl_low)) * ((x >= expl_low) & (x <= expl_high))

# 전 프레임에서 최대 PDF 계산
max_pdf = 0
for i in range(n_frames):
    t = i / (n_frames - 1)
    w_p = initial_policy_weight + t * (final_policy_weight - initial_policy_weight)
    w_pr = initial_prior_weight  + t * (final_prior_weight   - initial_prior_weight)
    pdf_m = w_p * pdf_policy_base + w_pr * pdf_prior_base + exploration_weight * pdf_expl_base
    max_pdf = max(max_pdf, pdf_m.max())
ylim_max = max_pdf * 1.1  # 10% 여유

# 플롯 초기화
fig, ax = plt.subplots(figsize=(12, 7))
line_p, = ax.plot([], [], '--', linewidth=2, label='Policy')
line_pr, = ax.plot([], [], '--', linewidth=2, label='Prior')
line_e, = ax.plot([], [], '--', linewidth=2, label='Exploration')
line_mix, = ax.plot([], [], '-', linewidth=3, label='Mixture')

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(0, ylim_max)
ax.set_xlabel('Action value')
ax.set_ylabel('Density')
ax.legend(fontsize=12)
ax.grid(True)

def init():
    for line in (line_p, line_pr, line_e, line_mix):
        line.set_data([], [])
    return line_p, line_pr, line_e, line_mix

def animate(i):
    t = i / (n_frames - 1)
    w_p = initial_policy_weight + t * (final_policy_weight - initial_policy_weight)
    w_pr = initial_prior_weight  + t * (final_prior_weight   - initial_prior_weight)
    w_e = exploration_weight

    # 계속 업데이트 되는 policy gradient 분포
    policy_std=policy_std_start*10/(i+10)
    policy_mean =policy_mean_start-i/80
    pdf_policy_base = (1/(policy_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - policy_mean)/policy_std)**2)
    
    pdf_p = w_p  * pdf_policy_base
    pdf_pr = w_pr * pdf_prior_base
    pdf_e = w_e  * pdf_expl_base
    pdf_m = pdf_p + pdf_pr + pdf_e
    
    line_p.set_data(x, pdf_p)
    line_pr.set_data(x, pdf_pr)
    line_e.set_data(x, pdf_e)
    line_mix.set_data(x, pdf_m)
    ax.set_title(f'Frame {i+1}/{n_frames} — weights(policy={w_p:.2f}, prior={w_pr:.2f}, expl={w_e:.2f})')
    return line_p, line_pr, line_e, line_mix

anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=100, blit=True)
anim.save('mixtureD.gif', writer=PillowWriter(fps=10))

print("GIF saved to /mixture_transition.gif")
