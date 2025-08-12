import os, json, math, argparse, csv
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.monitor import Monitor
from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera
from metaurban.obs.mix_obs import ThreeSourceMixObservation

# ============== 수학 유틸 ==============

def unwrap(theta_arr):
    """[-pi, pi) 래핑된 heading을 연속 각도로 변환"""
    return np.unwrap(theta_arr)

def path_length(xs, ys):
    dx = np.diff(xs); dy = np.diff(ys)
    return float(np.sum(np.hypot(dx, dy)))

def kasa_circle_fit(x, y):
    """Kåsa algebraic circle fit"""
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x**2 + y**2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, d = c
    R = np.sqrt(cx**2 + cy**2 + d)
    return float(cx), float(cy), float(R)

def refine_circle_gn(x, y, cx, cy, R, iters=20):
    """Gauss-Newton refinement for circle parameters"""
    for _ in range(iters):
        dx = x - cx; dy = y - cy
        dist = np.hypot(dx, dy)
        ri = dist - R
        dist_safe = np.where(dist == 0, 1e-9, dist)
        J = np.column_stack([-dx/dist_safe, -dy/dist_safe, -np.ones_like(dist)])
        JTJ = J.T @ J
        JTr = J.T @ ri
        try:
            delta = -np.linalg.solve(JTJ, JTr)
        except np.linalg.LinAlgError:
            break
        cx += delta[0]; cy += delta[1]; R += delta[2]
        if np.linalg.norm(delta) < 1e-9:
            break
    return float(cx), float(cy), float(R)

def fit_circle_and_errors(x, y):
    cx0, cy0, R0 = kasa_circle_fit(x, y)
    cx, cy, R = refine_circle_gn(x, y, cx0, cy0, R0, iters=30)
    dist = np.hypot(x - cx, y - cy)
    res = dist - R
    rmse = float(np.sqrt(np.mean(res**2)))
    max_abs = float(np.max(np.abs(res)))
    return (cx, cy, R, rmse, max_abs)

def fit_kappa_linear(a_list, R_list):
    # kappa = 1/R ; kappa = beta * a  => least squares on (a, kappa)
    a = np.array(a_list); kappa = 1.0/np.array(R_list)
    beta = float(np.linalg.lstsq(a.reshape(-1,1), kappa, rcond=None)[0][0])
    return beta

def fit_kappa_cubic(a_list, R_list):
    a = np.array(a_list); kappa = 1.0/np.array(R_list)
    A = np.column_stack([a, a**3])
    coef, *_ = np.linalg.lstsq(A, kappa, rcond=None)
    beta1, beta3 = coef
    return float(beta1), float(beta3)

# ============== 환경 구성 ==============

def make_env(map_type='X', use_render=True, seed=4):
    config = dict(
        crswalk_density=1,
        object_density=0.01,
        use_render=use_render,
        walk_on_all_regions=False,
        map=map_type,
        manual_control=True,
        drivable_area_extension=55,
        height_scale=1,
        spawn_deliveryrobot_num=2,
        show_mid_block_map=False,
        show_ego_navigation=False,
        debug=False,
        horizon=300,
        on_continuous_line_done=False,
        out_of_route_done=True,
        vehicle_config=dict(
            show_lidar=False,
            show_navi_mark=True,
            show_line_to_navi_mark=False,
            show_dest_mark=False,
            enable_reverse=True,
            policy_reverse=False,
        ),
        show_sidewalk=True,
        show_crosswalk=True,
        random_spawn_lane_index=False,
        num_scenarios=100,
        accident_prob=0,
        window_size=(1200, 900),
        relax_out_of_road_done=True,
        max_lateral_dist=1e10,
        camera_dist = 0.8,
        camera_height = 1.5,
        camera_pitch = None,
        camera_fov = 66,
        norm_pixel=False,
        image_observation=True,
        sensors=dict(
            rgb_camera=(RGBCamera, 640, 360),
            depth_camera=(DepthCamera, 640, 360),
            semantic_camera=(SemanticCamera, 640, 360),
            top_down_semantic=(SemanticCamera, 512, 512),
        ),
        agent_observation=ThreeSourceMixObservation,
        interface_panel=[]
    )
    env = SidewalkStaticMetaUrbanEnv(config)
    obs, _ = env.reset(seed=seed)
    return env

# ============== 실험 루틴 ==============

@dataclass
class RunResult:
    a_cmd: float
    throttle_cmd: float
    dt_mean: float
    v_mean: float
    R_meas_circle: float
    R_pred_model: float
    rmse_circle_m: float
    max_abs_res_m: float
    sign_check: float  # sign of measured curvature vs a

def run_constant_steer(env, a: float, throttle: float, time_horizon: float) -> Dict[str, np.ndarray]:
    """상수 조향/스로틀로 주행하며 궤적, 헤딩, 시간 기록"""
    xs, ys, ths, ts = [], [], [], []
    t = 0.0
    # 첫 state 기록
    xs.append(float(env.agent.position[0]))
    ys.append(float(env.agent.position[1]))
    ths.append(float(env.agent.heading_theta))
    ts.append(t)

    # env가 고정 dt를 쓴다고 가정: 한 step에 동일 시간 경과
    # 확실치 않으면 time 모듈로 실시간 측정 대신 step 카운트 기반 dt_est 사용
    steps = 0
    done = False
    while t < time_horizon and not done:
        o, r, tm, tc, info = env.step([a, throttle])
        steps += 1
        # 기록
        xs.append(float(env.agent.position[0]))
        ys.append(float(env.agent.position[1]))
        ths.append(float(env.agent.heading_theta))
        ts.append(t)  # 임시: 균일 dt 가정
        done = (tm or tc)

    # dt 추정: 경로 길이와 평균 속도 일관성 검사용으로 step 기반 dt_hat 사용
    dt_hat = 1.0 / 20.0  # 가정: 20 Hz. 필요 시 환경에서 끌어올 수 있으면 수정
    ts = np.arange(len(xs)) * dt_hat

    return dict(x=np.array(xs), y=np.array(ys), theta=np.array(ths), t=np.array(ts), steps=steps)

def estimate_speed_and_omega(traj):
    x, y, th, t = traj["x"], traj["y"], traj["theta"], traj["t"]
    th_u = unwrap(th)
    ds = np.hypot(np.diff(x), np.diff(y))
    dt = np.diff(t)
    # 작은 수 방어
    dt[dt==0] = 1e-6
    v = ds / dt
    dth = np.diff(th_u)
    omega = dth / dt
    return float(np.mean(v)), float(np.mean(omega)), float(np.mean(dt))

def evaluate_run(env, a, throttle, time_horizon, R0):
    traj = run_constant_steer(env, a, throttle, time_horizon)
    x, y = traj["x"], traj["y"]
    v_mean, omega_mean, dt_mean = estimate_speed_and_omega(traj)

    # 원피팅 기반 R
    cx, cy, R_circ, rmse, max_abs = fit_circle_and_errors(x, y)

    # 선형 모델 예측
    R_pred = math.inf if a==0 else abs(R0/ a)

    # 부호 검증: kappa_meas ≈ omega/v
    if v_mean < 1e-6:
        sign_check = 0.0
    else:
        kappa_meas = omega_mean / v_mean
        sign_check = np.sign(kappa_meas) * np.sign(a)  # ideally negative if beta=-1/R0

    return RunResult(
        a_cmd=float(a),
        throttle_cmd=float(throttle),
        dt_mean=dt_mean,
        v_mean=v_mean,
        R_meas_circle=R_circ,
        R_pred_model=R_pred,
        rmse_circle_m=rmse,
        max_abs_res_m=max_abs,
        sign_check=sign_check
    ), (x, y, cx, cy, R_circ)

def draw_traj_with_circle(x, y, cx, cy, R, title, save_path):
    theta = np.linspace(0, 2*np.pi, 512)
    circ_x = cx + R*np.cos(theta)
    circ_y = cy + R*np.sin(theta)
    plt.figure(figsize=(6,6))
    plt.plot(x, y, '.', markersize=2, label='traj')
    plt.plot(circ_x, circ_y, '-', label='fit circle')
    plt.axis('equal')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

# ============== 메인 ==============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_per_run', type=float, default=6.0)
    parser.add_argument('--steers', type=float, nargs='+', default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument('--throttle', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--render', action='store_true', help='enable on-screen rendering')
    parser.add_argument('--outdir', type=str, default='results')
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)

    # 0) 환경 생성 + 보정(run with a0=-1)
    env = make_env(use_render=args.render, seed=args.seed)
    calib_res, (x0, y0, cx0, cy0, R0) = evaluate_run(env, a=-1.0, throttle=args.throttle,
                                                     time_horizon=args.time_per_run, R0=3.0)  # R0 임시
    # 보정은 측정값으로:
    R0 = calib_res.R_meas_circle
    print(f'[CALIB] a0=-1: R0 = {R0:.4f} m  (mean v ≈ {calib_res.v_mean:.3f} m/s)')

    draw_traj_with_circle(x0, y0, cx0, cy0, R0,
                          title=f'Calib a=-1, R0={R0:.3f}m',
                          save_path=os.path.join(outdir, f'calib_a_-1.png'))

    results: List[RunResult] = [calib_res]
    ar_list, Rm_list = [-1.0], [R0]

    # 1) 시험 조향들 평가
    for a in args.steers:
        res, (x, y, cx, cy, Rm) = evaluate_run(env, a=a, throttle=args.throttle,
                                               time_horizon=args.time_per_run, R0=R0)
        results.append(res)
        ar_list.append(a); Rm_list.append(Rm)
        draw_traj_with_circle(x, y, cx, cy, Rm,
                              title=f'a={a:.2f}, R_meas={Rm:.3f}m, R_pred={res.R_pred_model:.3f}m',
                              save_path=os.path.join(outdir, f'run_a_{a:.2f}.png'))
        print(f'[RUN] a={a:+.2f}: R_meas={Rm:.3f} m, R_pred={res.R_pred_model:.3f} m, '
              f'v≈{res.v_mean:.3f} m/s, sign_check={res.sign_check:+.0f}')

    # 2) 선형 모델/3차 모델 적합
    beta = fit_kappa_linear(ar_list, Rm_list)  # kappa = beta * a
    beta1, beta3 = fit_kappa_cubic(ar_list, Rm_list)

    # 이론상 a0=-1 보정이면 beta ≈ -1/R0
    beta_theory = -1.0 / R0

    print(f'[MODEL] linear beta_fit = {beta:.6f} [1/m], beta_theory = {beta_theory:.6f} [1/m]')
    print(f'[MODEL] cubic  kappa(a) = {beta1:.6f}*a + {beta3:.6f}*a^3')

    # 3) 리포트 저장
    report = {
        "calibration": {
            "a0": -1.0,
            "R0_meas": R0,
            "beta_theory": beta_theory
        },
        "linear_model": {
            "kappa(a)": "beta * a",
            "beta_fit": beta
        },
        "cubic_model": {
            "kappa(a)": "beta1 * a + beta3 * a^3",
            "beta1": beta1,
            "beta3": beta3
        },
        "runs": [asdict(r) for r in results]
    }
    with open(os.path.join(outdir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    with open(os.path.join(outdir, 'runs.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # 4) 모델-데이터 곡률 비교 플롯
    A = np.linspace(-1.0, 1.0, 201)
    k_lin = beta * A
    k_lin_theory = beta_theory * A
    k_cub = beta1 * A + beta3 * (A**3)

    plt.figure()
    plt.plot(A, k_lin, label=f'linear fit: beta={beta:.4f}')
    plt.plot(A, k_lin_theory, '--', label=f'linear (theory): beta={beta_theory:.4f}')
    plt.plot(A, k_cub, label=f'cubic fit: b1={beta1:.4f}, b3={beta3:.4f}')
    plt.scatter(np.array(ar_list), 1.0/np.array(Rm_list), s=20, label='measured κ=1/R')
    plt.xlabel('steer command a'); plt.ylabel('curvature κ [1/m]')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'kappa_vs_a.png'))
    plt.close()

    env.close()
    print(f'[DONE] saved to {outdir}/')

if __name__ == '__main__':
    main()
