# validate_dynamics_fixed.py
import os, json, math, argparse, csv
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt

from metaurban import SidewalkStaticMetaUrbanEnv
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.depth_camera import DepthCamera
from metaurban.component.sensors.semantic_camera import SemanticCamera
from metaurban.obs.mix_obs import ThreeSourceMixObservation

def unwrap(th): return np.unwrap(th)
def kasa(x,y):
    A=np.column_stack([2*x,2*y,np.ones_like(x)]); b=x**2+y**2
    c,*_=np.linalg.lstsq(A,b,rcond=None); cx,cy,d=c; R=float(np.sqrt(cx*cx+cy*cy+d)); return float(cx),float(cy),R
def refine(x,y,cx,cy,R,it=30):
    for _ in range(it):
        dx=x-cx; dy=y-cy; dist=np.hypot(dx,dy); res=dist-R
        dist[dist==0]=1e-9
        J=np.column_stack([-dx/dist,-dy/dist,-np.ones_like(dist)])
        JTJ=J.T@J; JTr=J.T@res
        try: dc=-np.linalg.solve(JTJ,JTr)
        except np.linalg.LinAlgError: break
        cx+=dc[0]; cy+=dc[1]; R+=dc[2]
        if np.linalg.norm(dc)<1e-9: break
    return float(cx),float(cy),float(R)
def fit_circle(x,y):
    cx0,cy0,R0=kasa(x,y); return (*refine(x,y,cx0,cy0,R0),)

def ensure_dir(d): os.makedirs(d,exist_ok=True); return d

def make_env(map_type='X', render=False, seed=4):
    cfg=dict(
        crswalk_density=1, object_density=0.01, use_render=render,
        walk_on_all_regions=False, map=map_type, manual_control=False,  # 키보드 off
        drivable_area_extension=55, height_scale=1, spawn_deliveryrobot_num=2,
        show_mid_block_map=False, show_ego_navigation=False, debug=False, horizon=600,
        on_continuous_line_done=False, out_of_route_done=True,
        vehicle_config=dict(show_lidar=False, show_navi_mark=True, show_line_to_navi_mark=False,
                            show_dest_mark=False, enable_reverse=True, policy_reverse=False),
        show_sidewalk=True, show_crosswalk=True, random_spawn_lane_index=False,
        num_scenarios=100, accident_prob=0, window_size=(1200,900),
        relax_out_of_road_done=True, max_lateral_dist=1e10,
        camera_dist=0.8, camera_height=1.5, camera_pitch=None, camera_fov=66, norm_pixel=False,
        image_observation=True,
        sensors=dict(
            rgb_camera=(RGBCamera,640,360),
            depth_camera=(DepthCamera,640,360),
            semantic_camera=(SemanticCamera,640,360),
            top_down_semantic=(SemanticCamera,512,512),
        ),
        agent_observation=ThreeSourceMixObservation, interface_panel=[],
        # autopilot/전문가 비활성 (키가 없을 수도 있으니 아래에서 한 번 더 시도)
        expert_takeover=False
    )
    env=SidewalkStaticMetaUrbanEnv(cfg); env.reset(seed=seed)
    # 가능한 모든 경로로 autopilot 끄기
    for key in ["expert_takeover","use_expert","expert_mode","default_controller"]:
        try:
            if key=="default_controller": 
                if key in env.config: env.config[key]="external"
            else:
                env.config[key]=False
        except: pass
    try: env.agent.expert_takeover=False
    except: pass
    try: env.engine.global_config["expert_takeover"]=False
    except: pass
    return env

def record_run(env, steer_idx, speed_idx, steer, speed, T, hz=20):
    xs=[]; ys=[]; th=[]; N=int(T*hz)
    act=[0.0,0.0]
    for _ in range(N):
        act[steer_idx]=steer; act[speed_idx]=speed
        _,_,tm,tc,_=env.step(act)
        xs.append(float(env.agent.position[0]))
        ys.append(float(env.agent.position[1]))
        th.append(float(env.agent.heading_theta))
        if tm or tc: break
    t=np.arange(len(xs))/hz
    return dict(x=np.array(xs),y=np.array(ys),th=np.array(th),t=t)

def mean_v_omega(tr):
    x,y,th,t=tr["x"],tr["y"],unwrap(tr["th"]),tr["t"]
    if len(x)<3: return 0.0,0.0
    ds=np.hypot(np.diff(x),np.diff(y)); dt=np.diff(t); dt[dt==0]=1e-6
    v=float(np.mean(ds/dt)); omega=float(np.mean(np.diff(th)/dt))
    return v, omega

def sniff_axes(env, test_mag=0.6, speed=0.6, T=2.0):
    # 축 0 테스트
    tr0=record_run(env,0,1,test_mag,speed,T); v0,o0=mean_v_omega(tr0)
    # 재설정 후 축 1 테스트
    env.reset(seed=env.current_seed+1)
    tr1=record_run(env,1,0,test_mag,speed,T); v1,o1=mean_v_omega(tr1)
    # 어느 축이 조향인가?
    if abs(o0)>abs(o1): steer_idx,speed_idx,sign=np.array([0,1,np.sign(o0)])
    else: steer_idx,speed_idx,sign=np.array([1,0,np.sign(o1)])
    # 효과성 확인
    if max(abs(o0),abs(o1))<1e-3:
        raise RuntimeError("액션이 헤딩에 영향을 주지 않습니다. autopilot이 켜져 있거나 액션 스케일이 0입니다.")
    return int(steer_idx), int(speed_idx), float(sign)

@dataclass
class RunRes:
    a_cmd: float; speed_cmd: float; v_mean: float; R_meas: float; R_pred: float
    rmse_circle: float; max_res: float; kappa_sign_match: int

def fit_and_eval(env, steer_idx, speed_idx, a, speed, T, R0, sign_gain, hz=20):
    tr=record_run(env,steer_idx,speed_idx,steer=a,speed=speed,T=T,hz=hz)
    x,y=tr["x"],tr["y"]
    cx,cy,R=fit_circle(x,y); dist=np.hypot(x-cx,y-cy); res=dist-R
    rmse=float(np.sqrt(np.mean(res**2))); max_abs=float(np.max(np.abs(res)))
    v,omega=mean_v_omega(tr)
    # 부호 체크: 실제 곡률 부호 = sign(omega/v) * sign_gain (환경 좌표계→조향축 부호 보정)
    sign_ok=int(np.sign((omega/(v+1e-9))*sign_gain)*np.sign(a))
    R_pred=math.inf if a==0 else abs(R0/ a)
    return RunRes(a_cmd=a, speed_cmd=speed, v_mean=v, R_meas=R, R_pred=R_pred,
                  rmse_circle=rmse, max_res=max_abs, kappa_sign_match=sign_ok), (x,y,cx,cy,R)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--time_per_run',type=float,default=6.0)
    ap.add_argument('--steers',type=float,nargs='+',default=[0.2,0.4,0.6,0.8])
    ap.add_argument('--speed',type=float,default=1.0)
    ap.add_argument('--seed',type=int,default=4)
    ap.add_argument('--render',action='store_true')
    ap.add_argument('--outdir',type=str,default='results_fixed')
    args=ap.parse_args()
    outdir=ensure_dir(args.outdir)

    env=make_env(render=args.render, seed=args.seed)
    print("[INFO] sniffing action axes...")
    steer_idx, speed_idx, sign_gain = sniff_axes(env)
    print(f"[INFO] steering axis = {steer_idx}, speed axis = {speed_idx}, sign_gain = {sign_gain:+.0f}")

    # 보정: a0 = -1
    env.reset(seed=env.current_seed+1)
    calib, (x0,y0,cx0,cy0,R0) = fit_and_eval(env, steer_idx, speed_idx, a=-1.0, speed=args.speed,
                                             T=args.time_per_run, R0=3.0, sign_gain=sign_gain)
    R0 = calib.R_meas
    print(f"[CALIB] a0=-1 ⇒ R0 = {R0:.4f} m, v≈{calib.v_mean:.3f} m/s, sign_match={calib.kappa_sign_match:+d}")

    def save_traj(x,y,cx,cy,R,title,fname):
        th=np.linspace(0,2*np.pi,512); X=cx+R*np.cos(th); Y=cy+R*np.sin(th)
        plt.figure(figsize=(6,6)); plt.plot(x,y,'.',ms=2); plt.plot(X,Y,'-'); plt.axis('equal')
        plt.title(title); plt.tight_layout(); plt.savefig(os.path.join(outdir,fname)); plt.close()

    save_traj(x0,y0,cx0,cy0,R0,f'Calib a=-1, R0={R0:.3f}m','calib.png')

    # 시험 조향
    results=[calib]; A=[-1.0]; Rm=[R0]
    for a in args.steers:
        env.reset(seed=env.current_seed+1)
        r,(x,y,cx,cy,R)=fit_and_eval(env, steer_idx, speed_idx, a=a, speed=args.speed,
                                     T=args.time_per_run, R0=R0, sign_gain=sign_gain)
        results.append(r); A.append(a); Rm.append(R)
        print(f"[RUN] a={a:+.2f}: R_meas={R:.3f} m, R_pred={r.R_pred:.3f} m, "
              f"v≈{r.v_mean:.3f} m/s, sign_match={r.kappa_sign_match:+d}")
        save_traj(x,y,cx,cy,R,f'a={a:.2f}, Rm={R:.3f}m','run_a_{:.2f}.png'.format(a))

    # 모델 적합
    A=np.array(A); K=1.0/np.array(Rm)
    beta=np.linalg.lstsq(A.reshape(-1,1),K,rcond=None)[0][0]                 # kappa = beta * a
    A_mat=np.column_stack([A, A**3]); b1,b3=np.linalg.lstsq(A_mat,K,rcond=None)[0]
    beta_theory = -1.0/R0 * sign_gain                                       # 부호 보정 반영

    print(f"[MODEL] linear beta_fit = {beta:.6f} [1/m], beta_theory = {beta_theory:.6f} [1/m]")
    print(f"[MODEL] cubic  kappa(a) = {b1:.6f}*a + {b3:.6f}*a^3")

    # 저장
    with open(os.path.join(outdir,'report.json'),'w') as f:
        json.dump({
            "axes": {"steer_idx":steer_idx,"speed_idx":speed_idx,"sign_gain":sign_gain},
            "calibration":{"a0":-1.0,"R0_meas":R0,"beta_theory":beta_theory},
            "linear":{"beta_fit":float(beta)},
            "cubic":{"beta1":float(b1),"beta3":float(b3)},
            "runs":[asdict(r) for r in results]
        }, f, indent=2)

    with open(os.path.join(outdir,'runs.csv'),'w',newline='') as f:
        import csv
        writer=csv.DictWriter(f,fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader(); [writer.writerow(asdict(r)) for r in results]

    # 곡률-조향 곡선 그림
    a_grid=np.linspace(-1,1,201)
    k_lin=beta*a_grid; k_th=beta_theory*a_grid; k_cub=b1*a_grid+b3*a_grid**3
    plt.figure(); plt.plot(a_grid,k_lin,label=f'linear fit {beta:.4f}')
    plt.plot(a_grid,k_th,'--',label=f'linear theory {beta_theory:.4f}')
    plt.plot(a_grid,k_cub,label=f'cubic {b1:.4f},{b3:.4f}')
    plt.scatter(A,K,s=20,label='measured κ')
    plt.xlabel('steer command a'); plt.ylabel('curvature κ [1/m]')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,'kappa_vs_a.png')); plt.close()

    env.close(); print(f"[DONE] saved to {outdir}/")

if __name__=="__main__":
    main()
