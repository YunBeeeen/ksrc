"""MuJoCo 경사 직진 기록기 (실물 run_policy 직진 시험과 같은 조건·지표).

  고정 명령 (vx, 0, 0) 을 T 초.  모래 = terramech.py (legacy: 바퀴별 침하 + 시간 회복, 10-18 저녁 수정 전) 기본.
  기록: 높이, 헛돎(참값), duty, 바퀴 토크 (drive_torque), 견인력/수직력 (μ_mob), terramech 내부 (전단 μ, 침하 z, 저항 R, 수직력).

  python3 slope_drive_log.py --policy ik --mu 0.35 --out /tmp/ik_035.npz
  python3 slope_drive_log.py --policy runs/isaac/.../model_1000.pt --mu 0.40 --out /tmp/v3_040.npz
"""
import argparse
import sys

import numpy as np
import torch

import sim2sim_isaac as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="ik")
    ap.add_argument("--mu", type=float, default=0.35)
    ap.add_argument("--slope-deg", type=float, default=13.5)
    ap.add_argument("--cmd", type=float, default=0.15)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--soil", choices=("legacy", "granular"), default="legacy")
    ap.add_argument("--spawn-x", type=float, default=None, help="출발 x (기본: 램프 시작 0.3 m 앞)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rover = S.MjRover("slope_sand", 0.005, rng, drive="pi", sensors="real", slope_deg=a.slope_deg, soil=a.soil)
    S.SLOPE_SPAWN_XY[:] = (S.SLOPE_X0 - 0.3 if a.spawn_x is None else a.spawn_x, 0.0)
    rover.reset()
    rover.set_mu(np.full(4, a.mu))
    if a.policy == "ik":
        policy, obs = None, None
        amap = S.ActionMap("Res8")
    else:
        env_cfg, agent_cfg = S.load_run_cfg(a.policy)
        policy = S.Policy(a.policy, agent_cfg)
        obs = S.ObsBuilder(env_cfg, agent_cfg["obs_groups"]["policy"], 0, False, rng, vo="dr")
        rov = env_cfg["actions"]["rover"]
        amap = S.ActionMap("Res8", 0, rov.get("rate_limit", 0.0), rov.get("stand_mask", False))
        policy.reset()
        obs.reset()
    amap.reset()
    cmd = np.array([a.cmd, 0.0, 0.0])
    mass = float(rover.m.body_mass.sum())
    tm = rover.tm
    L = {k: [] for k in ("t", "pos", "pitch", "duty", "wheel_w", "tau", "slip", "mu_mob", "tm_mu", "tm_z", "tm_R", "tm_Fn", "action")}
    for k in range(int(a.seconds / S.CONTROL_DT_S)):
        s = rover.state()
        if policy is not None:
            act = policy(obs(dict(cmd=cmd, s=s, amap=amap, mu_eff=rover.mu_effective(), act_delay=0, obs_delay=0)))
        else:
            act = np.zeros(8)
        rover.step(amap.process(act, cmd))
        s = rover.state()
        tau = S.drive_torque(torch.as_tensor(rover.duty), torch.as_tensor(s["wheel_w"])).numpy()
        g = s["grav_b"]
        pitch = float(np.degrees(np.arctan2(-g[0], -g[2])))
        N = mass * 9.81 * max(-g[2], 0.5)
        L["t"].append(k * S.CONTROL_DT_S)
        L["pos"].append(s["pos"].copy())
        L["pitch"].append(pitch)
        L["duty"].append(np.asarray(rover.duty, float).copy())
        L["wheel_w"].append(s["wheel_w"].copy())
        L["tau"].append(tau)
        L["slip"].append(S.wheel_slip(s))
        L["mu_mob"].append(float(np.sum(tau) / S.WHEEL_RADIUS_M / N))
        # terramech 내부 (terramech 순서 fl, rl, fr, rr -> 모듈 순서 fl, fr, rl, rr)
        o = [0, 2, 1, 3]
        L["tm_mu"].append(rover.m.geom_friction[tm.wg, 0][o].copy())
        L["tm_z"].append(np.asarray(tm.z)[o].copy())
        L["tm_R"].append(np.asarray(tm.R)[o].copy())
        L["tm_Fn"].append(np.asarray(tm.Fn)[o].copy())
        L["action"].append(np.asarray(act, float).copy())
    np.savez_compressed(a.out, **{k: np.asarray(v) for k, v in L.items()}, mass=mass, mu=a.mu, slope_deg=a.slope_deg,
                        policy=a.policy, soil=a.soil, ramp_x0=S.SLOPE_X0)
    P = np.asarray(L["pos"])
    h = np.clip(P[:, 0] - S.SLOPE_X0, 0, S.SLOPE_LEN) * np.tan(np.radians(a.slope_deg))
    print(f"[drive] {a.policy.split('/')[-1]} μ {a.mu} {a.soil}: 60 s 높이 {100 * h[-1]:.1f} cm, 헛돎 {np.mean(np.abs(L['slip'][-1000:])):.2f}, "
          f"μ_mob {np.mean(L['mu_mob'][-1000:]):.3f}, duty {np.round(np.mean(L['duty'][-1000:], 0))}", flush=True)


if __name__ == "__main__":
    main()
