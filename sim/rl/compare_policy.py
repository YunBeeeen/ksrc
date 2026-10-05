#!/usr/bin/env python3
"""Compare a saved 3-D driving-assist policy with zero residual on paired paths.

Default: unseen fixed paths on the competition terrain, without randomization.
Use --randomize for a second, noisy robustness check on the same path set.
"""
import argparse
import math
import pickle

import numpy as np
from stable_baselines3 import PPO

from config import RoverCfg
from env import HISTORY, OBS_PER_FRAME, RoverEnv
from train import stage1_paths


# cmdvel 과제의 지표.  v_err/om_err 는 무차원(v_max, om_max 로 나눔)이라
# 100 을 곱해 "명령 대비 %" 로 읽는다.
METRICS_CMDVEL = (
    ("v_err_m", "병진속도오차 평균", 100.0, "%vmax"),
    ("v_err_p90", "병진속도오차 p90", 100.0, "%vmax"),
    ("om_err_m", "요속도오차 평균", 100.0, "%ommax"),
    ("om_err_p90", "요속도오차 p90", 100.0, "%ommax"),
    ("slip_m", "슬립 평균", 1.0, ""),
    ("slip_p90", "슬립 p90", 1.0, ""),
    ("sink_m", "침하 평균", 100.0, "%zmax"),
    ("stuck", "고착", 100.0, "%"),
    ("tip", "전복", 100.0, "%"),
    ("a_use", "잔차 사용률", 100.0, "%"),
    ("a_p95", "잔차 크기 p95", 100.0, "%"),
    ("cmd_sat", "차체명령 포화", 100.0, "%"),
    ("t_elapsed", "에피소드 길이", 1.0, "s"),
)

METRICS = (
    ("success", "완주율", 100.0, "%p"),
    ("frac", "경로 진행률", 100.0, "%p"),
    ("e_y_m", "횡오차 평균", 1000.0, "mm"),
    ("e_y_p90", "횡오차 p90", 1000.0, "mm"),
    ("e_psi_m", "기수오차 평균", 1.0, "deg"),
    ("slip_m", "슬립 평균", 1.0, ""),
    ("slip_p90", "슬립 p90", 1.0, ""),
    # 침하는 슬립의 대가다 (terramech: zdot = alpha|i||v| - beta z,
    # Rn = Fn(c_r + k_z z), k_z 8.0).  2026-10-04 까지 비교표에 없어서
    # 슬립 +26% 를 쓰는 정책의 비용이 측정되지 않았다.
    ("sink_m", "침하 평균", 1000.0, "mm"),
    ("t_elapsed", "소요시간", 1.0, "s"),
    ("a_use", "잔차 사용률", 100.0, "%"),
    ("a_p95", "잔차 크기 p95", 100.0, "%"),
    ("cmd_sat", "차체명령 포화", 100.0, "%"),
    ("drive_gate", "조향 대기 비율", 100.0, "%"),
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--vecnorm", required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--terrain", choices=("sand", "rock"), default="sand")
    p.add_argument("--path-seed", type=int, default=4321,
                   help="holdout path seed; training default is 1234")
    p.add_argument("--n-paths", type=int, default=30)
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--episode-s", type=float, default=20.0)
    p.add_argument("--difficulty", type=float, default=0.8)
    p.add_argument("--drive-align-gate-deg", type=float, default=10.0,
                   help="학습/ROS와 같은 조향 게이트 [deg], 0=이전 동작")
    p.add_argument("--min-slope-deg", type=float, default=0.0,
                   help="경로가 지나는 평균 지형경사의 하한 [deg]. 0 이면 기존 동작(경사 무시). 2026-10-04 측정: 하한이 없으면 경로 경사 중앙값이 0도라 견인력 수요가 R/N=0.057 뿐이고, 바퀴별 mu 를 +-16%% 흔들어도 목적 현상이 발생하지 않는다. 8 이면 실제 평균 9.5도(수요 0.165)로 가용 mu(i) 0.17~0.39 와 같은 범위가 된다")
    p.add_argument("--path-d-ep", type=float, default=0.0,
                   help="make_path 의 급사면 선호 확률. min-slope-deg 와 함께 쓴다 (0.9 권장)")
    p.add_argument("--soil-amp", type=float, default=None,
                   help="모래 비대칭 장(바퀴별 mu 배율) 진폭. **이 RL 의 과제 본체**다 (terrain.soil_field 참고). --stage1 은 randomize 를 통째로 꺼서 이것까지 끄므로, Stage 1 에서 목적 현상을 쓰려면 이 값을 명시한다. 미지정이면 기존 동작(randomize 에 종속). 0.25 권장, 0.5 는 장이 음수가 된다")
    p.add_argument("--randomize", action="store_true",
                   help="enable motor/sensor/soil randomization and noise")
    # 기준선과 정책에 **같은** lookahead 를 적용한다.  정책은 cmd_nom 위에
    # 잔차를 얹으므로 lookahead 가 바뀌면 정책이 보는 명령도 바뀐다 -- 즉
    # 이것은 "튜닝된 기준선 위에서도 RL 이 뭘 더하는가" 를 묻는 노브다.
    # 2026-10-04 스윕: 순수 IK 가 L=0.264 에서 20.72mm, L=0.12 에서 14.81mm.
    # L 하나가 5.91mm 를 움직이는데 RL 이 주장한 개선은 1.88mm 였다.
    p.add_argument("--task", default="path", choices=("path", "cmdvel"),
                   help="학습 때와 **같은** 값이어야 한다. cmdvel 은 경로 지표 "
                        "대신 속도추종 지표를 낸다.")
    p.add_argument("--pp-k-v", type=float, default=None,
                   help="pure pursuit lookahead 속도계수 (기본 1.2)")
    p.add_argument("--pp-l-min", type=float, default=None,
                   help="lookahead 하한 [m] (기본 0.18)")
    p.add_argument("--pp-l-max", type=float, default=None,
                   help="lookahead 상한 [m] (기본 0.45)")
    args = p.parse_args()
    if args.episodes < 1 or args.n_paths < 1:
        p.error("episodes and n-paths must be positive")
    if not math.isfinite(args.drive_align_gate_deg) or args.drive_align_gate_deg < 0:
        p.error("--drive-align-gate-deg 는 0 이상의 유한한 숫자여야 합니다")

    model = PPO.load(args.model, device="cpu")
    if model.action_space.shape != (3,):
        p.error(f"expected a 3-D residual policy, got {model.action_space.shape}")
    obs_dim = int(model.observation_space.shape[0])
    regular_dim = OBS_PER_FRAME * HISTORY
    if obs_dim < regular_dim:
        p.error(f"model observation has {obs_dim} values; current env needs {regular_dim}")
    with open(args.vecnorm, "rb") as f:
        norm = pickle.load(f)
    if norm.obs_rms.mean.shape != (obs_dim,):
        p.error("VecNormalize statistics do not match the model observation")

    over = {}
    for flag in ("pp_k_v", "pp_l_min", "pp_l_max"):
        v = getattr(args, flag)
        if v is not None:
            if not math.isfinite(v) or v <= 0:
                p.error(f"--{flag.replace('_', '-')} 는 양의 유한한 값이어야 합니다")
            over[flag] = float(v)
    _base = RoverCfg()
    if over.get("pp_l_min", _base.pp_l_min) > over.get("pp_l_max", _base.pp_l_max):
        p.error("--pp-l-min 이 --pp-l-max 보다 큽니다")
    cfg = RoverCfg(**over) if over else None

    paths = stage1_paths(args.terrain, args.path_seed, args.n_paths,
                         min_slope_deg=args.min_slope_deg, d_ep=args.path_d_ep)
    env_args = dict(cfg=cfg, task=args.task,
                    difficulty=args.difficulty, episode_s=args.episode_s,
                    arena_eval=True, eval_kind=args.terrain, paths=paths,
                    randomize=args.randomize, privileged=obs_dim > regular_dim,
                    drive_align_gate_deg=args.drive_align_gate_deg,
                    soil_amp=args.soil_amp)
    base_env = RoverEnv(**env_args, seed=args.seed0)
    rl_env = RoverEnv(**env_args, seed=args.seed0)
    if base_env.observation_space.shape != (obs_dim,):
        p.error(f"model observation has {obs_dim} values but current env has "
                f"{base_env.observation_space.shape[0]}")

    def episode(env, seed, assist):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            if assist:
                normalized = np.clip(
                    (obs - norm.obs_rms.mean) / np.sqrt(norm.obs_rms.var + 1e-8),
                    -10, 10).astype(np.float32)
                action = model.predict(normalized, deterministic=True)[0]
            else:
                action = np.zeros(3, dtype=np.float32)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        return info

    base, rl = [], []
    # soil_amp 미지정이면 env 가 randomize 에 종속시켜 cfg.soil_amp 를 쓴다.
    # 헤더에 유효값을 찍지 않으면 soil 0.25 와 0 을 구분할 수 없다 (baseline.py
    # 의 라벨 버그와 같은 뿌리).
    _soil_eff = RoverCfg().soil_amp if args.randomize else 0.0
    print(f"{args.terrain} | holdout paths seed={args.path_seed} | "
          f"randomize={args.randomize} | episodes={args.episodes} | "
          f"drive_align_gate_deg={args.drive_align_gate_deg} | "
          f"soil_amp={args.soil_amp}"
          f"{'' if args.soil_amp is not None else f' -> 유효 {_soil_eff}'} | "
          f"lookahead k_v={over.get('pp_k_v', _base.pp_k_v)} "
          f"L={over.get('pp_l_min', _base.pp_l_min):.2f}"
          f"~{over.get('pp_l_max', _base.pp_l_max):.2f}m",
          flush=True)
    for i in range(args.episodes):
        seed = args.seed0 + i
        base.append(episode(base_env, seed, False))
        rl.append(episode(rl_env, seed, True))
        if (i + 1) % 10 == 0 or i + 1 == args.episodes:
            print(f"{i + 1}/{args.episodes} pairs complete", flush=True)

    print(f"{'지표':<17} {'순수 IK':>10} {'보조 정책':>10} "
          f"{'차이(RL-IK)':>14} {'95% CI (짝 bootstrap)':>24}")
    rng = np.random.default_rng(0)
    resamples = rng.integers(0, args.episodes, size=(10000, args.episodes))
    for key, label, scale, unit in (METRICS_CMDVEL if args.task == 'cmdvel'
                                    else METRICS):
        b = np.array([v[key] for v in base], dtype=float) * scale
        r = np.array([v[key] for v in rl], dtype=float) * scale
        delta = r - b
        lo, hi = np.percentile(delta[resamples].mean(axis=1), (2.5, 97.5))
        print(f"{label:<17} {np.mean(b):10.2f} {np.mean(r):10.2f} "
              f"{np.mean(delta):+13.2f} [{lo:+.2f}, {hi:+.2f}] {unit}")
    wins = sum(not b["success"] and r["success"] for b, r in zip(base, rl))
    losses = sum(b["success"] and not r["success"] for b, r in zip(base, rl))
    discordant = wins + losses
    p_exact = min(1.0, 2.0 * sum(math.comb(discordant, k)
                               for k in range(min(wins, losses) + 1))
                  / (2 ** discordant)) if discordant else 1.0
    print(f"성공 짝비교 RL만/IK만: {wins}/{losses}; "
          f"McNemar exact p={p_exact:.3f}")
    print("같은 seed와 경로를 짝지었지만, 주행 결과가 갈라지면 이후 노이즈의 "
          "난수 소비는 완전히 같지 않을 수 있습니다.")


if __name__ == "__main__":
    main()
