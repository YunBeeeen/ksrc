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

from env import HISTORY, OBS_PER_FRAME, RoverEnv
from train import stage1_paths


METRICS = (
    ("success", "완주율", 100.0, "%p"),
    ("frac", "경로 진행률", 100.0, "%p"),
    ("e_y_m", "횡오차 평균", 1000.0, "mm"),
    ("e_y_p90", "횡오차 p90", 1000.0, "mm"),
    ("e_psi_m", "기수오차 평균", 1.0, "deg"),
    ("slip_m", "슬립 평균", 1.0, ""),
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
    p.add_argument("--randomize", action="store_true",
                   help="enable motor/sensor/soil randomization and noise")
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

    paths = stage1_paths(args.terrain, args.path_seed, args.n_paths)
    env_args = dict(difficulty=args.difficulty, episode_s=args.episode_s,
                    arena_eval=True, eval_kind=args.terrain, paths=paths,
                    randomize=args.randomize, privileged=obs_dim > regular_dim,
                    drive_align_gate_deg=args.drive_align_gate_deg)
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
    print(f"{args.terrain} | holdout paths seed={args.path_seed} | "
          f"randomize={args.randomize} | episodes={args.episodes} | "
          f"drive_align_gate_deg={args.drive_align_gate_deg}", flush=True)
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
    for key, label, scale, unit in METRICS:
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
