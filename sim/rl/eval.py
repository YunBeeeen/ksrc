#!/usr/bin/env python3
"""평가 -- 학습 지형이 아니라 **실제 경기장 STL** 에서 재는 게 핵심.

  python3 eval.py --model runs/ppo/final --episodes 30

비교군 (RL 이 이겨야 하는 상대):
    풀스로틀      최대 스로틀. 명령 속도를 무시하는 상한 기준선
    고정스로틀    peak 0.7 고정. 흙 상태 전체 평균이 가장 좋았던 단일 스로틀
    일률감속      IK 명령에 0.7 곱. 명령을 존중하면서 슬립만 줄이는 쪽
    경사스케줄    IMU 피치로 듀티 상한을 내리는 개루프 고전 제어
    순수 IK       보정 없음
    RL 8-D        조향+속도 잔차 (±10도, ±0.6)
    RL 4-D        속도 잔차만
모든 비교군은 조향은 순수 IK 를 쓴다 (고전 제어에 조향 보정이 없으므로).
"""
import argparse, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from env import RoverEnv, OBS_PER_FRAME, HISTORY

OBS_DIM = OBS_PER_FRAME * HISTORY
from reward import PRESETS as REWARD_PRESETS

from traction import SlopeScheduledLimit
import unicodedata


def pad(t, w):
    """한글은 터미널에서 2칸을 먹는다. 표시폭 기준으로 맞춘다."""
    d = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(t))
    return str(t) + " " * max(w - d, 0)

CLIP_RL = (10.0, 0.60)      # RL 잔차 권한. 0.2 로는 모래에서 감속 여유가 없었다
CLIP_BASE = (0.0, 1.00)     # 고전 제어는 조향 보정 없음 / 듀티는 전권한


def _duty_action(e, k=None, peak=None):
    """IK 듀티를 **비율 유지**로 스케일하는 잔차 행동.

    고전 제어가 하는 일은 스로틀 레벨 조절뿐이고 바퀴 간 속도비(=기구학)는
    건드리지 않는다.  두 가지 방식을 구분한다:
      k    : IK 듀티에 곱한다.  nav2 속도 명령을 존중하면서 감속만 한다.
      peak : 최대 듀티를 이 값에 맞춘다.  명령을 무시하고 스로틀을 고정한다.
    """
    du = e.duty_ik
    m = float(np.abs(du).max())
    if m < 1e-6:
        return np.zeros(8)
    tgt = du * k if k is not None else du / m * peak
    a = np.zeros(8)
    a[4:] = np.clip((tgt - du) / max(e.clip_duty, 1e-9), -1.0, 1.0)
    return a


def make_baselines():
    """(이름, 클립, 정책, 상태객체) 목록.  정책은 (obs, env) 를 받는다."""
    sched = SlopeScheduledLimit()

    def full(o, e):                      # 명령 무시, 최대 스로틀
        return _duty_action(e, peak=1.0)

    def fixed07(o, e):                   # 명령 무시, 스로틀 0.7 고정
        return _duty_action(e, peak=0.70)

    def scale07(o, e):                   # 명령 존중, 일률 30% 감속
        return _duty_action(e, k=0.70)

    def slope(o, e):                     # 명령 존중, 피치로 상한 스케줄
        pitch = float(e.hist[-1][16]) * (np.pi / 2)     # 관측에 들어있는 피치
        k = float(np.abs(sched(np.ones(1), pitch))[0])
        return _duty_action(e, k=k)

    return [("풀스로틀(peak1.0)", CLIP_BASE, full, None),
            ("고정스로틀(peak0.7)", CLIP_BASE, fixed07, None),
            ("일률감속(x0.7)", CLIP_BASE, scale07, None),
            ("경사스케줄", CLIP_BASE, slope, sched)]


def rollout(policy, n, arena, difficulty, clip, reward, episode_s, seed0=1000,
            stateful=None, kind="sand", privileged=False):
    out = []; causes = []
    for i in range(n):
        e = RoverEnv(rew=REWARD_PRESETS[reward], difficulty=difficulty,
                     episode_s=episode_s, arena_eval=arena, eval_kind=kind,
                     privileged=privileged, seed=seed0 + i)
        e.set_clip(*clip)
        o, _ = e.reset(seed=seed0 + i)
        if stateful is not None:
            stateful.reset()
        R = 0.0; term = trunc = False; info = {}
        while not (term or trunc):
            a = np.zeros(8) if policy is None else policy(o, e)
            o, r, term, trunc, info = e.step(a); R += r
        # 전 구간 집계를 쓴다 (종료 스텝 순간값 금지 -- slip 이 3.2배 틀렸다)
        out.append((R, info["dist"], info["slip_m"], info["sink_m"],
                    float(info["success"]), float(info["tip"]), float(info["stuck"]),
                    info["f_belly"], info["f_lifted"], info["f_blocked"],
                    info["duty_sat"], info["duty_lost"], info["duty_use"],
                    info["steer_sat"], info["steer_use"],
                    float(info["goals"])))
        causes.append(info["cause"])
    return np.array(out), causes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--vecnorm", default=None)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--difficulty", type=float, default=0.8)
    p.add_argument("--reward", default="balanced")
    p.add_argument("--episode-s", type=float, default=20.0)
    p.add_argument("--arena", action="store_true", help="실제 경기장 STL 로 평가")
    p.add_argument("--terrain", default="sand", choices=["sand", "rock"],
                   help="실측 지형 종류. sand=규사 경사지형(주), rock=암석 착륙지")
    args = p.parse_args()

    policy = None
    # teacher 정책은 특권정보까지 붙은 관측(1193)을 받는다.  env 를 1150 으로
    # 만들면 정규화 단계에서 모양이 안 맞아 터진다.  모델에서 읽어 맞춘다.
    # 특권정보는 관측 뒤에 덧붙을 뿐 동역학을 바꾸지 않으므로, 비교군도 같은
    # env 로 돌려야 조건이 완전히 같다.
    priv = False
    if args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")
        priv = int(model.observation_space.shape[0]) > OBS_DIM
        if priv:
            print(f"[eval] teacher 모델 감지 (관측 {model.observation_space.shape[0]}) "
                  f"-> 특권 env 로 평가\n")
        norm = None
        if args.vecnorm:
            from stable_baselines3.common.vec_env import VecNormalize
            import pickle
            with open(args.vecnorm, "rb") as f:
                norm = pickle.load(f)

        def policy(o, e=None):
            if norm is not None:
                o = np.clip((o - norm.obs_rms.mean) / np.sqrt(norm.obs_rms.var + 1e-8),
                            -10, 10).astype(np.float32)
            return model.predict(o, deterministic=True)[0]

    tag = (("규사 경사지형 STL" if args.terrain == "sand" else "암석 착륙지 STL")
           if args.arena else f"랜덤 지형 d={args.difficulty}")
    print(f"평가: {tag},  {args.episodes} 에피소드,  보상 프리셋 '{args.reward}'\n")
    print(f"{pad('변형',20)} {'보상':>9s} {'목표수':>7s} {'이동 m':>8s} {'슬립':>7s} "
          f"{'침하%':>7s} {'성공%':>6s} {'전복%':>6s} {'고착%':>6s}")
    runs = list(make_baselines())
    runs.append(("순수 IK", (0.0, 0.0), None, None))
    if policy is not None:
        runs.append(("RL 8-D", CLIP_RL, policy, None))
        runs.append(("RL 4-D", (0.0, CLIP_RL[1]), policy, None))
    breakdown = {}
    for name, clip, pol, stateful in runs:
        r, causes = rollout(pol, args.episodes, args.arena, args.difficulty, clip,
                            args.reward, args.episode_s, stateful=stateful,
                            kind=args.terrain, privileged=priv)
        breakdown[name] = (r, causes)
        print(f"{pad(name,20)} {r[:,0].mean():+9.1f} {r[:,15].mean():7.2f} "
              f"{r[:,1].mean():8.3f} {r[:,2].mean():7.3f} "
              f"{r[:,3].mean()*100:7.1f} {r[:,4].mean()*100:6.0f} {r[:,5].mean()*100:6.0f} "
              f"{r[:,6].mean()*100:6.0f}")

    # --- 왜 멈췄나 --------------------------------------------------------
    print(f"\n{pad('변형',20)}  --- 멈춘 원인 (에피소드 수) ---        "
          f"--- 에피소드 중 상태 비율 ---")
    print(f"{pad('',20)} {'하중상실':>9s} {'배접촉':>8s} {'막힘':>6s} {'슬립고착':>9s}   "
          f"{'배접촉':>8s} {'하중상실':>9s} {'막힘':>6s}")
    for name, (r, causes) in breakdown.items():
        cnt = {k: causes.count(k) for k in
               ("배걸림-하중상실", "배걸림-접촉", "막힘", "슬립고착")}
        print(f"{pad(name,20)} {cnt['배걸림-하중상실']:9d} {cnt['배걸림-접촉']:8d} "
              f"{cnt['막힘']:6d} {cnt['슬립고착']:9d}   "
              f"{r[:,7].mean()*100:7.1f}% {r[:,8].mean()*100:8.1f}% {r[:,9].mean()*100:5.1f}%")

    # --- 잔차 권한: 요청한 보정이 실제로 인가됐나 -------------------------
    print(f"\n{pad('변형',20)} {'duty 포화':>9s} {'duty 손실':>9s} {'duty 사용':>9s}   "
          f"{'조향 포화':>9s} {'조향 사용':>9s}")
    for name, (r, _) in breakdown.items():
        print(f"{pad(name,20)} {r[:,10].mean()*100:8.1f}% {r[:,11].mean()*100:8.1f}% "
              f"{r[:,12].mean()*100:8.1f}%   {r[:,13].mean()*100:8.1f}% "
              f"{r[:,14].mean()*100:8.1f}%")


if __name__ == "__main__":
    main()
