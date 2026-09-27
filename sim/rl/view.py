#!/usr/bin/env python3
"""지형·로버를 MuJoCo 뷰어로 띄운다 (눈으로 확인하는 용도).

  python3 view.py                     # 규사 경사지형 STL, 순수 IK 주행
  python3 view.py --terrain proc --d 1.0
  python3 view.py --model runs/reach/final --vecnorm runs/reach/vecnorm.pkl
  python3 view.py --static            # 물리 정지, 지형만 둘러보기

마우스: 좌드래그 회전 / 우드래그 이동 / 휠 줌.  스페이스로 일시정지.
"""
import argparse, pathlib, sys, time
import numpy as np
import mujoco, mujoco.viewer

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from env import RoverEnv, CTRL_HZ, OBS_PER_FRAME, HISTORY

OBS_DIM = OBS_PER_FRAME * HISTORY
from reward import PRESETS as REWARD_PRESETS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--terrain", default="sand", choices=["sand", "rock", "proc"],
                   help="sand/rock=실측 STL, proc=절차생성 학습지형")
    p.add_argument("--d", type=float, default=1.0, help="난이도")
    p.add_argument("--model", default=None, help="학습된 정책 (없으면 순수 IK)")
    p.add_argument("--vecnorm", default=None)
    p.add_argument("--clip-steer-deg", type=float, default=10.0)
    p.add_argument("--clip-duty", type=float, default=0.60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--static", action="store_true", help="물리 안 돌리고 지형만 본다")
    p.add_argument("--realtime", type=float, default=1.0, help="재생 배속")
    args = p.parse_args()

    # 모델을 **먼저** 읽어야 한다.  teacher 정책은 특권정보까지 붙은 관측(1193)을
    # 받는데, env 를 1150 으로 만들어 두면 정규화에서 모양이 안 맞아 터진다.
    policy = None
    priv = False
    sb = norm = None
    if args.model:
        from stable_baselines3 import PPO
        sb = PPO.load(args.model, device="cpu")
        priv = int(sb.observation_space.shape[0]) > OBS_DIM
        if args.vecnorm:
            import pickle
            with open(args.vecnorm, "rb") as f:
                norm = pickle.load(f)

        def policy(o):
            if norm is not None:
                o = np.clip((o - norm.obs_rms.mean) / np.sqrt(norm.obs_rms.var + 1e-8),
                            -10, 10).astype(np.float32)
            return sb.predict(o, deterministic=True)[0]

    arena = args.terrain in ("sand", "rock")
    e = RoverEnv(rew=REWARD_PRESETS["balanced"], difficulty=args.d,
                 arena_eval=arena, eval_kind=(args.terrain if arena else "sand"),
                 privileged=priv, seed=args.seed)
    e.set_clip(0.0 if args.model is None else args.clip_steer_deg,
               0.0 if args.model is None else args.clip_duty)
    obs, _ = e.reset(seed=args.seed)

    tag = {"sand": "규사 경사지형 STL", "rock": "암석 착륙지 STL",
           "proc": f"절차생성 학습지형 d={args.d}"}[args.terrain]
    print(f"[view] {tag}  |  흙 mu_max {e.terr_p.mu_max:.2f}  모래두께 {e.terr_p.z_max*1000:.0f}mm")
    print(f"[view] 정책: {'순수 IK' if policy is None else args.model}"
          + ("  (특권 관측 teacher)" if priv else ""))
    print("[view] 좌드래그 회전 / 우드래그 이동 / 휠 줌 / 스페이스 일시정지")

    dt = 1.0 / CTRL_HZ
    with mujoco.viewer.launch_passive(e.m, e.d) as v:
        v.cam.distance = 3.0
        v.cam.elevation = -25
        while v.is_running():
            t0 = time.time()
            if args.static:
                mujoco.mj_forward(e.m, e.d)
            else:
                a = np.zeros(8) if policy is None else policy(obs)
                obs, r, term, trunc, info = e.step(a)
                if term or trunc:
                    print(f"[view] 에피소드 끝: 성공 {info['success']} 고착 {info['stuck']} "
                          f"전복 {info['tip']} 원인 {info['cause']} "
                          f"슬립 {info['slip']:.2f} 이동 {info['dist']:.2f}m")
                    obs, _ = e.reset()
                    v.cam.lookat[:] = e.d.qpos[:3]
            v.sync()
            lag = dt / max(args.realtime, 1e-3) - (time.time() - t0)
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()
