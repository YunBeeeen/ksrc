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
    p.add_argument("--path", default=None,
                   help="draw_path.py 로 찍은 경로 파일(.npy).  주면 그 경로만 달린다")
    p.add_argument("--stage1", action="store_true",
                   help="학습과 같은 조건 (경로 세트 고정, 랜덤화 OFF)")
    p.add_argument("--path-seed", type=int, default=1234)
    p.add_argument("--n-paths", type=int, default=10)
    p.add_argument("--static", action="store_true", help="물리 안 돌리고 지형만 본다")
    # nargs="?" 로 두면 `--realtime` 만 써도 1.0 이 된다 (값을 안 적으면 다음
    # 인자를 값으로 먹으려 해서 "expected one argument" 가 났다).
    p.add_argument("--realtime", type=float, nargs="?", const=1.0, default=1.0,
                   help="재생 배속 (기본 1.0). `--realtime 0.5` 로 느리게")
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
    # 경로 지정 / Stage 1 조건 재현
    paths = None; s0f = None
    if args.path:
        import path as pth
        paths = [pth.RefPath(np.load(args.path))]
        s0f = 0.0                     # 지정 경로는 시작점부터
        print("경로 %s: 꼭짓점 %d, 길이 %.2fm" % (args.path, len(paths[0].P), paths[0].total))
    elif args.stage1:
        import terrain as terr, path as pth
        from config import RoverCfg
        kind = args.terrain if args.terrain != "proc" else "sand"
        Z = terr.load_arena(kind); ex, ey = terr.eval_extent(kind)
        drive, _ = terr.drivable_mask(Z, ex, ey, max_slope_deg=RoverCfg().max_slope_deg)
        paths = pth.make_path_set(args.n_paths, drive, Z, ex, ey, seed=args.path_seed)
        print("Stage 1 경로 세트 %d개 (seed %d)" % (args.n_paths, args.path_seed))
    e = RoverEnv(rew=REWARD_PRESETS["balanced"], difficulty=args.d,
                 arena_eval=arena, eval_kind=(args.terrain if arena else "sand"),
                 privileged=priv, seed=args.seed,
                 paths=paths, s0_frac=s0f,
                 # --path / --stage1 은 학습 조건 재현이므로 랜덤화·교란을 끈다
                 randomize=not (args.path or args.stage1),
                 perturb=0.0)
    e.set_clip(0.0 if args.model is None else args.clip_steer_deg,
               0.0 if args.model is None else args.clip_duty)
    obs, _ = e.reset(seed=args.seed)

    tag = {"sand": "규사 경사지형 STL", "rock": "암석 착륙지 STL",
           "proc": f"절차생성 학습지형 d={args.d}"}[args.terrain]
    print(f"[view] {tag}  |  흙 mu_max {e.terr_p.mu_max:.2f}  모래두께 {e.terr_p.z_max*1000:.0f}mm")
    print(f"[view] 정책: {'순수 IK' if policy is None else args.model}"
          + ("  (특권 관측 teacher)" if priv else ""))
    print("[view] 좌드래그 회전 / 우드래그 이동 / 휠 줌")
    print("[view] 키: p 일시정지 · r 리셋(새 경로) · o 보정ON/OFF · , . 배속")

    # ---- 키보드 (passive viewer 는 마우스 콜백이 없고 key_callback 만 받는다) ----
    st = {"pause": False, "reset": False, "resid": policy is not None,
          "rt": args.realtime}

    def on_key(k):
        c = chr(k).lower() if 32 <= k < 127 else ""
        if c == "p":
            st["pause"] = not st["pause"]; print("[view] " + ("일시정지" if st["pause"] else "재생"))
        elif c == "r":
            st["reset"] = True
        elif c == "o" and policy is not None:
            st["resid"] = not st["resid"]
            print("[view] 잔차 보정 %s" % ("ON" if st["resid"] else "OFF (순수 IK)"))
        elif c == ".":
            st["rt"] = min(st["rt"] * 1.5, 16.0); print("[view] 배속 %.2f" % st["rt"])
        elif c == ",":
            st["rt"] = max(st["rt"] / 1.5, 0.05); print("[view] 배속 %.2f" % st["rt"])

    def draw_path(v):
        """기준경로를 뷰어에 그린다.  이게 없으면 무엇을 따라가는지 안 보인다."""
        v.user_scn.ngeom = 0
        P = e.path.P
        zt = float(e.d.qpos[2]) + 0.05
        n = 0
        for i in range(len(P) - 1):
            if n >= v.user_scn.maxgeom - 2:
                break
            g = v.user_scn.geoms[n]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                np.array([0.1, 0.4, 1.0, 0.9], np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012,
                                 np.array([P[i][0], P[i][1], zt]),
                                 np.array([P[i+1][0], P[i+1][1], zt]))
            n += 1
        # 목표점 (마지막 꼭짓점)
        if n < v.user_scn.maxgeom:
            g = v.user_scn.geoms[n]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.05, 0.05, 0.05]),
                                np.array([P[-1][0], P[-1][1], zt]), np.eye(3).ravel(),
                                np.array([1.0, 0.3, 0.1, 0.9], np.float32))
            n += 1
        v.user_scn.ngeom = n

    dt = 1.0 / CTRL_HZ
    with mujoco.viewer.launch_passive(e.m, e.d, key_callback=on_key) as v:
        v.cam.distance = 3.0
        v.cam.elevation = -25
        draw_path(v)
        while v.is_running():
            t0 = time.time()
            if st["reset"]:
                st["reset"] = False
                obs, _ = e.reset()
                v.cam.lookat[:] = e.d.qpos[:3]
                draw_path(v)
                print("[view] 리셋.  경로 %.2fm  s0 %.2f" % (e.path.total, e.s0))
            if args.static or st["pause"]:
                mujoco.mj_forward(e.m, e.d)
            else:
                a = (np.zeros(3) if (policy is None or not st["resid"])
                     else policy(obs))
                obs, r, term, trunc, info = e.step(a)
                if term or trunc:
                    print("[view] 끝: 성공 %s 전복 %s 고착 %s | %.1fs %.2fm | "
                          "CTE 평균 %.0fmm p90 %.0fmm | 기수 %.1f도 | a %.2f"
                          % (info["success"], info["tip"], info["stuck"],
                             info["t_elapsed"], info["dist"],
                             1000 * info["e_y_m"], 1000 * info["e_y_p90"],
                             info["e_psi_m"], info["a_use"]))
                    obs, _ = e.reset()
                    v.cam.lookat[:] = e.d.qpos[:3]
                    draw_path(v)
            v.sync()
            lag = dt / max(st["rt"], 1e-3) - (time.time() - t0)
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()
