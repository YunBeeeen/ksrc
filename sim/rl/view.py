#!/usr/bin/env python3
"""지형·로버를 MuJoCo 뷰어로 띄운다 (눈으로 확인하는 용도).

  python3 view.py                     # 규사 경사지형 STL, 순수 IK 주행
  python3 view.py --terrain proc --d 1.0
  python3 view.py --model runs/reach/final --vecnorm runs/reach/vecnorm.pkl
  python3 view.py --stage1 --watch-run runs/training
  python3 view.py --static            # 물리 정지, 지형만 둘러보기

마우스: 좌드래그 회전 / 우드래그 이동 / 휠 줌.  p 로 일시정지.
"""
import argparse, pathlib, re, sys, time
import numpy as np
import mujoco, mujoco.viewer

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from env import RoverEnv, CTRL_HZ, OBS_PER_FRAME, HISTORY

OBS_DIM = OBS_PER_FRAME * HISTORY


def latest_checkpoint_pair(run):
    """Find the newest policy and its matching observation statistics."""
    root = pathlib.Path(run)
    final = root / "final.zip"
    final_norm = root / "vecnorm.pkl"
    if final.is_file() and final_norm.is_file():
        return final, final_norm
    pairs = []
    for model in root.glob("ppo_*_steps.zip"):
        match = re.fullmatch(r"ppo_(\d+)_steps\.zip", model.name)
        if match:
            norm = root / f"ppo_vecnormalize_{match.group(1)}_steps.pkl"
            if norm.is_file():
                pairs.append((int(match.group(1)), model, norm))
    if not pairs:
        return None
    _, model, norm = max(pairs, key=lambda pair: pair[0])
    return model, norm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--terrain", default="sand", choices=["sand", "rock", "proc"],
                   help="sand/rock=실측 STL, proc=절차생성 학습지형")
    p.add_argument("--d", type=float, default=1.0, help="난이도")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--model", default=None, help="학습된 정책 (없으면 순수 IK)")
    source.add_argument("--watch-run", default=None,
                        help="학습 run의 최신 모델·정규화 쌍을 에피소드마다 자동 로드")
    p.add_argument("--vecnorm", default=None)
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
    p.add_argument("--drive-align-gate-deg", type=float, default=10.0,
                   help="구동 정렬 게이트 해제 오차 [deg], 0=게이트 끔. "
                        "뷰어에서 g 키로 실시간 토글 가능")
    p.add_argument("--steer-limit-deg", type=float, default=None,
                   help="조향 가동범위 ±[deg] 덮어쓰기 (기본 config 의 ±100). "
                        "±110 이상은 모터가 타이어보다 바깥으로 나간다")
    p.add_argument("--realtime", type=float, nargs="?", const=1.0, default=1.0,
                   help="재생 배속 (기본 1.0). `--realtime 0.5` 로 느리게")
    args = p.parse_args()

    if args.watch_run and args.vecnorm:
        p.error("--watch-run 은 체크포인트와 짝인 vecnorm을 자동 선택합니다")
    if args.watch_run and not (args.stage1 or args.path):
        p.error("--watch-run 은 모델을 재생할 고정 경로가 필요합니다: --stage1 또는 --path")
    if args.watch_run and args.static:
        p.error("--watch-run 은 주행 재생용이므로 --static 과 함께 사용할 수 없습니다")
    # 정책과 통계는 한 쌍으로 교체한다. 학습 중 파일을 쓰는 순간 읽었다면
    # 다음 에피소드에서 재시도하고, 주행 중에는 정책을 바꾸지 않는다.
    from stable_baselines3 import PPO
    import pickle
    loaded = {"model": None, "norm": None, "pair": None}

    def load_policy(model_path, norm_path=None, expected_obs_dim=None):
        sb = PPO.load(str(model_path), device="cpu")
        if (expected_obs_dim is not None and
                sb.observation_space.shape != (expected_obs_dim,)):
            raise ValueError(f"정책 관측 {sb.observation_space.shape}와 뷰어 환경 "
                             f"({expected_obs_dim},)이 다릅니다")
        stats = None
        if norm_path is not None:
            with open(norm_path, "rb") as f:
                stats = pickle.load(f)
            if stats.obs_rms.mean.shape != sb.observation_space.shape:
                raise ValueError("모델과 관측 정규화 통계의 차원이 다릅니다")
        loaded.update(model=sb, norm=stats, pair=(pathlib.Path(model_path), norm_path))
        print(f"[view] 정책 로드: {model_path} ({sb.num_timesteps:,} steps)")

    if args.model:
        load_policy(args.model, args.vecnorm)
    elif args.watch_run:
        pair = latest_checkpoint_pair(args.watch_run)
        if pair is not None:
            try:
                load_policy(*pair)
            except Exception as exc:
                print(f"[view] 체크포인트 로드 대기: {exc}")
        else:
            print(f"[view] {args.watch_run}: 첫 체크포인트 대기 중 (그동안 순수 IK)")

    # 모델을 먼저 읽어 teacher 정책의 특권 관측 차원을 확인한다.
    priv = (loaded["model"] is not None and
            int(loaded["model"].observation_space.shape[0]) > OBS_DIM)

    def policy(o):
        sb, norm = loaded["model"], loaded["norm"]
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
        cfg = RoverCfg()
        drive, _ = terr.drivable_mask(Z, ex, ey, max_slope_deg=cfg.max_slope_deg)
        route = terr.center_clearance_mask(drive, ex, ey, cfg.route_clearance)
        paths = pth.make_path_set(args.n_paths, route, Z, ex, ey, seed=args.path_seed)
        print("Stage 1 경로 세트 %d개 (seed %d)" % (args.n_paths, args.path_seed))
    cfg = None
    if args.steer_limit_deg is not None:
        if not (90.0 <= args.steer_limit_deg <= 180.0):
            p.error("--steer-limit-deg 는 90~180 사이여야 한다 "
                    "(90 미만이면 낼 수 없는 방향이 생긴다)")
        lim = float(args.steer_limit_deg)
        cfg = RoverCfg(steer_lo_deg=(-lim,) * 4, steer_hi_deg=(lim,) * 4)
    e = RoverEnv(cfg=cfg, difficulty=args.d,
                 drive_align_gate_deg=args.drive_align_gate_deg,
                 arena_eval=arena, eval_kind=(args.terrain if arena else "sand"),
                 privileged=priv, seed=args.seed,
                 paths=paths, s0_frac=s0f,
                 # --path / --stage1 은 학습 조건 재현이므로 랜덤화·교란을 끈다
                 randomize=not (args.path or args.stage1),
                 perturb=0.0)
    obs, _ = e.reset(seed=args.seed)
    # 어떤 설정으로 보고 있는지 창을 열기 전에 찍는다.  조향범위·게이트는
    # 눈으로 구분이 안 되는데 결과를 크게 바꾼다 (±100 vs ±135, 게이트 ON/OFF).
    _lim = [round(float(np.degrees(x))) for x in e.st_hi_c]
    _gate = ("OFF" if e.drive_align_gate_rad is None
             else "%.0f도" % np.degrees(e.drive_align_gate_rad))
    print("[view] 조향범위 ±%s도 | 구동정렬게이트 %s | 명령 %s | 지형 %s | 배속 %.2f"
          % (_lim[0] if len(set(_lim)) == 1 else _lim, _gate,
             "pure pursuit + 정책잔차" if loaded["model"] else "pure pursuit (순수 IK)",
             args.terrain, args.realtime))
    print("[view] 키: p 정지 | r 리셋 | g 게이트토글 | o 잔차토글 | , 느리게 | . 빠르게")

    def maybe_reload():
        if not args.watch_run:
            return
        pair = latest_checkpoint_pair(args.watch_run)
        if pair is None or pair == loaded["pair"]:
            return
        was_empty = loaded["model"] is None
        try:
            load_policy(*pair, expected_obs_dim=e.observation_space.shape[0])
        except Exception as exc:
            # 학습 프로세스가 체크포인트를 쓰는 도중이면 다음 에피소드에서 재시도.
            print(f"[view] 새 체크포인트 로드 대기: {exc}")
            return
        if was_empty:
            st["resid"] = True
        print("[view] 다음 에피소드부터 새 정책 적용")

    tag = {"sand": "규사 경사지형 STL", "rock": "암석 착륙지 STL",
           "proc": f"절차생성 학습지형 d={args.d}"}[args.terrain]
    print(f"[view] {tag}  |  흙 mu_max {e.terr_p.mu_max:.2f}  모래두께 {e.terr_p.z_max*1000:.0f}mm")
    print(f"[view] 정책: {'순수 IK' if loaded['model'] is None else loaded['pair'][0]}"
          + ("  (특권 관측 teacher)" if priv else ""))
    print("[view] 좌드래그 회전 / 우드래그 이동 / 휠 줌")
    print("[view] 키: p 일시정지 · r 리셋(새 경로) · o 보정ON/OFF · , . 배속")

    # ---- 키보드 (passive viewer 는 마우스 콜백이 없고 key_callback 만 받는다) ----
    st = {"pause": False, "reset": False, "resid": loaded["model"] is not None,
          "rt": args.realtime}

    def on_key(k):
        c = chr(k).lower() if 32 <= k < 127 else ""
        if c == "p":
            st["pause"] = not st["pause"]; print("[view] " + ("일시정지" if st["pause"] else "재생"))
        elif c == "r":
            st["reset"] = True
        elif c == "o" and loaded["model"] is not None:
            st["resid"] = not st["resid"]
            print("[view] 잔차 보정 %s" % ("ON" if st["resid"] else "OFF (순수 IK)"))
        elif c == "g":
            # 같은 에피소드 안에서 켜고 끄며 눈으로 비교할 수 있게 한다.
            if e.drive_align_gate_rad is None:
                e.drive_align_gate_rad = float(np.radians(
                    args.drive_align_gate_deg or 10.0))
                print("[view] 구동 정렬 게이트 ON (%.0f도)"
                      % np.degrees(e.drive_align_gate_rad))
            else:
                e.drive_align_gate_rad = None
                print("[view] 구동 정렬 게이트 OFF")
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
    episode_return = 0.0
    with mujoco.viewer.launch_passive(e.m, e.d, key_callback=on_key) as v:
        v.cam.distance = 3.0
        v.cam.elevation = -25
        draw_path(v)
        while v.is_running():
            t0 = time.time()
            if st["reset"]:
                st["reset"] = False
                obs, _ = e.reset()
                episode_return = 0.0
                maybe_reload()
                v.cam.lookat[:] = e.d.qpos[:3]
                draw_path(v)
                print("[view] 리셋.  경로 %.2fm  s0 %.2f" % (e.path.total, e.s0))
            if args.static or st["pause"]:
                mujoco.mj_forward(e.m, e.d)
            else:
                a = (np.zeros(3) if (loaded["model"] is None or not st["resid"])
                     else policy(obs))
                obs, r, term, trunc, info = e.step(a)
                episode_return += float(r)
                if args.watch_run and e.k % CTRL_HZ == 0:
                    print("[view] t %.1fs | e_y %.0fmm | heading %.1f° | "
                          "gate %.2f | reward %+.3f | max|a| %.2f" %
                          (e.k / CTRL_HZ, 1000 * info["e_y"], info["e_psi"],
                           info["gate"], r, np.max(np.abs(a))))
                if term or trunc:
                    print("[view] 끝: 성공 %s 전복 %s 고착 %s | %.1fs %.2fm | "
                          "CTE 평균 %.0fmm p90 %.0fmm | 기수 %.1f도 | "
                          "a %.2f | 총보상 %.1f"
                          % (info["success"], info["tip"], info["stuck"],
                             info["t_elapsed"], info["dist"],
                             1000 * info["e_y_m"], 1000 * info["e_y_p90"],
                             info["e_psi_m"], info["a_use"], episode_return))
                    obs, _ = e.reset()
                    episode_return = 0.0
                    maybe_reload()
                    v.cam.lookat[:] = e.d.qpos[:3]
                    draw_path(v)
            v.sync()
            lag = dt / max(st["rt"], 1e-3) - (time.time() - t0)
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()
