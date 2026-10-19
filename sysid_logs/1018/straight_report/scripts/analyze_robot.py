"""10-18 실물 직진 4 주행 분석 (이 폴더만으로 재현).  python3 scripts/analyze_robot.py

입력  raw/robot/*.npz (Pi run_policy.py 로그)
출력  표 (stdout), fig/straight_1018.png, fig/traction_real.png, derived/traction_real.npz

오른 높이 = ∫ (VO 몸체 속도 vx, vy) · (IMU 가속도 방향 = 위) dt  (VO 공백은 앞뒤 값으로 메움)
바퀴 토크 = τ_stall (V/12) duty/799 - D ω   (시뮬 drive_torque 와 같은 직선 모델; V 는 --volt, 기본 = 서보 보고 전압)
쓴 마찰계수 μ_mob = Σ토크 / r / (m g cosθ)
"""
import argparse
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
TAU_STALL, W_NOLOAD, PWM_MAX, R_WHEEL, G = 1.96, 7.96, 799.0, 0.047503, 9.81
RUNS = [("IK", "ik_145429.npz", None), ("stv1", "stv1_145659.npz", 30), ("stv3", "stv3_145842.npz", 30)]


def sm(x, n=25):
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same") if x.ndim == 1 else np.stack([np.convolve(x[:, i], k, mode="same") for i in range(x.shape[1])], 1)


def analyze(path, mass, volt=None):
    d = np.load(path, allow_pickle=True)
    t = (d["host_ns"] - d["host_ns"][0]) / 1e9
    dt = np.diff(t, prepend=t[0])
    V = float(np.mean(d["steer_volt"])) if volt is None else volt
    duty = np.asarray(d["duty"], float)
    w = np.asarray(d["wheel_mps"], float) / R_WHEEL
    tau = TAU_STALL * (V / 12.0) * duty / PWM_MAX - TAU_STALL / W_NOLOAD * w
    a = sm(np.asarray(d["acc_g"], float))
    u = a / np.linalg.norm(a, axis=1, keepdims=True)
    pitch = np.degrees(np.arctan2(a[:, 0], a[:, 2]))
    N = mass * G * np.clip(u[:, 2], 0.5, 1.0)
    vo = d["vo"]
    ok = np.isfinite(vo[:, 0])
    vx = np.interp(t, t[ok], vo[ok, 0])
    vy = np.interp(t, t[ok], vo[ok, 1])
    first = t[np.argmax(ok)]
    inc = (vx * u[:, 0] + vy * u[:, 1]) * dt
    inc[t < first] = 0
    gz = d["gyro_dps"][:, 2] - np.degrees(d["gyro_bias_rad"][2])
    slip = np.nanmean(d["slip"], 1)
    slip[~ok] = np.nan
    return dict(t=t, V=V, duty=duty, w=w, tau=tau, F=tau.sum(1) / R_WHEEL, N=N, mu=tau.sum(1) / R_WHEEL / N, pitch=pitch,
                slip=slip, H=np.cumsum(inc), vx=vx, ok=ok, yaw=np.cumsum(gz * dt), action=np.asarray(d["action"], float),
                args=json.loads(str(d["args"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=float, default=2.397, help="로버 질량 [kg] (기본 CAD)")
    ap.add_argument("--volt", type=float, default=None, help="토크 계산 전압 (기본: 서보 보고 전압 ~11.1 V; 12 로 주면 시뮬과 같은 기준)")
    a = ap.parse_args()
    (ROOT / "fig").mkdir(exist_ok=True)
    (ROOT / "derived").mkdir(exist_ok=True)
    res = {n: analyze(ROOT / "raw/robot" / f, a.mass, a.volt) for n, f, _ in RUNS}
    print(f"질량 {a.mass} kg, 전압 {'서보 보고값' if a.volt is None else a.volt} V")
    print("정책   길이  VO%  높이cm  10s/20s/30s/40s/50s/60s 누적                 헛돎  전진cm/s  방향°  duty(FL FR RL RR)   μ_mob")
    for (n, f, sc) in RUNS:
        r = res[n]
        t, m = r["t"], r["t"] > 10
        hs = [100 * r["H"][np.searchsorted(t, q) - 1] if q <= t[-1] + 1 else np.nan for q in (10, 20, 30, 40, 50, 60)]
        print(f"{n:6s} {t[-1]:4.0f}s {100 * r['ok'].mean():3.0f}  {100 * r['H'][-1]:5.1f}  " + " ".join(f"{h:5.1f}" for h in hs)
              + f"  {100 * np.nanmean(r['slip'][m & r['ok']]):3.0f}%  {100 * np.mean(r['vx'][m]):4.1f}   {r['yaw'][-1]:+4.0f}  "
              f"{np.round(r['duty'][m].mean(0)).astype(int)}  {np.mean(r['mu'][m]):.3f}")
        if sc:
            print(f"        조향 잔차° {np.round(r['action'][m, :4].mean(0) * sc, 1)}  바퀴 잔차 {np.round(r['action'][m, 4:].mean(0), 2)}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for fam in ("Noto Sans CJK JP", "NanumGothic", "Malgun Gothic", "AppleGothic"):
        try:
            from matplotlib import font_manager
            font_manager.findfont(fam, fallback_to_default=False)
            plt.rcParams["font.family"] = fam
            break
        except Exception:
            pass
    # 그림 1: 실물 4 주행
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    ax = ax.ravel()
    for n, r in res.items():
        t = r["t"]
        H = np.where(t >= r["t"][np.argmax(r["ok"])], r["H"], np.nan)
        ax[0].plot(t, 100 * H, label=n)
        ax[1].plot(t, 100 * sm(np.nan_to_num(r["slip"], nan=np.nan)), label=n)
        ax[2].plot(t, 100 * sm(r["vx"]), label=n)
        ax[3].plot(t, r["yaw"], label=n)
        ax[4].plot(t, r["pitch"], label=n)
        ax[5].plot(t, r["duty"].mean(1), label=n)
    for a_, tt in zip(ax, ["오른 높이 [cm] (VO·IMU 위방향, 방향 무관)", "평균 헛돎 [%] (VO 기준)", "VO 전진 속도 [cm/s]",
                            "몸체 방향 변화 [deg] (자이로)", "피치 [deg] (IMU)", "평균 duty (4 바퀴)"]):
        a_.set_title(tt); a_.set_xlabel("시간 [s]"); a_.legend(fontsize=8); a_.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(ROOT / "fig/straight_1018.png", dpi=80)
    # 그림 2: 토크·견인
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    ax = ax.ravel()
    for n, r in res.items():
        t = r["t"]
        ax[0].plot(t, sm(r["tau"].sum(1), 50), label=n)
        ax[1].plot(t, sm(r["mu"], 50), label=n)
        ax[2].plot(t, 100 * np.where(t >= r["t"][np.argmax(r["ok"])], r["H"], np.nan), label=n)
        ax[3].plot(t, 100 * sm(np.nan_to_num(r["slip"], nan=np.nan)), label=n)
        mo = (t > 3) & np.isfinite(r["slip"])
        ax[4].scatter(np.clip(r["slip"][mo], -0.2, 1.2)[::5], r["mu"][mo][::5], s=3, alpha=0.4, label=n)
    for k, wn in enumerate(("FL", "FR", "RL", "RR")):
        ax[5].plot(res["stv3"]["t"], sm(res["stv3"]["tau"][:, k], 50), label=f"stv3 {wn}")
        ax[5].plot(res["IK"]["t"], sm(res["IK"]["tau"][:, k], 50), "--", lw=0.8, label=f"IK {wn}")
    for a_, tt in zip(ax, ["네 바퀴 토크 합 [N·m] (duty 모델, 1 s 평균)", "쓴 마찰계수 μ_mob", "오른 높이 [cm]", "헛돎 [%]",
                            "μ_mob vs 헛돎 (0.1 s 점)", "바퀴별 토크: stv3 (실선) / IK (점선)"]):
        a_.set_title(tt); a_.grid(alpha=.3); a_.legend(fontsize=7)
    ax[1].axhline(np.tan(np.radians(13.5)), color="k", ls=":", lw=1)
    fig.tight_layout(); fig.savefig(ROOT / "fig/traction_real.png", dpi=80)
    np.savez_compressed(ROOT / "derived/traction_real.npz",
                        **{f"{n}_{k}": v for n, r in res.items() for k, v in r.items() if isinstance(v, np.ndarray)})
    print("저장: fig/straight_1018.png fig/traction_real.png derived/traction_real.npz")


if __name__ == "__main__":
    main()
