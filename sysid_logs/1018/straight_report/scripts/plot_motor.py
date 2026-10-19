"""시뮬 구동 모델(펌웨어 PI + DC 모터 직선 모델)과 실물·시뮬 운전점 비교 -> fig/motor_model.png.  python3 scripts/plot_motor.py

모델 (ksrc sim/isaac/ksrc_rover/source/ksrc_rover/utils/actuators_torch.py, sim/rl/sim2sim_isaac.py MjRover.step):
  duty = WheelPI(ω_ref, ω_meas)            전향 ω_ref/0.0097 + 5 (+ 정지 출발 킥 40) + PI 보정 (kp 82.5, ki 2062, ±250 duty)
  τ    = τ_stall (V/12) duty/799 − D ω      τ_stall 1.96 N·m, D = τ_stall / 7.96 rad/s
  τ    ← clip(τ, −τ_stall(1 + ω/ω_max), τ_stall(1 − ω/ω_max))   정지 토크 한계 (12 V)
  (+ 바퀴 관절 쿨롱 마찰 frictionloss = τ_stall · F/799, F = --wheel-friction-duty)
"""
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
TAU_STALL, W_NOLOAD, PWM_MAX, R_WHEEL = 1.96, 7.96, 799.0, 0.047503
K_DUTY, F_KIN = 0.0097, 5.0
RUNS = [("실물 IK", "ik_145429.npz"), ("실물 stv1", "stv1_145659.npz"), ("실물 stv3", "stv3_145842.npz")]
SIMS = [("시뮬 IK μ0.35", "ik_mu0.35.npz"), ("시뮬 stv3 μ0.35", "stv3_mu0.35.npz")]


def tau_model(duty, w, V=12.0):
    t = TAU_STALL * (V / 12.0) * duty / PWM_MAX - TAU_STALL / W_NOLOAD * w
    hi = np.minimum(TAU_STALL, TAU_STALL * (1 - w / W_NOLOAD))
    lo = np.maximum(-TAU_STALL, TAU_STALL * (-1 - w / W_NOLOAD))
    return np.clip(t, lo, hi)


def main():
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
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    # (1) 모델: 토크–속도 직선
    w = np.linspace(0, 8, 200)
    for duty, c in ((200, "#bbb"), (331, "C3"), (400, "C1"), (600, "C2"), (799, "k")):
        ax[0].plot(w, tau_model(duty, w), color=c, label=f"duty {duty}" + ("  (0.15 m/s 전향값)" if duty == 331 else ""))
    ax[0].plot(w, tau_model(331, w, 11.1), "--", color="C3", lw=1, label="duty 331 @ 11.1 V")
    ax[0].axvline(0.15 / R_WHEEL, color="b", ls=":", lw=1)
    ax[0].text(0.15 / R_WHEEL + 0.05, 1.7, "0.15 m/s\n(3.16 rad/s)", color="b", fontsize=9)
    ax[0].set_xlabel("바퀴 각속도 ω [rad/s]")
    ax[0].set_ylabel("모터 토크 τ [N·m] (바퀴 축)")
    ax[0].set_title("시뮬 모터 모델: τ = 1.96 (V/12) duty/799 − 0.246 ω (12 V, 정지토크 한계)")
    ax[0].set_ylim(-0.3, 2.0)
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    # (2) 운전점: duty vs ω (경사 위, 10 s 이후)
    wl = np.linspace(2.5, 3.6, 10)
    ax[1].plot(wl, PWM_MAX * wl / W_NOLOAD, "k-", lw=1.2, label="무부하 duty (모델, 12 V)")
    ax[1].plot(wl, PWM_MAX * wl / (W_NOLOAD * 11.1 / 12), "k--", lw=1, label="무부하 duty (모델, 11.1 V)")
    ax[1].plot(wl, wl / K_DUTY + F_KIN, "b:", lw=1.2, label="펌웨어 PI 전향값 ω/0.0097 + 5")
    for (n, f), c in zip(RUNS, ("C0", "C2", "C3")):
        d = np.load(ROOT / "raw/robot" / f, allow_pickle=True)
        t = (d["host_ns"] - d["host_ns"][0]) / 1e9
        m = t > 10
        wm = np.asarray(d["wheel_mps"], float)[m] / R_WHEEL
        du = np.asarray(d["duty"], float)[m]
        ax[1].scatter(wm[::10, 2:].ravel(), du[::10, 2:].ravel(), s=4, color=c, alpha=0.35, label=f"{n} 뒷바퀴")
        ax[1].scatter(wm[::10, :2].ravel(), du[::10, :2].ravel(), s=4, color=c, alpha=0.35, marker="x")
    for (n, f), c in zip(SIMS, ("#555", "#b58")):
        d = np.load(ROOT / "raw/sim_legacy" / f)
        m = d["t"] > 10
        ax[1].scatter(d["wheel_w"][m][::10].ravel(), d["duty"][m][::10].ravel(), s=4, color=c, alpha=0.35, marker="^", label=n)
    ax[1].set_xlim(2.5, 3.6)
    ax[1].set_ylim(220, 480)
    ax[1].set_xlabel("바퀴 각속도 ω [rad/s] (엔코더)")
    ax[1].set_ylabel("duty")
    ax[1].set_title("경사 위 운전점 (10 s 이후): 무부하선 위로 뜬 만큼이 부하 토크\n(1 duty ≈ 0.0025 N·m @12 V; 점 = 뒷바퀴, x = 앞바퀴)")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=7, loc="upper left")
    # (3) 바퀴별 평균 duty
    labels, vals = [], []
    for n, f in RUNS:
        d = np.load(ROOT / "raw/robot" / f, allow_pickle=True)
        t = (d["host_ns"] - d["host_ns"][0]) / 1e9
        labels.append(n)
        vals.append(np.asarray(d["duty"], float)[t > 10].mean(0))
    for n, f in SIMS:
        d = np.load(ROOT / "raw/sim_legacy" / f)
        labels.append(n)
        vals.append(d["duty"][d["t"] > 10].mean(0))
    vals = np.array(vals)
    x = np.arange(len(labels))
    for k, (wn, c) in enumerate(zip(("FL", "FR", "RL", "RR"), ("#9cf", "#69c", "#f96", "#c63"))):
        ax[2].bar(x + (k - 1.5) * 0.2, vals[:, k], 0.2, color=c, label=wn)
    ax[2].axhline(0.15 / R_WHEEL / K_DUTY + F_KIN, color="b", ls=":", lw=1, label="PI 전향값 (331)")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(labels, rotation=15)
    ax[2].set_ylim(250, 420)
    ax[2].set_ylabel("평균 duty (10 s 이후)")
    ax[2].set_title("바퀴별 평균 duty: 실물 vs 시뮬 μ0.35 (뒷바퀴가 더 큼)")
    ax[2].legend(fontsize=8, ncol=5)
    ax[2].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(ROOT / "fig/motor_model.png", dpi=80)
    print("저장 fig/motor_model.png")
    for n, v in zip(labels, vals):
        print(f"  {n:16s} duty FL FR RL RR {np.round(v).astype(int)}")


if __name__ == "__main__":
    main()
