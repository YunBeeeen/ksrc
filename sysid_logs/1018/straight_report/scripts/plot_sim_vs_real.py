"""실물 직진 (analyze_traction.py 결과) vs MuJoCo legacy 모래 (sim_legacy/*.npz, slope_drive_log.py) 비교 그림."""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK JP"
HERE = pathlib.Path(__file__).resolve().parents[1]
R = np.load(HERE / "derived/traction_real.npz")
sm = lambda x, n=50: np.convolve(x, np.ones(n) / n, mode="same")
fig, ax = plt.subplots(2, 3, figsize=(18, 9.5))
ax = ax.ravel()
for n, c in (("IK", "C0"), ("stv1", "C2"), ("stv3", "C3")):
    t = R[f"{n}_t"]
    ax[0].plot(t, 100 * R[f"{n}_H"], c, lw=2.2, label=f"실물 {n}")
    ax[1].plot(t, 100 * sm(np.nan_to_num(R[f"{n}_slip"], nan=0.9), 25), c, lw=1.6, label=f"실물 {n}")
    ax[2].plot(t, sm(R[f"{n}_duty"].mean(1)), c, lw=1.6, label=f"실물 {n}")
sim = {}
for pol, ls in (("ik", "--"), ("stv3", ":")):
    for mu, c in (("0.30", "#888"), ("0.35", "k"), ("0.40", "#b58")):
        d = np.load(HERE / "raw/sim_legacy" / f"{pol}_mu{mu}.npz")
        sim[(pol, mu)] = d
        t = d["t"]
        h = np.clip(d["pos"][:, 0] - float(d["ramp_x0"]), 0, 3.0) * np.tan(np.radians(13.5))
        h = h - h[np.argmax(h > 0.002)] * 0 if False else h
        lab = f"시뮬 {pol.upper() if pol == 'ik' else pol} μ{mu}"
        ax[0].plot(t, 100 * h, ls, color=c, lw=1.2, label=lab)
        ax[1].plot(t, 100 * sm(np.abs(d["slip"]).mean(1), 25), ls, color=c, lw=1, label=lab)
        ax[2].plot(t, sm(d["duty"].mean(1)), ls, color=c, lw=1, label=lab)
for pol, ls in (("ik", "-"), ("stv3", "--")):
    d = sim[(pol, "0.35")]
    t = d["t"]
    ax[3].plot(t, sm(d["tm_mu"].mean(1), 10), ls, label=f"{pol} μ0.35: 순간 마찰계수 (4 바퀴 평균)")
    ax[4].plot(t, 1000 * sm(d["tm_z"].mean(1), 10), ls, label=f"{pol} μ0.35: 침하 z [mm]")
    ax[5].plot(t, sm(d["tm_R"].sum(1) / (float(d["mass"]) * 9.81), 10), ls, label=f"{pol} μ0.35: 저항 ΣR / mg")
    ax[5].plot(t, sm(d["mu_mob"], 25), ls, alpha=0.6, label=f"{pol} μ0.35: 쓴 견인 μ_mob (토크)")
ax[3].axhline(0.35, color="k", lw=0.6, ls=":")
ax[5].axhline(np.tan(np.radians(13.5)), color="k", lw=0.6, ls=":")
ttl = ["오른 높이 [cm]  (실물 = 굵은 실선, 시뮬 IK = 점선, stv3 = 점점선)", "헛돎 [%]", "평균 duty (4 바퀴)",
       "시뮬 내부: 매 스텝 다시 쓰는 바퀴 마찰계수 μ(i) (μ_max 0.35)", "시뮬 내부: 바퀴 침하 z [mm]", "시뮬 내부: 침하 저항과 쓴 견인 (÷ mg) — 점선 = tan 13.5°"]
for a_, tt in zip(ax, ttl):
    a_.set_title(tt, fontsize=10)
    a_.grid(alpha=0.3)
    a_.legend(fontsize=7, ncol=2)
    a_.set_xlabel("시간 [s]")
ax[0].set_ylim(0, 75)
fig.tight_layout()
fig.savefig(HERE / "fig/sim_vs_real.png", dpi=78)
print("저장 fig/sim_vs_real.png")
for (pol, mu), d in sim.items():
    t = d["t"]
    h = np.clip(d["pos"][:, 0] - float(d["ramp_x0"]), 0, 3.0) * np.tan(np.radians(13.5))
    hs = [100 * h[np.searchsorted(t, q) - 1] for q in (10, 20, 30, 40, 50, 60)]
    m = t > 10
    print(f"시뮬 {pol:4s} μ{mu}: 높이 " + " ".join(f"{x:5.1f}" for x in hs) + f" | 헛돎 {np.abs(d['slip'][m]).mean():.2f} μ(i) {d['tm_mu'][m].mean():.3f} 침하 {1000 * d['tm_z'][m].mean():.1f} mm "
          f"ΣR/mg {np.mean(d['tm_R'][m].sum(1)) / (float(d['mass']) * 9.81):.3f} μ_mob {d['mu_mob'][m].mean():.3f} 수직력 앞/뒤 {np.round(d['tm_Fn'][m].mean(0), 2)} N")
