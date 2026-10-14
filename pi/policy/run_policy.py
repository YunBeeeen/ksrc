"""Run the exported Res8 student on the rover over SPI (structure B: firmware IK + residual).

  # on the Pi, rover on blocks first (wheels free), then on the ground
  python3 pi/policy/run_policy.py --policy ~/policies/res8_mlp_h5 --cmd 0.10 0 0 --seconds 5
  python3 pi/policy/run_policy.py --policy ... --cmd 0.10 0 0 --residual-off      # pure IK via type 0x02
  python3 pi/policy/run_policy.py --policy ... --fake --seconds 2                  # laptop, no SPI

--policy is the export path without extension (<p>.pt TorchScript + <p>.json layout,
sim/isaac/ksrc_rover/scripts/export_student.py).  Teleop (V / SPI 0x01) is untouched: stop this
script and the old tools work as before; the firmware drops the residual 100 ms after the
last type-0x02 frame and stops 500 ms after the last command.

Loop (50 Hz, Pi clock):  telemetry of the previous exchange -> observation -> policy -> residual
-> SPI type 0x02 (vx, vy, w, d_steer, wheel_gain).  Before driving it polls 2 s standing still to
take the gyro bias (the student was trained with the bias the filter sees, minus nothing).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from student_runtime import FirmwareIK, StudentObs, action_to_residual, applied_for_obs, nl


class FakeLink:
    """No SPI: a rover that follows the command perfectly (for checking the loop on a laptop)."""

    def __init__(self):
        self.ik = FirmwareIK()
        self.goal, self.speed, self.t = np.zeros(4), np.zeros(4), 0

    def _tlm(self):
        self.t += 20
        return {"flags": nl.TLM_FLAG_IMU_VALID | nl.TLM_FLAG_RES_ACTIVE, "t_ms": self.t, "tick": self.t // 20 & 255,
                "gyro_dps": (0.0, 0.0, 0.0), "acc_g": (0.0, 0.0, 1.0), "steer_meas_rad": tuple(self.goal),
                "steer_speed": (0, 0, 0, 0), "wheel_mps": tuple(self.speed), "duty": (0, 0, 0, 0)}

    def exchange(self, vx, vy, w, poll_only=False):
        if not poll_only:
            self.goal, self.speed = self.ik.step(vx, vy, w)
        return self._tlm()

    def exchange_res(self, vx, vy, w, d_steer, wheel_gain):
        self.goal, self.speed = self.ik.step(vx, vy, w, d_steer, wheel_gain)
        return self._tlm()

    def close(self):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="내보낸 정책 경로 (확장자 없이: <p>.pt, <p>.json)")
    ap.add_argument("--cmd", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("VX", "VY", "W"))
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--residual-off", action="store_true", help="잔차 0 으로 (타입 0x02 경로·관측만 확인)")
    ap.add_argument("--fake", action="store_true", help="SPI 없이 (노트북에서 루프 확인)")
    ap.add_argument("--bias-s", type=float, default=2.0, help="시작 전 정지 자이로 바이어스 측정 시간")
    ap.add_argument("--log", default=None, help="npz 로 관측·행동·텔레메트리·루프 주기 저장")
    ap.add_argument("--rt", type=int, default=0, metavar="PRIO",
                    help="실시간 우선순위 (SCHED_FIFO, 예: 50). 권한 필요: sudo 또는 'sudo setcap cap_sys_nice+ep $(readlink -f $(which python3))'")
    ap.add_argument("--cpu", type=int, default=None, help="이 코어에만 (예: 3. ORB-SLAM 등과 코어 나누기)")
    ap.add_argument("--gc-off", action="store_true", help="루프 동안 파이썬 GC 끄기 (가끔 수 ms 멈춤 방지)")
    args = ap.parse_args()

    import gc
    import os
    import torch
    torch.set_num_threads(1)                 # 작은 MLP: 스레드 여럿은 오히려 흔들림만 늘림
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
        print(f"[run] 코어 {args.cpu} 에 고정", flush=True)
    if args.rt:
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(args.rt))
            print(f"[run] 실시간 우선순위 SCHED_FIFO {args.rt}", flush=True)
        except PermissionError:
            print("[run] 실시간 우선순위 권한 없음 (--rt 무시). sudo 로 실행하거나 setcap 필요", flush=True)
    meta = json.load(open(args.policy + ".json"))
    if meta["action_dim"] != 8:
        raise SystemExit(f"Res8 정책이 아님 (action_dim {meta['action_dim']})")
    pol = torch.jit.load(args.policy + ".pt", map_location="cpu").eval()
    h = torch.zeros(*meta["hidden_shape"]) if meta.get("recurrent") else None
    with torch.no_grad():                     # 첫 호출은 TorchScript 최적화로 수십 ms -> 루프 전에 한 번
        x0 = torch.zeros(1, int(meta["obs_dim"]))
        for _ in range(3):
            pol(x0) if h is None else pol(x0, h)
    dt = float(meta.get("control_dt_s", 0.02))
    obs_b = StudentObs(meta, dt)
    ik = FirmwareIK()
    link = FakeLink() if args.fake else nl.SpiLink()

    # 1) standing still: gyro bias (POLL frames do not feed the Nucleo watchdog)
    g = []
    t_end = time.monotonic() + args.bias_s
    while time.monotonic() < t_end:
        tlm = link.exchange(0.0, 0.0, 0.0, poll_only=True)
        if tlm is not None and tlm["flags"] & nl.TLM_FLAG_IMU_VALID:
            g.append(np.asarray(tlm["gyro_dps"], float) * math.pi / 180.0)
        time.sleep(dt)
    if not g:
        raise SystemExit("IMU 값이 안 들어옴 (TLM_FLAG_IMU_VALID) -- 배선·펌웨어 확인")
    bias = np.mean(g, 0)
    print(f"[run] 자이로 바이어스 {np.round(bias, 4)} rad/s ({len(g)} 샘플)", flush=True)

    vx, vy, w = args.cmd
    applied = applied_for_obs(*ik.step(0.0, 0.0, 0.0), None)
    log = {k: [] for k in ("t_ms", "obs", "action", "flags", "duty", "wheel_mps", "steer_meas", "steer_cmd")}
    n_bad = 0
    t_start, t_comp, n_late, n_resync = [], [], 0, 0
    if args.gc_off:
        gc.collect()
        gc.disable()
    next_t = time.monotonic()
    try:
        for k in range(int(args.seconds / dt)):
            t0 = time.monotonic()
            t_start.append(t0)
            if t0 - next_t > 0.005:
                n_late += 1
            if tlm is None:
                n_bad += 1
                tlm = link.exchange(vx, vy, w, poll_only=True)
                continue
            obs = obs_b.push(obs_b.frame((vx, vy, w), tlm, bias, applied))
            with torch.no_grad():
                x = torch.as_tensor(obs)
                if h is None:
                    a = pol(x)
                else:
                    a, h = pol(x, h)
            a = a[0].numpy()
            d_steer, gain = action_to_residual(a)
            if args.residual_off:
                d_steer, gain = np.zeros(4, np.float32), np.ones(4, np.float32)
            tlm_next = link.exchange_res(vx, vy, w, d_steer.tolist(), gain.tolist())
            t_comp.append(time.monotonic() - t0)
            # what the Nucleo applies from this command on = the next frame's applied_targets
            applied = applied_for_obs(*ik.step(vx, vy, w, d_steer, gain), tlm_next)
            for key, v in (("t_ms", tlm["t_ms"]), ("obs", obs[0]), ("action", a), ("flags", tlm["flags"]),
                           ("duty", tlm["duty"]), ("wheel_mps", tlm["wheel_mps"]),
                           ("steer_meas", tlm["steer_meas_rad"]), ("steer_cmd", tlm.get("steer_cmd_rad", (0,) * 4))):
                log[key].append(np.asarray(v))
            if k % 25 == 0:
                fl = tlm["flags"]
                win = np.diff(t_start[-26:]) * 1000 if len(t_start) > 1 else np.zeros(1)
                print(f"[run] t={tlm['t_ms']} 잔차 {'ON' if fl & nl.TLM_FLAG_RES_ACTIVE else 'off'} "
                      f"게이트 {'보류' if fl & nl.TLM_FLAG_GATE_PENDING else '-'} "
                      f"주기 최대 {win.max():.1f} ms  계산 {t_comp[-1] * 1000:.1f} ms  "
                      f"a={np.round(a, 2)} duty={tlm['duty']}", flush=True)
            tlm = tlm_next
            next_t += dt
            if time.monotonic() - next_t > dt:      # 한 주기 넘게 밀렸으면 몰아서 따라잡지 않고 다시 맞춤
                next_t = time.monotonic()
                n_resync += 1
            time.sleep(max(0.0, next_t - time.monotonic()))
    finally:
        for _ in range(5):      # stop: plain velocity command 0 (also clears the residual in firmware)
            link.exchange(0.0, 0.0, 0.0)
            time.sleep(dt)
        link.close()
    if args.gc_off:
        gc.enable()
    print(f"[run] 끝. 텔레메트리 실패 {n_bad} 회", flush=True)
    per = np.diff(np.asarray(t_start)) * 1000 if len(t_start) > 1 else np.zeros(1)
    comp = np.asarray(t_comp) * 1000 if t_comp else np.zeros(1)
    q = lambda x, p: float(np.quantile(x, p))
    print(f"[run] 루프 주기 [ms] (목표 {dt * 1000:.0f}): 평균 {per.mean():.2f}  중앙 {q(per, .5):.2f}  p95 {q(per, .95):.2f}  "
          f"p99 {q(per, .99):.2f}  최대 {per.max():.2f}  |  25 ms 넘음 {int((per > 25).sum())}회, "
          f"40 ms 넘음 {int((per > 40).sum())}회 / {len(per)}", flush=True)
    print(f"[run] 한 주기 계산 (관측+정책+SPI) [ms]: 중앙 {q(comp, .5):.2f}  p99 {q(comp, .99):.2f}  최대 {comp.max():.2f}  |  "
          f"5 ms 넘게 늦게 시작 {n_late}회, 한 주기 넘게 밀려 재정렬 {n_resync}회", flush=True)
    if args.log:
        log["loop_t"], log["period_ms"], log["compute_ms"] = [t_start], [per], [comp]
        np.savez_compressed(args.log, meta=json.dumps(meta), **{k: np.asarray(v[0] if k in ("loop_t", "period_ms", "compute_ms") else v)
                                                                 for k, v in log.items()})
        print(f"[run] 저장 {args.log}", flush=True)


if __name__ == "__main__":
    main()
