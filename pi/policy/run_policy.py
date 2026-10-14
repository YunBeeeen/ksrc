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
    ap.add_argument("--log", default=None, help="npz 로 관측·행동·텔레메트리 저장")
    args = ap.parse_args()

    import torch
    meta = json.load(open(args.policy + ".json"))
    if meta["action_dim"] != 8:
        raise SystemExit(f"Res8 정책이 아님 (action_dim {meta['action_dim']})")
    pol = torch.jit.load(args.policy + ".pt", map_location="cpu").eval()
    h = torch.zeros(*meta["hidden_shape"]) if meta.get("recurrent") else None
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
    next_t = time.monotonic()
    try:
        for k in range(int(args.seconds / dt)):
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
            # what the Nucleo applies from this command on = the next frame's applied_targets
            applied = applied_for_obs(*ik.step(vx, vy, w, d_steer, gain), tlm_next)
            for key, v in (("t_ms", tlm["t_ms"]), ("obs", obs[0]), ("action", a), ("flags", tlm["flags"]),
                           ("duty", tlm["duty"]), ("wheel_mps", tlm["wheel_mps"]),
                           ("steer_meas", tlm["steer_meas_rad"]), ("steer_cmd", tlm.get("steer_cmd_rad", (0,) * 4))):
                log[key].append(np.asarray(v))
            if k % 25 == 0:
                fl = tlm["flags"]
                print(f"[run] t={tlm['t_ms']} 잔차 {'ON' if fl & nl.TLM_FLAG_RES_ACTIVE else 'off'} "
                      f"게이트 {'보류' if fl & nl.TLM_FLAG_GATE_PENDING else '-'} "
                      f"a={np.round(a, 2)} duty={tlm['duty']}", flush=True)
            tlm = tlm_next
            next_t += dt
            time.sleep(max(0.0, next_t - time.monotonic()))
    finally:
        for _ in range(5):      # stop: plain velocity command 0 (also clears the residual in firmware)
            link.exchange(0.0, 0.0, 0.0)
            time.sleep(dt)
        link.close()
    print(f"[run] 끝. 텔레메트리 실패 {n_bad} 회", flush=True)
    if args.log:
        np.savez_compressed(args.log, meta=json.dumps(meta), **{k: np.asarray(v) for k, v in log.items()})
        print(f"[run] 저장 {args.log}", flush=True)


if __name__ == "__main__":
    main()
