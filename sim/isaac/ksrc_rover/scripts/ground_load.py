"""Ground load check (2026-10-14): does the sim wheel need as much PI correction on flat ground as the rover?

Rover (bringup.py duty --vx 0.15, WPI ON, concrete floor): wheels exactly 0.150 m/s, duty 375-402 vs FF
w/0.0097 + 5 = 331 -> PI adds +45..+70 duty (rolling resistance + gearbox friction under load).
Open loop (WPI OFF, duty 317): 0.120-0.127 m/s.

Here: plane terrain, straight vx = 0.15 (no vy, no yaw), IK (Res8 zero action), "pi" and "openloop" drive,
DR stage 2 as in training.  Prints per-wheel duty - FF and wheel speed over envs (after 1 s).

  bash run.sh /workspace/scripts/ground_load.py --headless
  bash run.sh /workspace/scripts/ground_load.py --headless --set DR_WIDE=True
"""

import argparse
import ast
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--vx", type=float, default=0.15)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import ksrc_rover  # noqa: F401,E402
import ksrc_rover.rover_env_cfg as rcfg  # noqa: E402
from ksrc_rover.utils.contract import WHEEL_RADIUS_M  # noqa: E402


def run(drive):
    rcfg.TERRAIN_KIND = "plane"
    rcfg.DRIVE_MODEL = drive
    for kv in args.set:
        k, v = kv.split("=", 1)
        setattr(rcfg, k, ast.literal_eval(v))
    cfg = rcfg.RoverRes8EnvCfg()
    cfg.scene.num_envs = args.num_envs
    r = cfg.commands.base_velocity.ranges
    r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (args.vx, args.vx), (0.0, 0.0), (0.0, 0.0)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    if hasattr(cfg.commands.base_velocity, "heading_command"):
        cfg.commands.base_velocity.heading_command = False
    env = gym.make("Isaac-KSRC-Rover-Res8-v0", cfg=cfg)
    u = env.unwrapped
    term = u.action_manager.get_term("rover")
    robot = u.scene["robot"]
    env.reset()
    zero = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
    duty, w, ff = [], [], []
    for k in range(args.steps):
        env.step(zero)
        if k >= 50:
            duty.append(term.duty.clone())
            w.append(robot.data.joint_vel[:, term.wheel_ids].clone())
            w_ref = term._processed[:, 4:]
            ff.append(w_ref / term.pi.cfg["k_duty"] + torch.sign(w_ref) * term.pi.cfg["f_kin"])
    duty, w, ff = torch.stack(duty), torch.stack(w), torch.stack(ff)
    v = w.mean(0) * WHEEL_RADIUS_M                         # (N,4) m/s per env
    extra = (duty - ff).mean(0)                           # (N,4)
    q = lambda x: [round(float(x.quantile(p)), 3) for p in (0.05, 0.5, 0.95)]
    print(f"GROUND drive={drive} vx={args.vx} DR_WIDE={rcfg.DR_WIDE} envs={u.num_envs}", flush=True)
    print(f"GROUND   바퀴 속도 m/s  5/50/95%: {q(v)}   (실물 PI 0.150, 개루프 0.120-0.127)", flush=True)
    print(f"GROUND   duty 평균 5/50/95%: {q(duty.mean(0))}   (실물 PI 375-402, 개루프 317)", flush=True)
    if drive == "pi":
        print(f"GROUND   duty - FF 5/50/95%: {q(extra)}   (실물 +45..+70)", flush=True)
    env.close()


def main():
    run("pi")


try:
    main()
except Exception:
    traceback.print_exc()
sys.stdout.flush()
sys.stderr.flush()
app.close()
