"""VO 관측 항(mdp/vo.py) 스모크 (2026-10-15): 정책 없이 IK(zero action)로 돌리며 student 그룹의 vo 값을 참 속도와 비교.

  bash run.sh /workspace/scripts/vo_check.py --headless --num_envs 64 --steps 400 --set STUDENT_VO=True
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--set", action="append", default=[])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

os.environ["KSRC_SET"] = "\n".join(args.set)
import ksrc_rover  # noqa: F401,E402
import ksrc_rover.rover_env_cfg as rcfg  # noqa: E402

cfg = rcfg.RoverRes8EnvCfg_PLAY()
cfg.scene.num_envs = args.num_envs
env = gym.make("Isaac-KSRC-Rover-Res8-Play-v0", cfg=cfg)
u = env.unwrapped
obs, _ = env.reset()
om = u.observation_manager
names = om.active_terms["student"]
dims = om.group_obs_term_dim["student"]
print("student terms:", list(zip(names, dims)), flush=True)
assert "vo" in names, "STUDENT_VO=True 로 실행"
hist = rcfg.STUDENT_HISTORY
off = 0
for n, dshape in zip(names, dims):
    size = int(torch.tensor(dshape).prod())
    if n == "vo":
        sl = slice(off + size - 4, off + size)          # newest frame (history flattened oldest first)
    off += size
act = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
errs, valid, ages = [], [], []
for t in range(args.steps):
    obs, _, _, _, _ = env.step(act)
    vo = obs["student"][:, sl]
    true = u.scene["robot"].data.root_lin_vel_b[:, :2]
    m = vo[:, 2] > 0.5
    valid.append(m.float().mean().item()); ages.append(vo[:, 3].mean().item())
    if m.any():
        errs.append((vo[m, :2] - true[m]).abs().mean().item())
    if t % 100 == 0:
        print(f"t={t} valid={valid[-1]:.2f} age={ages[-1]:.3f} vo0={vo[0].tolist()} true0={true[0].tolist()}", flush=True)
import statistics as st  # noqa: E402
print(f"VO_CHECK valid_frac {st.mean(valid):.3f}  mean_age {st.mean(ages):.3f}s  |vo-true| {st.mean(errs) if errs else float('nan'):.4f} m/s  obs_dim {obs['student'].shape[1]}", flush=True)
env.close()
app.close()
