"""Simulated visual-odometry body velocity for the student (2026-10-15, provisional).

The real rover would get its translational velocity from ORB-SLAM (stereo IR, laptop or Pi), sent to the
policy loop.  Nothing about that stream is measured yet, so this model is deliberately pessimistic and wide
(rover_env_cfg.VO_DR); narrow it after the 10-16/17 measurements.  Per env and episode:

  rate      a new measurement every k policy steps (k in rate_steps -> 5-17 Hz), held in between
  latency   the measurement describes the body velocity delay_steps ago
  noise     scale error (1 +- scale) * v + N(0, sigma + sigma_w * |wz|) per axis (blur grows with rotation)
  spikes    with spike_prob per measurement, + N(0, spike_std) (map switch / relocalisation jumps)
  outages   whole episode without VO with off_prob; otherwise outages start at drop_rate [1/s] and last
            drop_s seconds
Output (N,4): [vx, vy, valid, age]  body frame [m/s], valid in {0,1}, age = seconds since the measured
instant (latency + hold), capped at age_cap.  When not valid: vx = vy = 0 and age = age_cap -- what the Pi
feeds when it has no VO (pi/policy/student_runtime.py), so a student trained with this runs without VO too.
"""

import torch
from isaaclab.managers import ManagerTermBase


class VoVelocityObs(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params["dr"]
        self.p = p
        n, dev = env.num_envs, env.device
        self.dt = float(env.step_dt)
        self.L = int(p["delay_steps"][1]) + 1
        self.hist = torch.zeros(self.L, n, 3, device=dev)             # vx, vy, wz (newest at 0)
        self.t = torch.zeros(n, dtype=torch.long, device=dev)
        self.k = torch.ones(n, dtype=torch.long, device=dev)
        self.delay = torch.zeros(n, dtype=torch.long, device=dev)
        self.sigma = torch.zeros(n, device=dev)
        self.scale = torch.ones(n, device=dev)
        self.off = torch.zeros(n, dtype=torch.bool, device=dev)
        self.drop_left = torch.zeros(n, device=dev)
        self.hold = torch.zeros(n, 2, device=dev)
        self.age = torch.full((n,), float(p["age_cap"]), device=dev)
        self.has = torch.zeros(n, dtype=torch.bool, device=dev)
        self.out = torch.zeros(n, 4, device=dev)
        self._last_step = -1
        self.reset(None)

    def _u(self, m, lo, hi):
        return lo + (hi - lo) * torch.rand(m, device=self.hist.device)

    def reset(self, env_ids=None):
        ids = torch.arange(self.t.numel(), device=self.t.device) if env_ids is None else torch.as_tensor(
            env_ids, device=self.t.device)
        m = ids.numel()
        if m == 0:
            return
        p = self.p
        self.t[ids] = 0
        self.k[ids] = torch.randint(int(p["rate_steps"][0]), int(p["rate_steps"][1]) + 1, (m,), device=ids.device)
        self.delay[ids] = torch.randint(int(p["delay_steps"][0]), int(p["delay_steps"][1]) + 1, (m,), device=ids.device)
        self.sigma[ids] = self._u(m, *p["sigma"])
        self.scale[ids] = 1.0 + self._u(m, -p["scale"], p["scale"])
        self.off[ids] = torch.rand(m, device=ids.device) < p["off_prob"]
        self.drop_left[ids] = 0.0
        self.hold[ids] = 0.0
        self.age[ids] = float(p["age_cap"])
        self.has[ids] = False
        self.hist[:, ids] = 0.0
        self.out[ids] = torch.tensor([0.0, 0.0, 0.0, float(p["age_cap"])], device=ids.device)

    def __call__(self, env, dr):
        step = int(env.common_step_counter)
        if step == self._last_step:
            return self.out.clone()
        self._last_step = step
        p, dt = self.p, self.dt
        d = env.scene["robot"].data
        cur = torch.cat([d.root_lin_vel_b[:, :2], d.root_ang_vel_b[:, 2:3]], dim=1)
        fresh = self.t == 0                                           # first step of the episode: fill history
        self.hist = torch.roll(self.hist, 1, dims=0)
        self.hist[0] = cur
        if fresh.any():
            self.hist[:, fresh] = cur[fresh].unsqueeze(0)
        self.t += 1
        # outages
        start = (~self.off) & (self.drop_left <= 0) & (torch.rand_like(self.sigma) < p["drop_rate"] * dt)
        self.drop_left = torch.where(start, self._u(self.t.numel(), *p["drop_s"]), (self.drop_left - dt).clamp(min=0))
        avail = (~self.off) & (self.drop_left <= 0)
        upd = avail & (self.t % self.k == 0)
        if upd.any():
            ar = torch.arange(self.t.numel(), device=self.t.device)
            past = self.hist[self.delay.clamp(max=self.L - 1), ar]      # (N,3) velocity delay steps ago
            sd = (self.sigma + p["sigma_w"] * past[:, 2].abs()).unsqueeze(1)
            meas = self.scale.unsqueeze(1) * past[:, :2] + sd * torch.randn_like(past[:, :2])
            spike = (torch.rand_like(self.sigma) < p["spike_prob"]).unsqueeze(1)
            meas = meas + spike * p["spike_std"] * torch.randn_like(meas)
            self.hold = torch.where(upd.unsqueeze(1), meas, self.hold)
            self.age = torch.where(upd, self.delay.float() * dt, self.age + dt)
            self.has |= upd
        else:
            self.age = self.age + dt
        self.age = self.age.clamp(max=p["age_cap"])
        valid = self.has & avail & (self.age < p["valid_age"])
        v = torch.where(valid.unsqueeze(1), self.hold, torch.zeros_like(self.hold))
        age = torch.where(valid, self.age, torch.full_like(self.age, p["age_cap"]))
        self.out = torch.cat([v, valid.float().unsqueeze(1), age.unsqueeze(1)], dim=1)
        return self.out.clone()
