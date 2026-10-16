"""Velocity command with raw tracking-error metrics for TensorBoard (2026-10-12).

Isaac Lab's UniformVelocityCommand logs error_vel_* as an episode SUM divided by the
max command duration, so with 2-6 s commands in a 16 s episode it is not m/s (~x2.7) and
moves with episode length. This term keeps those (for continuity) and adds per-step
means in m/s, rad/s, split like scripts/eval_analyze.py:
  err_xy / err_yaw                 all steps
  err_xy_steady / err_yaw_steady   >= TRANSIENT_S after a command change or reset
  err_xy_slick / err_yaw_slick     steady steps with 1-3 slick wheels (mu < SLICK_MU) --
                                   the part a policy can actually improve
  spin_slick / spin_grip           mean |surface - ground speed| [m/s] of slick / grippy wheels
Logged as Metrics/base_velocity/<name>. Values are pooled over the envs that reset at a
step (sum / count), so a short episode does not weigh as much as a full one.

Rate limit (2026-10-14): a resampled command is a *target*; the command the policy sees and is scored on
moves toward it like a path follower would: the xy velocity as (signed speed s, direction phi), phi turning
at most dir_rate [rad/s] and s changing at most acc_lin [m/s^2], omega at most acc_ang [rad/s^2].
phi lives in (-90, 90] deg around the current one: going the opposite way is a speed sign flip (the
firmware IK reverses the wheels instead of steering 180 deg), so a reversal passes s through 0 without
swinging the direction.  (A straight-line ramp in (vx, vy) passes near 0 speed where the direction swings
fast -- it made the gate hold MORE, 13 -> 17 %, first try 10-14.)  A fraction abrupt_prob of resamples still jumps at once (evasive moves, joystick).
An episode starts from a standing command.  acc_* <= 0 turns the limit off (old behaviour).
Why: with every 2-6 s command jumping in direction, the firmware drive-align gate held the wheels 17 %
of the time in sim (IK, 10-14 eval) -- far more than a Nav2-driven rover sees.
"""

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

# Nav2-like command mix (mode="nav", 10-15, evaluation): Regulated Pure Pursuit outputs (v, kappa) with
# omega = v*kappa and no vy; it slows on curves and rotates in place to a waypoint heading.  Mostly forward
# at 0.08-0.25 m/s, half nearly straight (|kappa| <= 0.3 1/m), half curving (radius 0.67-3.3 m), plus
# in-place rotations, short back-ups and a few small sideways corrections.  Still smoothed by the rate limit.
NAV = dict(v=(0.08, 0.25), straight_prob=0.5, kappa_straight=0.3, kappa_turn=(0.3, 1.5),
           rotate_prob=0.10, w_rotate=(0.3, 0.6), reverse_prob=0.05, v_reverse=(0.05, 0.12),
           crab_prob=0.05, vy_crab=(0.03, 0.08))
TRANSIENT_S = 1.0
SLICK_MU = 0.2
_KEYS = ("err_xy", "err_yaw", "err_xy_steady", "err_yaw_steady", "err_xy_slick", "err_yaw_slick",
         "spin_slick", "spin_grip")


class RoverVelocityCommand(UniformVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, k = self.num_envs, len(_KEYS)
        self._sum = torch.zeros(n, k, device=self.device)
        self._cnt = torch.zeros(n, k, device=self.device)
        self._t_cmd = torch.zeros(n, device=self.device)
        self._last = {key: 0.0 for key in _KEYS}   # logged when no reset env has such steps
        self._target = torch.zeros(n, 3, device=self.device)     # sampled command (rate limit goes toward it)
        self._out = torch.zeros(n, 3, device=self.device)        # what the policy sees
        self._s = torch.zeros(n, device=self.device)             # signed xy speed of _out
        self._phi = torch.zeros(n, device=self.device)           # its direction, (-pi/2, pi/2] kept near itself
        self._abrupt = torch.zeros(n, dtype=torch.bool, device=self.device)
        if getattr(cfg, "heading_command", False) and cfg.acc_lin > 0:
            raise ValueError("command rate limit assumes heading_command=False")

    def _update_metrics(self):
        super()._update_metrics()
        from .rewards import _wheel_surface_and_ground_speed

        e_xy = torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1)
        e_yaw = torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        steady = (self._t_cmd >= TRANSIENT_S).float()
        mu4 = getattr(self._env, "_wheel_mu4", None)   # set by randomize_wheel_mu_patch (CPU)
        if mu4 is not None and not torch.isnan(mu4).any():
            slick = mu4.to(self.device) < SLICK_MU                          # (N,4)
        else:
            slick = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        n_slick = slick.sum(dim=1)
        part = steady * ((n_slick >= 1) & (n_slick <= 3)).float()
        surf, v_long = _wheel_surface_and_ground_speed(self._env)
        spin = (surf - v_long).abs()
        sf = slick.float()
        one = torch.ones_like(e_xy)
        vals = torch.stack((e_xy, e_yaw, e_xy * steady, e_yaw * steady, e_xy * part, e_yaw * part,
                            (spin * sf).sum(1), (spin * (1 - sf)).sum(1)), dim=1)
        cnts = torch.stack((one, one, steady, steady, part, part, sf.sum(1), (1 - sf).sum(1)), dim=1)
        self._sum += vals
        self._cnt += cnts
        self._t_cmd += self._env.step_dt

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if self.cfg.mode == "nav":
            self._sample_nav(env_ids)
        elif self.cfg.mode != "uniform":
            raise ValueError(f"unknown command mode {self.cfg.mode!r}")
        self._t_cmd[env_ids] = 0.0
        if self.cfg.acc_lin > 0:
            self._target[env_ids] = self.vel_command_b[env_ids]
            self._abrupt[env_ids] = torch.rand(len(env_ids), device=self.device) < self.cfg.abrupt_prob
            self.vel_command_b[env_ids] = self._out[env_ids]     # keep the current command; ramp in _update

    def _sample_nav(self, env_ids):
        """Nav2 RPP-like command (v, kappa) -> (vx, 0, v*kappa); see NAV for the mix."""
        n, dev = len(env_ids), self.device
        u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, device=dev)
        sign = lambda: torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
        v = u(*NAV["v"])
        straight = torch.rand(n, device=dev) < NAV["straight_prob"]
        kappa = torch.where(straight, u(-NAV["kappa_straight"], NAV["kappa_straight"]), sign() * u(*NAV["kappa_turn"]))
        cmd = torch.stack((v, torch.zeros_like(v), v * kappa), dim=1)
        r = torch.rand(n, device=dev)
        p_rot, p_rev, p_crab = NAV["rotate_prob"], NAV["reverse_prob"], NAV["crab_prob"]
        rot = r < p_rot                                          # rotate to heading at a waypoint
        rev = (r >= p_rot) & (r < p_rot + p_rev)                 # short back-up (recovery)
        crab = (r >= p_rot + p_rev) & (r < p_rot + p_rev + p_crab)   # small sideways correction
        cmd[rot] = torch.stack((torch.zeros(n, device=dev), torch.zeros(n, device=dev),
                                sign() * u(*NAV["w_rotate"])), dim=1)[rot]
        cmd[rev, 0] = -u(*NAV["v_reverse"])[rev]
        cmd[rev, 2] = 0.0
        cmd[crab, 1] = (sign() * u(*NAV["vy_crab"]))[crab]
        self.vel_command_b[env_ids] = cmd

    def _update_command(self):
        if self.cfg.acc_lin <= 0:
            super()._update_command()
            return
        import math
        tgt = self._target.clone()
        tgt[self.is_standing_env.nonzero(as_tuple=False).flatten()] = 0.0
        dt = self._env.step_dt
        s_t = torch.linalg.norm(tgt[:, :2], dim=1)
        phi_t = torch.atan2(tgt[:, 1], tgt[:, 0])
        # target direction within +-90 deg of the current one (else flip it and the speed sign)
        dphi = torch.remainder(phi_t - self._phi + math.pi, 2 * math.pi) - math.pi
        flip = dphi.abs() > math.pi / 2
        dphi = torch.where(flip, dphi - torch.sign(dphi) * math.pi, dphi)
        s_t = torch.where(flip, -s_t, s_t)
        moving = (s_t.abs() > 1e-4) | (self._s.abs() > 1e-4)
        # standing start: take the target direction at once (nothing to steer away from)
        fresh = self._s.abs() <= 1e-4
        phi = torch.where(fresh, self._phi + dphi, self._phi + dphi.clamp(-self.cfg.dir_rate * dt, self.cfg.dir_rate * dt))
        phi = torch.where(moving, phi, self._phi)
        s_new = self._s + (s_t - self._s).clamp(-self.cfg.acc_lin * dt, self.cfg.acc_lin * dt)
        w_new = self._out[:, 2] + (tgt[:, 2] - self._out[:, 2]).clamp(-self.cfg.acc_ang * dt, self.cfg.acc_ang * dt)
        ab = self._abrupt
        phi_ab = phi_t
        self._phi = torch.where(ab, phi_ab, phi)
        self._s = torch.where(ab, torch.linalg.norm(tgt[:, :2], dim=1), s_new)
        w_new = torch.where(ab, tgt[:, 2], w_new)
        self._out = torch.stack((self._s * torch.cos(self._phi), self._s * torch.sin(self._phi), w_new), dim=1)
        self.vel_command_b[:] = self._out

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._out[ids] = 0.0                                     # an episode starts standing
        self._s[ids] = 0.0
        s, c = self._sum[ids].sum(dim=0), self._cnt[ids].sum(dim=0)
        extras = super().reset(env_ids)
        for i, key in enumerate(_KEYS):
            if c[i] > 0:
                self._last[key] = (s[i] / c[i]).item()
            extras[key] = self._last[key]
        self._sum[ids] = 0.0
        self._cnt[ids] = 0.0
        return extras


@configclass
class RoverVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = RoverVelocityCommand
    acc_lin: float = 0.0          # [m/s^2] command rate limit (xy norm). 0 = off
    acc_ang: float = 0.0          # [rad/s^2]
    abrupt_prob: float = 0.0      # fraction of resamples that still jump at once
    dir_rate: float = 1.5         # [rad/s] max turn rate of the xy command direction (servo ~1.8 rad/s avg over 90 deg)
    mode: str = "uniform"         # "uniform" (ranges) or "nav" (Nav2-like mix, NAV above)
