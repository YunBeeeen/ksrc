"""Proprioceptive observations shared by all three action modes."""

from isaaclab.managers import SceneEntityCfg


def steering_position(env, asset_cfg: SceneEntityCfg):
    return env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]


def steering_velocity(env, asset_cfg: SceneEntityCfg):
    return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]


def wheel_velocity(env, asset_cfg: SceneEntityCfg):
    return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]


def applied_targets(env):
    """The joint targets the policy just commanded (IK + residual, before the command delay and the drive
    gate) -- what the Pi knows it sent (pi/policy/student_runtime.py applied_for_obs).  2026-10-14: was the
    *delayed* target (processed_actions); with a 1-2 step delay the policy then could not see the commands
    still in flight, and the Pi side meant something else."""
    return env.action_manager.get_term("rover").commanded_targets


def zeros3(env):
    """Masked privileged term (rover_env_cfg.PRIV_MASK, diagnostics only)."""
    import torch
    return torch.zeros(env.num_envs, 3, device=env.device)


def zeros4(env):
    import torch
    return torch.zeros(env.num_envs, 4, device=env.device)


# --- Privileged (teacher-only) observations ------------------------------------
# Ground truth the real robot cannot measure. Only the "privileged" group carries
# these; the teacher actor/critic read it, the student never does (it must infer
# them from its proprioceptive history). Values that are set once at startup are
# cached on first use; before the randomizing event has run they read as nominal.

def wheel_mu(env):
    """Realized static μ per wheel (fl, fr, rl, rr). 1.0 until the μ event has run."""
    import torch
    mu = getattr(env, "_wheel_mu4", None)
    if mu is None:
        return torch.ones(env.num_envs, 4, device=env.device)
    return torch.nan_to_num(mu.to(env.device), nan=1.0)


def wheel_slip(env):
    """Signed longitudinal slip per wheel, (surface speed - ground speed) / max(|.|, 0.05)."""
    from .rewards import wheel_slip_ratio
    return wheel_slip_ratio(env)


def base_mass_com(env):
    """[mass scale, CoM offset x,y,z (m)] of base_link, written by events.randomize_base_mass_com."""
    import torch
    v = getattr(env, "_priv_mass_com", None)
    if v is None:
        v = torch.zeros(env.num_envs, 4, device=env.device)
        v[:, 0] = 1.0
    return v
