# KSRC Isaac Lab task

This directory mirrors `~/gyusa/ksrc_rover` on `hpi-lab`. Its `source/ksrc_rover/` package registers six Gymnasium task IDs (three action modes and their `-Play` variants). `scripts/train.py` and `scripts/play.py` run the version-matched Isaac Lab rsl_rl scripts from the server image. The image already contains `rsl_rl`; no extra pip install is needed.

From the repository root, `bash sim/isaac/ksrc_rover/deploy.sh` synchronizes the tracked task files without replacing the server's `assets/` or legacy diagnostic scripts. `scripts/_spikes/` is reserved for those diagnostics; they remain at their original paths on the server to preserve existing CLI references.

**Current state (2026-10-16).** The task is past bring-up and is the main training environment. The robot is the rev5 assembly with cylinder wheel collisions (`export KSRC_ROVER_USD=/workspace/assets/usd_rev5cyl/rover.usd`, built from `ros/rover_description/rover_rev5_isaac_cyl.urdf`, the same URDF as the MuJoCo `mjcf/rover_rev5.xml`). Physics 5 ms, decimation 4 (20 ms control). Actuators follow the 10-13/14 bench measurements: STS3215 servo model (34 ms dead time, 0.25 deg deadband, 7.7 rad/s^2 trapezoid), firmware wheel PI + duty motor (`DRIVE_MODEL="pi"`), firmware drive-align gate with the measured trigger, command delay 1-2 steps and observation delay 0-1 step.

The production line is **Res8 teacher -> MLP student**: `Isaac-KSRC-Rover-Teacher-Res8-v0` (privileged body velocity, wheel slip, mu, mass/CoM, DR values) distilled into `Isaac-KSRC-Rover-Student-MLP-Res8-v0` (Pi-measurable observations, history 5; optional VO body velocity with `STUDENT_VO=True`). Experiments are configured with `--set KEY=VALUE` on top of the constants in `source/ksrc_rover/rover_env_cfg.py`. Commands, the current SET, the evaluation protocol (`eval_log.py`, `logs/rank_nl.py`, `logs/jit_ana.py`) and student export (`scripts/export_student.py` -> `pi/policy/`) are in [CLI.md](CLI.md); results and decisions are in the repository root `HANDOFF.md` section 7.

Latest results (rev5 cylinder, vs pure IK, 384 episodes): teacher rc4000 big wheel spin -57%, student v1 -39% (deployed on the Pi), student v3 with VO -56%, jitter-penalised teacher smooth1015 @3000 -44% with policy-output jitter cut to one third.

`Residual3`, `Res8` and the students use `utils/ik_torch.py`, a batched port of `firmware/common/swerve_kinematics.c`; it matches the C IK exactly (20 000 random steps including the +-90 deg fold, 2026-10-16), and `python3 sim/rl/ik_validate.py` checks the parity. Trained policies are cross-checked in MuJoCo with `sim/rl/sim2sim_isaac.py` and `ros/isaac_bridge.py` (RViz + Nav2).

The [2026-10-09 pure-IK benchmark](benchmarks/2026-10-09_pure_ik/RESULTS.md) records the early matched flat-ground Isaac/MuJoCo rollouts (before the measured actuator models).
