"""경로추종 보상이 스워브의 게걸음·후진과 횡오차를 올바르게 평가하는지 검사."""
import math

import numpy as np

from reward import PRESETS, tracking_terms


ZERO = np.zeros(3)


def score(cfg, *, progress=1.0, cross_track=0.0, heading=0.0,
          course_error=0.0, movement=0.0,
          action=ZERO, previous=ZERO, previous2=ZERO):
    gate, terms = tracking_terms(progress, cross_track, heading, action,
                                 previous, previous2, cfg,
                                 course_error=course_error, movement=movement)
    return gate, sum(terms.values()), terms


def test_crab_and_reverse_keep_path_reward():
    cfg = PRESETS["balanced"]
    center = [score(cfg, heading=yaw, movement=1.0)[1] for yaw in
              (0.0, math.pi / 2, math.pi)]
    assert np.allclose(center, 1.0), center
    offset = [score(cfg, cross_track=0.07, heading=yaw, movement=1.0)[1] for yaw in
              (0.0, math.pi / 2, math.pi)]
    assert np.allclose(offset, offset[0]), offset


def test_cross_track_gate_dominates_progress():
    cfg = PRESETS["balanced"]
    _, center, _ = score(cfg)
    gate, seven_cm, terms_seven = score(cfg, cross_track=0.07, movement=1.0)
    _, far, terms = score(cfg, cross_track=1.0)
    assert math.isclose(gate, math.exp(-0.49), rel_tol=1e-12)
    assert 0.6 < seven_cm < 0.7 and center == 1.0
    assert terms_seven["lat"] < 0.0
    assert math.isclose(far, cfg.w_prog, rel_tol=1e-12)
    assert terms["align"] < 1e-12
    assert score(cfg, progress=0.0)[1] == 0.0


def test_constant_large_residual_is_penalized():
    cfg = PRESETS["balanced"]
    full = np.ones(3)
    _, _, terms = score(cfg, progress=0.0, action=full,
                        previous=full, previous2=full)
    assert terms["smooth"] == 0.0
    assert math.isclose(terms["resid"], -cfg.w_resid)


def test_movement_heading_penalizes_wrong_travel_direction():
    cfg = PRESETS["balanced"]
    _, forward, _ = score(cfg, movement=1.0)
    gate, diagonal, terms = score(cfg, course_error=math.pi / 4, movement=1.0)
    assert math.isclose(gate, math.sqrt(0.5), rel_tol=1e-12)
    assert diagonal < forward
    assert terms["course"] < 0.0
    _, stopped, stopped_terms = score(cfg, progress=0.0, cross_track=0.10,
                                      course_error=math.pi / 4, movement=0.0)
    assert stopped == 0.0 and stopped_terms["lat"] == 0.0


def test_legacy_reproduces_old_heading_gate():
    cfg = PRESETS["legacy"]
    _, forward, _ = score(cfg)
    _, crab, _ = score(cfg, heading=math.pi / 2)
    assert math.isclose(forward, 1.8)
    assert math.isclose(crab, 1.0)


def test_environment_uses_heading_independent_terms():
    from env import RoverEnv
    from path import heading_at

    env = RoverEnv(arena_eval=True, eval_kind="sand", randomize=False, seed=0)
    try:
        env.reset(seed=0)
        rewards = []
        for yaw_error in (0.0, math.pi):
            env.s_prev = env.s_path - 0.001
            tangent_angle = heading_at(env.path, env.s_path - 0.0005)
            tangent = np.array([math.cos(tangent_angle), math.sin(tangent_angle)])
            env.last_xy = env.d.qpos[:2].copy() - 0.002 * tangent
            env.e_y = 0.07
            env.e_psi = yaw_error
            reward, _, info = env._reward(ZERO, np.zeros(4))
            assert math.isclose(info["gate"], math.exp(-0.49), rel_tol=1e-6)
            assert info["course_error_deg"] < 0.1
            rewards.append(reward)
        assert math.isclose(*rewards, rel_tol=1e-12), rewards
    finally:
        env.close()


if __name__ == "__main__":
    test_crab_and_reverse_keep_path_reward()
    test_cross_track_gate_dominates_progress()
    test_constant_large_residual_is_penalized()
    test_movement_heading_penalizes_wrong_travel_direction()
    test_legacy_reproduces_old_heading_gate()
    test_environment_uses_heading_independent_terms()
    print("path reward: OK")
