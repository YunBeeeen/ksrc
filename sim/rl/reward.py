"""논문의 경로추종 보상을 홀로노믹 스워브 이동에 맞춰 적용한다."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class RewardCfg:
    """진행·정렬·횡오차·이동방향오차·매끄러움으로 경로추종을 평가한다.

    2026년 4WIS 논문은 차체 기수와 경로 접선을 비교한다. 여기서는 스워브의
    게걸음·후진을 허용하기 위해 실제 이동 방향과 경로 접선을 비교한다.
    legacy 는 이전 차체 기수 보상을, position_only 는 방향 항을 빼던 직전
    보상을 같은 동역학에서 재현하는 비교군이다.
    """
    w_prog:    float = 0.15    # + 진행 바닥값
    w_align:   float = 0.85    # + 횡이탈·이동방향 게이트가 걸린 진행
    w_lat:     float = 0.05    # - 이동 중 횡오차 (정지 유도 방지)
    w_course:  float = 0.02    # - 이동 방향과 경로 접선의 오차
    w_smooth:  float = 0.10    # - 행동 2차 차분
    w_resid:   float = 0.03    # - 지속적인 큰 차체속도 잔차
    r_goal:    float = 20.0
    r_tip:     float = -10.0
    r_stuck:   float = -5.0
    r_oob:     float = -10.0   # 예전엔 벌점이 **없어서** 경계 밖 탈출이 공짜였다
    sigma_y:   float = 0.10    # m.  수 cm 추종오차를 구별하는 폭
    sigma_course: float = 0.35  # rad.  이동 방향오차 정규화 폭
    course_gate: bool = True
    heading_gate: bool = False  # legacy 비교군에만 차체 기수 게이트 적용


PRESETS = {
    "speed":    RewardCfg(w_prog=0.30, w_align=0.90, w_resid=0.02, r_goal=25.0),
    "safe":     RewardCfg(w_prog=0.10, w_align=0.90, sigma_y=0.08,
                          w_resid=0.04),
    "balanced": RewardCfg(),
    "position_only": RewardCfg(w_lat=0.0, w_course=0.0, course_gate=False),
    "legacy":   RewardCfg(w_prog=1.0, w_align=0.80, w_resid=0.0,
                          w_lat=0.0, w_course=0.0, sigma_y=0.30,
                          course_gate=False, heading_gate=True),
}


def tracking_terms(prog, e_y, e_psi, action, prev_action, prev2_action, cfg,
                   *, course_error=0.0, movement=0.0):
    """참값 위치·이동방향으로 논문식 항을 계산한다. movement 는 기준속도 대비 이동량."""
    if cfg.sigma_y <= 0 or not math.isfinite(cfg.sigma_y):
        raise ValueError("sigma_y 는 양의 유한한 숫자여야 합니다")
    if cfg.sigma_course <= 0 or not math.isfinite(cfg.sigma_course):
        raise ValueError("sigma_course 는 양의 유한한 숫자여야 합니다")
    lateral_error_sq = (e_y / cfg.sigma_y) ** 2
    lateral_gate = math.exp(-lateral_error_sq)
    gate = lateral_gate
    if cfg.course_gate:
        gate *= max(0.0, math.cos(course_error))
    if cfg.heading_gate:
        gate *= max(0.0, math.cos(e_psi))
    moving = float(np.clip(movement, 0.0, 1.0))
    # 정지 중에는 이동 방향이 정의되지 않는다. 횡오차·방향 벌점도 정지 중
    # 누적되지 않게 하여, 일찍 실패해서 비용을 끊는 해를 만들지 않는다.
    course_error_sq = min((course_error / cfg.sigma_course) ** 2, 4.0)
    a = np.asarray(action, dtype=float)
    d2 = a - 2.0 * np.asarray(prev_action) + np.asarray(prev2_action)
    terms = {
        "prog": cfg.w_prog * prog,
        "align": cfg.w_align * prog * gate,
        "lat": -cfg.w_lat * moving * min(lateral_error_sq, 4.0),
        "course": -cfg.w_course * moving * lateral_gate * course_error_sq,
        "smooth": -cfg.w_smooth * float(np.mean(d2 ** 2)),
        "resid": -cfg.w_resid * float(np.mean(a ** 2)),
    }
    return gate, terms
