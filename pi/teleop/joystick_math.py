"""
화면 조이스틱 패드의 순수 수학 부분.

패드 중심으로부터의 픽셀 변위 -> [-1,1] 정규화 스틱 위치 -> 차체 속도.
pygame 도 I/O 도 없어서 **디스플레이 없이 유닛테스트가 된다**
(test_joystick_math.py 참고).

패드 둘:
  왼쪽  -> 병진 (vx, vy)   [스워브는 홀로노믹이라 옆으로 가려고 차체를
                            돌릴 필요가 없다]
  오른쪽 -> 회전 (omega), 가로축만 사용

차체 좌표계는 firmware/common/swerve_kinematics.h 와 동일:
x=전진, y=좌측, omega 는 +가 반시계.
"""
import math


def clamp_to_unit_circle(dx: float, dy: float, radius: float):
    """dx, dy 는 패드 중심으로부터의 픽셀 변위. (nx, ny) 를 돌려준다.
    각각 [-1, 1] 범위이며 **단위 원**(정사각형이 아니라)으로 클램프한다.
    그래야 대각선으로 끈 게 직선으로 끈 것보다 크게 잡히지 않는다. out to the pad's edge."""
    if radius <= 0:
        return 0.0, 0.0
    nx = dx / radius
    ny = dy / radius
    mag = math.hypot(nx, ny)
    if mag > 1.0:
        nx /= mag
        ny /= mag
    return nx, ny


def left_pad_to_linear(nx: float, ny: float, max_lin_mps: float):
    """화면 위 = 전진(+vx), 화면 왼쪽 = +vy (차체 y 는 좌측이 +)."""
    vx = -ny * max_lin_mps
    vy = -nx * max_lin_mps
    return vx, vy


def right_pad_to_angular(nx: float, max_ang_rps: float):
    """오른쪽으로 끌면 시계방향 = omega 음수 (차체 omega 는 반시계가 +)."""
    return -nx * max_ang_rps
