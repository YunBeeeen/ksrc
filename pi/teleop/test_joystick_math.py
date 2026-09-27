#!/usr/bin/env python3
"""조이스틱 수학 유닛테스트. 실행: python3 test_joystick_math.py"""
import math

from joystick_math import clamp_to_unit_circle, left_pad_to_linear, right_pad_to_angular

g_fail = False


def expect_near(label, got, want, tol=1e-6):
    global g_fail
    if abs(got - want) > tol:
        print(f"FAIL {label}: got {got:.4f} want {want:.4f}")
        g_fail = True
    else:
        print(f"ok   {label}: {got:.4f}")


# 1) 원 안쪽: 클램프 없음
nx, ny = clamp_to_unit_circle(50, 0, 100)
expect_near("inside.nx", nx, 0.5)
expect_near("inside.ny", ny, 0.0)

# 2) 축 방향으로 가장자리 밖까지 끌기: 정확히 1.0 으로 클램프
nx, ny = clamp_to_unit_circle(150, 0, 100)
expect_near("clamp_straight.nx", nx, 1.0)
expect_near("clamp_straight.ny", ny, 0.0)

# 3) 패드 사각형 모서리까지 대각선으로 끌기: 방향은 유지, 크기만 1 로 클램프
nx, ny = clamp_to_unit_circle(100, 100, 100)
expect_near("clamp_diag.nx", nx, math.sqrt(2) / 2)
expect_near("clamp_diag.ny", ny, math.sqrt(2) / 2)
expect_near("clamp_diag.mag", math.hypot(nx, ny), 1.0)

# 4) 반경이 0 인 퇴화 케이스
nx, ny = clamp_to_unit_circle(10, 10, 0)
expect_near("zero_radius.nx", nx, 0.0)
expect_near("zero_radius.ny", ny, 0.0)

# 5) 왼쪽 패드 -> 병진 속도, 상하좌우 네 방향
vx, vy = left_pad_to_linear(0.0, -1.0, 2.0)  # 끝까지 위
expect_near("left.up.vx", vx, 2.0)
expect_near("left.up.vy", vy, 0.0)

vx, vy = left_pad_to_linear(0.0, 1.0, 2.0)  # 끝까지 아래
expect_near("left.down.vx", vx, -2.0)

vx, vy = left_pad_to_linear(-1.0, 0.0, 2.0)  # 끝까지 왼쪽
expect_near("left.left.vy", vy, 2.0)
expect_near("left.left.vx", vx, 0.0)

vx, vy = left_pad_to_linear(1.0, 0.0, 2.0)  # 끝까지 오른쪽
expect_near("left.right.vy", vy, -2.0)

# 6) 오른쪽 패드 -> 각속도, 가로축만
expect_near("right.center", right_pad_to_angular(0.0, 3.0), 0.0)
expect_near("right.full_right_is_cw", right_pad_to_angular(1.0, 3.0), -3.0)
expect_near("right.full_left_is_ccw", right_pad_to_angular(-1.0, 3.0), 3.0)

if g_fail:
    print("\nSOME TESTS FAILED")
    raise SystemExit(1)
print("\nALL TESTS PASSED")
