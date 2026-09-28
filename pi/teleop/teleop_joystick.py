#!/usr/bin/env python3
"""
KSRC 스워브 로버용 화면 마우스 조이스틱 텔레옵.

패드 둘:
  왼쪽  (마우스 드래그) -> 병진 (vx, vy)
  오른쪽(마우스 드래그) -> 회전 (omega), 가로축만

*** Q/E 키 회전이 따로 있는 이유 ***
마우스 커서는 하나라 패드 둘을 동시에 끌 수 없다. 오른쪽 패드로 회전을 유지한
채 왼쪽 패드로 병진을 주는 게 불가능하다는 뜻이다. 그래서 Q/E 가 omega 로 가는
두 번째 경로를 제공한다 (터미널 기반 teleop_keyboard.py 와 달리 pygame 은 진짜
키다운/키업 이벤트를 주므로 **누르고 있는 동안** 적용된다).
최종 omega = 오른쪽 패드를 끄는 중이면 그 값, 아니면 Q/E 로 누르고 있는 값,
아니면 0. 즉 마우스로 왼쪽 패드를 끌면서 Q/E 를 눌러 **호를 그리며 주행**할 수 있다.

두 패드 모두 마우스를 떼면 (0,0) 으로 돌아온다 -- "명령하지 않는 중" 이라는
모호하지 않은 안전 상태다. 스페이스바를 누르거나 창 포커스를 잃으면 전부 즉시
0 이 된다 (비상정지 / 알트탭 안전장치).

와이어 포맷 두 가지 (--format):
  ascii  -> "V <vx> <vy> <omega>\\n" 줄. STM32 펌웨어의 터미널 파서가 그대로
            이해한다 (펌웨어 수정 불필요). 실제 로버는 이걸 쓴다.
  binary -> common/nucleo_link.py 의 프레임 프로토콜. 시뮬레이터의 가상
            Nucleo 가 쓰는 포맷.

사용법:
  python3 teleop_joystick.py --port /dev/ttyACM0                 # 실제 로버
  python3 teleop_joystick.py --port /dev/pts/N --format binary   # 시뮬레이터
  python3 teleop_joystick.py --ros --max-lin 0.25 --max-ang 0.8 # ROS2 대회맵
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # pi/ -> import common.*
from common.nucleo_link import encode_velocity_cmd
from joystick_math import clamp_to_unit_circle, left_pad_to_linear, right_pad_to_angular

WIDTH, HEIGHT = 640, 380
PAD_RADIUS = 100
LEFT_CENTER = (170, 190)
RIGHT_CENTER = (470, 190)
GRAB_SLACK = 1.4  # click within radius*GRAB_SLACK of center still grabs the pad


class Pad:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.dragging = False
        self.nx = 0.0
        self.ny = 0.0

    def contains(self, pos):
        dx = pos[0] - self.center[0]
        dy = pos[1] - self.center[1]
        return math.hypot(dx, dy) <= self.radius * GRAB_SLACK

    def start_drag(self, pos):
        self.dragging = True
        self.update(pos)

    def update(self, pos):
        dx = pos[0] - self.center[0]
        dy = pos[1] - self.center[1]
        self.nx, self.ny = clamp_to_unit_circle(dx, dy, self.radius)

    def release(self):
        self.dragging = False
        self.nx = 0.0
        self.ny = 0.0

    def handle_pixel_pos(self):
        return (self.center[0] + self.nx * self.radius, self.center[1] + self.ny * self.radius)


def make_frame_encoder(fmt):
    """Returns encode(vx, vy, omega) -> bytes for the chosen wire format."""
    if fmt == "ascii":
        return lambda vx, vy, omega: f"V {vx:.3f} {vy:.3f} {omega:.3f}\n".encode("ascii")
    return encode_velocity_cmd


class RosSink:
    """Publish the same body twist as the serial teleop to the MuJoCo bridge."""

    def __init__(self, topic):
        import rclpy
        from geometry_msgs.msg import Twist

        rclpy.init()
        self.rclpy = rclpy
        self.Twist = Twist
        self.node = rclpy.create_node("ksrc_teleop_joystick")
        self.pub = self.node.create_publisher(Twist, topic, 10)

    def send(self, vx, vy, omega):
        msg = self.Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(omega)
        self.pub.publish(msg)

    def close(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


def run(args):
    import pygame  # local import so this module stays importable headlessly for tooling

    # sim/test_e2e_teleop.py 는 이 필드 없이 자기 Args 객체를 만들어 가상
    # Nucleo 를 구동하는데, 그쪽은 바이너리 프로토콜을 쓴다. 따라서 속성이
    # 없으면 "binary" 를 뜻해야 한다.
    ros_mode = getattr(args, "ros", False)
    if ros_mode:
        sink = RosSink(getattr(args, "topic", "/cmd_vel"))
        send = sink.send
    else:
        import serial
        encode = make_frame_encoder(getattr(args, "format", "binary"))
        ser = serial.Serial(args.port, args.baud, timeout=0)
        send = lambda vx, vy, omega: ser.write(encode(vx, vy, omega))

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("KSRC swerve teleop (joystick)")
    font = pygame.font.SysFont(None, 22)
    clock = pygame.time.Clock()

    left_pad = Pad(LEFT_CENTER, PAD_RADIUS)
    right_pad = Pad(RIGHT_CENTER, PAD_RADIUS)

    period = 1.0 / args.rate
    next_send = time.monotonic()
    running = True

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if left_pad.contains(event.pos):
                        left_pad.start_drag(event.pos)
                    elif right_pad.contains(event.pos):
                        right_pad.start_drag(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    left_pad.release()
                    right_pad.release()
                elif event.type == pygame.MOUSEMOTION:
                    if left_pad.dragging:
                        left_pad.update(event.pos)
                    if right_pad.dragging:
                        right_pad.update(event.pos)
                elif event.type in (pygame.WINDOWFOCUSLOST, pygame.ACTIVEEVENT):
                    left_pad.release()
                    right_pad.release()

            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                left_pad.release()
                right_pad.release()

            vx, vy = left_pad_to_linear(left_pad.nx, left_pad.ny, args.max_lin)
            if right_pad.dragging:
                omega = right_pad_to_angular(right_pad.nx, args.max_ang)
            elif keys[pygame.K_q] and not keys[pygame.K_e]:
                omega = args.max_ang
            elif keys[pygame.K_e] and not keys[pygame.K_q]:
                omega = -args.max_ang
            else:
                omega = 0.0
            if keys[pygame.K_SPACE]:
                vx = vy = omega = 0.0

            now = time.monotonic()
            if now >= next_send:
                send(vx, vy, omega)
                next_send = now + period

            screen.fill((24, 24, 28))
            for pad, label in ((left_pad, "translate"), (right_pad, "rotate")):
                pygame.draw.circle(screen, (70, 70, 80), pad.center, pad.radius, 2)
                pygame.draw.circle(screen, (60, 60, 68), pad.center, pad.radius, 0)
                color = (90, 200, 255) if pad.dragging else (140, 140, 150)
                pygame.draw.circle(screen, color, [int(c) for c in pad.handle_pixel_pos()], 18)
                text = font.render(label, True, (150, 150, 160))
                screen.blit(text, (pad.center[0] - text.get_width() // 2, pad.center[1] + pad.radius + 10))

            readout = f"vx={vx:+.2f} m/s  vy={vy:+.2f} m/s  omega={omega:+.2f} rad/s"
            help_text = "drag pads with mouse | Q/E rotate | space=stop | Esc=quit"
            screen.blit(font.render(readout, True, (220, 220, 230)), (16, 16))
            screen.blit(font.render(help_text, True, (140, 140, 150)), (16, 44))

            pygame.display.flip()
            clock.tick(60)
    finally:
        for _ in range(5):
            send(0.0, 0.0, 0.0)
            time.sleep(0.02)
        if ros_mode:
            sink.close()
        else:
            ser.close()
        pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0",
                    help="serial device to the Nucleo (ST-Link VCP is usually /dev/ttyACM0)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--rate", type=float, default=20.0,
                    help="command send rate, Hz. Above ~30 the firmware's line parser "
                         "starts dropping bytes while it services a command.")
    ap.add_argument("--format", choices=("ascii", "binary"), default="ascii",
                    help="ascii: 'V vx vy omega' lines for the STM32 terminal parser "
                         "(real rover). binary: framed protocol for the simulator.")
    ap.add_argument("--ros", action="store_true",
                    help="serial 대신 ROS2 Twist 를 /cmd_vel 로 발행")
    ap.add_argument("--topic", default="/cmd_vel", help="--ros 발행 토픽")
    ap.add_argument("--max-lin", type=float, default=0.5, help="pad-full-deflection |vx|,|vy| in m/s")
    ap.add_argument("--max-ang", type=float, default=2.0, help="pad-full-deflection |omega| in rad/s")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
