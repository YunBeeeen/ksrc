"""
라즈베리파이 -> Nucleo 시리얼 링크: 프레이밍, CRC, 메시지 인코딩.

firmware/common/teleop_protocol.h/.c 와 **바이트 단위로 동일**하다. 한쪽을
고치면 반드시 다른 쪽도 고치고 재검증해야 한다. 안 그러면 Nucleo 가 모든
프레임을 조용히 버린다 (에러도 안 난다).

프레임워크 의존성이 없는 순수 파이썬이다. ROS2 도, 시리얼 I/O 도 여기 없다.
Nucleo 와 대화해야 하는 쪽(지금은 텔레옵 스크립트, 나중에는 Nav2 cmd_vel
브리지 노드)이 이걸 import 하고 자기 pyserial.Serial 을 직접 연다.
"""
import struct

SYNC0 = 0xAA
SYNC1 = 0x55
MSG_VELOCITY_CMD = 0x01


def crc8(data: bytes) -> int:
    """CRC-8. poly 0x07, init 0x00, reflect/xorout 없음.
    firmware/common/teleop_protocol.c 의 teleop_crc8() 과 비트 단위로 동일."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def encode_velocity_cmd(vx: float, vy: float, omega: float) -> bytes:
    """차체 좌표계 속도 명령 프레임. vx, vy 는 m/s (x=전진, y=좌측),
    omega 는 rad/s (+가 반시계). swerve_kinematics.h 와 같은 규약."""
    payload = struct.pack("<fff", vx, vy, omega)
    body = bytes([MSG_VELOCITY_CMD, len(payload)]) + payload
    return bytes([SYNC0, SYNC1]) + body + bytes([crc8(body)])
