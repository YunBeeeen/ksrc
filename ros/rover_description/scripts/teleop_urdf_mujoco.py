"""URDF 를 MuJoCo 로 바로 띄워 펌웨어 IK 로 몰아 본다 (2026-10-15, rev5 URDF 확인용).

URDF -> MuJoCo 변환 (freejoint 바닥, 로커 1:1 역방향 구속, 조향 위치·바퀴 속도 액추에이터) 후
명령 (vx, vy, w) 를 펌웨어와 같은 C IK (sim/libksrc_common.so: swerve_ik_compute + fold ±90°) 로
조향각·바퀴 속도로 바꿔 넣는다.  실물은 URDF 를 안 쓰므로, 여기서 명령과 같은 방향으로 움직이면
URDF 의 조인트 부호·이름·위치가 펌웨어 규약과 맞는다는 뜻이다.

모델은 이름(rev4 / rev5orig / rev5) 또는 URDF 경로로 준다:
  rev4      rover_portable.urdf                              지금 Isaac·Nav2 가 쓰는 것
  rev5orig  urdf/rocker_bogie_4wd_rover_rev5_original.urdf   받은 원본 그대로 (조인트 이름·축 오류 포함)
  rev5      rover_rev5.urdf                                  fix_rev5.py 로 고친 것

  # 텔레옵: MuJoCo 창 + 화면 조이스틱 창(pi/teleop/teleop_joystick.py, 실기와 같은 UDP) 같이 뜸
  #   몸체 좌표축(빨강 x 앞 · 초록 y 왼쪽 · 파랑 z 위)과 명령/실제 속도 화살표 표시
  python3 scripts/teleop_urdf_mujoco.py rev5
  python3 scripts/teleop_urdf_mujoco.py rev5orig
  # 자동 검사: 전진 / 후진 / 옆걸음 / 회전 / 대각 / 회전+전진, 명령 vs 실제 몸체 속도
  python3 scripts/teleop_urdf_mujoco.py --check rev4 rev5orig rev5
면이 20만 개를 넘는 메쉬(rev5 base_link)는 MuJoCo 한도라 불러올 때 볼록 껍질로 바꾼다 (충돌은 원래 볼록 껍질).
"""
import argparse
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[3]          # ~/ksrc
sys.path.insert(0, str(ROOT / "sim"))
import ksrc_common as kc  # noqa: E402

MODULES = ("fl", "fr", "rl", "rr")
MODULE_XY = ((0.102431, 0.1325), (0.102431, -0.1325), (-0.102431, 0.1325), (-0.102431, -0.1325))
MAX_WHEEL_MPS = 0.378
R_WHEEL = 0.047503
STEER_LIM = math.radians(90.0)
CTRL_DT = 0.02
DESC = Path(__file__).resolve().parents[1]           # ros/rover_description
MODELS = {"rev4": DESC / "rover_portable.urdf",
          "rev5orig": DESC / "urdf/rocker_bogie_4wd_rover_rev5_original.urdf",
          "rev5": DESC / "rover_rev5.urdf"}


def _stl_bytes(path: Path) -> bytes:
    """STL bytes; meshes over MuJoCo's 200k-face limit become their convex hull."""
    import struct
    raw = path.read_bytes()
    n = struct.unpack("<I", raw[80:84])[0] if len(raw) >= 84 else 0
    if len(raw) != 84 + 50 * n or n <= 200000:
        return raw
    from scipy.spatial import ConvexHull
    dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    v = np.unique(np.frombuffer(raw, dt, count=n, offset=84)["v"].reshape(-1, 3), axis=0)
    tri = v[ConvexHull(v).simplices].astype("<f4")
    rec = np.zeros(len(tri), dt); rec["v"] = tri
    print(f"[mesh] {path.name}: {n} 면 -> 볼록 껍질 {len(tri)} 면 (MuJoCo 한도)")
    return b"hull".ljust(80, b" ") + struct.pack("<I", len(tri)) + rec.tobytes()


def build(urdf_path: Path):
    """URDF -> (MjModel, MjData).  Meshes are found by basename next to the URDF (recursive)."""
    urdf_path = Path(MODELS.get(str(urdf_path), urdf_path)).resolve()
    txt = urdf_path.read_text()
    assets = {}
    for fn in set(re.findall(r'filename="([^"]+)"', txt)):
        base = os.path.basename(fn)
        sub = "rev5" if ("rev5" in urdf_path.name or "rocker_bogie" in urdf_path.name) else ""
        hits = [p for p in DESC.rglob(base) if (sub in p.parent.name) == bool(sub)] or list(urdf_path.parent.rglob(base))
        if not hits:
            raise FileNotFoundError(f"{base} (URDF {urdf_path.name})")
        assets[base] = _stl_bytes(hits[0])
    comp = '<mujoco><compiler strippath="true" discardvisual="true" fusestatic="false" balanceinertia="true"/></mujoco>'
    txt = re.sub(r"(<robot[^>]*>)", r"\1" + comp, txt, count=1)
    m0 = mujoco.MjModel.from_xml_string(txt, assets)
    tmp = Path("/tmp") / f"_urdf2mj_{os.getpid()}.xml"
    mujoco.mj_saveLastXML(str(tmp), m0)
    root = ET.parse(tmp).getroot(); tmp.unlink()
    wb = root.find("worldbody")
    base = wb.find("body")
    if base.find("freejoint") is None and base.find("joint") is None:
        ET.SubElement(base, "freejoint", {"name": "root"})
        base.insert(0, base[-1])
        del base[-1]
    base.set("pos", "0 0 0.25")
    ET.SubElement(wb, "geom", {"name": "floor", "type": "plane", "size": "5 5 0.1", "rgba": ".6 .6 .55 1",
                                "friction": "1.0 0.005 0.0001"})
    ET.SubElement(wb, "light", {"pos": "0 0 3", "dir": "0 0 -1", "directional": "true"})
    opt = root.find("option") if root.find("option") is not None else ET.SubElement(root, "option")
    opt.set("timestep", "0.002")
    for j in root.iter("joint"):
        n = j.get("name", "")
        if "wheel" in n:
            j.set("armature", "0.0098"); j.set("damping", "0.0")
        elif "steering" in n:
            j.set("armature", "0.01"); j.set("damping", "0.3")
        elif "rocker" in n:
            j.set("damping", "0.05")
    eq = root.find("equality") if root.find("equality") is not None else ET.SubElement(root, "equality")
    ET.SubElement(eq, "joint", {"joint1": "right_rocker_joint", "joint2": "left_rocker_joint", "polycoef": "0 -1 0 0 0"})
    act = ET.SubElement(root, "actuator")
    for mname in MODULES:
        ET.SubElement(act, "position", {"name": f"{mname}_steer", "joint": f"{mname}_steering_joint", "kp": "30",
                                        "forcerange": "-2.94 2.94", "forcelimited": "true"})
    for mname in MODULES:
        ET.SubElement(act, "velocity", {"name": f"{mname}_drive", "joint": f"{mname}_wheel_joint", "kv": "1.0",
                                        "forcerange": "-1.96 1.96", "forcelimited": "true"})
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets)
    return m, mujoco.MjData(m)


class Rover:
    def __init__(self, urdf):
        self.name = f"{urdf} ({Path(MODELS.get(str(urdf), urdf)).name})"
        self.m, self.d = build(urdf)
        self.mods = (kc.SwerveModulePos * 4)(*[kc.SwerveModulePos(x, y) for x, y in MODULE_XY])
        self.state = (kc.SwerveModuleState * 4)()
        self.sid = [self.m.actuator(f"{n}_steer").id for n in MODULES]
        self.wid = [self.m.actuator(f"{n}_drive").id for n in MODULES]
        self.sj = [self.m.joint(f"{n}_steering_joint").qposadr[0] for n in MODULES]
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.state = (kc.SwerveModuleState * 4)()
        for _ in range(500):
            mujoco.mj_step(self.m, self.d)

    def command(self, vx, vy, w):
        out = kc.swerve_ik_compute(vx, vy, w, self.mods, MAX_WHEEL_MPS, self.state)
        kc.swerve_fold_to_range([-STEER_LIM] * 4, [STEER_LIM] * 4, self.state, out)
        for k in range(4):
            self.d.ctrl[self.sid[k]] = out[k].angle_rad
            self.d.ctrl[self.wid[k]] = out[k].speed_mps / R_WHEEL
        for _ in range(int(CTRL_DT / self.m.opt.timestep)):
            mujoco.mj_step(self.m, self.d)

    def body_vel(self):
        q = self.d.qpos[3:7]; R = np.zeros(9); mujoco.mju_quat2Mat(R, q); R = R.reshape(3, 3)
        v = R.T @ self.d.qvel[0:3]
        return v[0], v[1], self.d.qvel[5]

    def steer(self):
        return np.degrees([self.d.qpos[a] for a in self.sj])


def check(urdfs):
    cases = [("전진", (0.15, 0, 0)), ("후진", (-0.15, 0, 0)), ("왼쪽 옆걸음", (0.01, 0.12, 0)),
             ("반시계 회전", (0, 0, 0.6)), ("시계 회전", (0, 0, -0.6)), ("왼앞 대각", (0.1, 0.1, 0.0)),
             ("전진+좌회전", (0.15, 0, 0.4))]
    for u in urdfs:
        r = Rover(u)
        print(f"\n== {r.name}  질량 {r.m.body_subtreemass[1]:.3f} kg")
        print(f"{'명령':<10} {'cmd vx vy w':>20} {'실제 vx vy w':>24}  조향각(fl fr rl rr) 판정")
        for name, c in cases:
            r.reset()
            vs = []
            for k in range(150):                      # 3 s, 뒤 1.5 s 평균
                r.command(*c)
                if k >= 75:
                    vs.append(r.body_vel())
            v = np.mean(vs, 0)
            big = max(abs(c[0]), abs(c[1]), 0.1 * abs(c[2]))
            ok = all((abs(cc) < 0.02 and abs(vv) < 0.35 * big + 0.02) or (np.sign(cc) == np.sign(vv) and abs(vv) > 0.5 * abs(cc))
                     for cc, vv in zip(c, v))
            print(f"{name:<10} {c[0]:+6.2f} {c[1]:+6.2f} {c[2]:+6.2f}   {v[0]:+7.3f} {v[1]:+7.3f} {v[2]:+7.3f}  "
                  f"{np.round(r.steer(), 0)} {'OK' if ok else '** 틀림 **'}")


AXIS_L = 0.15        # 몸체 좌표축 화살표 길이 [m] (로버 반폭 ~0.15)


def _arrow(scn, p0, p1, rgba, width=0.006, label=""):
    if scn.ngeom >= scn.maxgeom or np.linalg.norm(np.subtract(p1, p0)) < 1e-4:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(p0, float), np.asarray(p1, float))
    if label:
        g.label = label
    scn.ngeom += 1


def draw_frames(scn, r, cmd):
    """몸체 좌표축 (x 빨강 앞 · y 초록 왼쪽 · z 파랑 위, 원점 = base_link) + 속도 화살표
    (하늘색 명령 · 주황 실제, 1 m/s = 1 m, 몸체 위 12 cm) + 바닥 월드 축 (흐린 색)."""
    scn.ngeom = 0
    p = r.d.qpos[:3].copy()
    R = np.zeros(9); mujoco.mju_quat2Mat(R, r.d.qpos[3:7]); R = R.reshape(3, 3)
    for k, (c, lab) in enumerate((((1, .1, .1, 1), "x"), ((.1, .9, .1, 1), "y"), ((.2, .3, 1, 1), "z"))):
        _arrow(scn, p, p + AXIS_L * R[:, k], c, label=lab)
    top = p + R @ np.array([0, 0, 0.12])
    _arrow(scn, top, top + R @ np.array([cmd[0], cmd[1], 0.0]), (.2, .9, 1, .9), 0.008, "cmd")
    vw = r.d.qvel[0:3].copy(); vw[2] = 0
    _arrow(scn, top, top + vw, (1, .55, .1, .9), 0.008, "real")
    for k, c in enumerate(((1, .4, .4, .5), (.4, 1, .4, .5), (.4, .5, 1, .5))):
        e = np.zeros(3); e[k] = 0.3
        _arrow(scn, (0, 0, 0.001), e + (0, 0, 0.001), c, 0.004)


def teleop(urdf, joy=True, port=5005, max_lin=0.15, max_ang=0.8):
    import socket
    import struct
    import subprocess
    import mujoco.viewer
    r = Rover(urdf)
    cmd = np.zeros(3); st = {"reset": False, "t_udp": 0.0, "key": False}
    K = {265: (0, +0.03), 264: (0, -0.03), 263: (2, +0.2), 262: (2, -0.2), 74: (1, +0.03), 76: (1, -0.03)}

    def on_key(k):
        if k in K:
            st["key"] = True
            i, dv = K[k]; cmd[i] = np.clip(cmd[i] + dv, -[0.3, 0.25, 1.2][i], [0.3, 0.25, 1.2][i])
        elif k == 32:
            cmd[:] = 0
        elif k == 82:
            st["reset"] = True

    pkt = struct.Struct("<4sIfff")          # pi/common/udp_spi_bridge.py 와 같은 형식
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port)); sock.setblocking(False)
    child = None
    if joy:
        js = ROOT / "pi/teleop/teleop_joystick.py"
        child = subprocess.Popen([sys.executable, str(js), "--udp", f"127.0.0.1:{port}", "--rate", "50",
                                  "--max-lin", str(max_lin), "--max-ang", str(max_ang)])
        print(f"[joy] 화면 조이스틱 창을 띄움 (실기와 같은 UDP {port}, max-lin {max_lin}, max-ang {max_ang})")
    print("조이스틱 창: 왼쪽 패드 병진 · 오른쪽 패드/Q·E 회전 · 스페이스 정지.  MuJoCo 창 키: R 리셋 (↑↓←→ J L 도 됨)")
    print("화살표: 빨강 x(앞) · 초록 y(왼쪽) · 파랑 z(위) = 몸체 좌표 / 하늘색 명령 속도 · 주황 실제 속도 (1 m/s = 1 m)")
    try:
        with mujoco.viewer.launch_passive(r.m, r.d, key_callback=on_key) as v:
            v.cam.distance = 1.2; v.cam.elevation = -35
            t_print = 0
            while v.is_running():
                t0 = time.time()
                while True:
                    try:
                        data = sock.recv(64)
                    except BlockingIOError:
                        break
                    if len(data) == pkt.size:
                        magic, _, vx, vy, w = pkt.unpack(data)
                        if magic == b"KSRC" and all(map(math.isfinite, (vx, vy, w))):
                            cmd[:] = (vx, vy, w); st["t_udp"] = t0; st["key"] = False
                if not st["key"] and st["t_udp"] and t0 - st["t_udp"] > 0.5:
                    cmd[:] = 0                                  # 워치독 (실기 0.5 s 와 같음)
                if st["reset"]:
                    st["reset"] = False; cmd[:] = 0; r.reset()
                r.command(*cmd)
                v.cam.lookat[:] = r.d.qpos[:3]
                with v.lock():
                    draw_frames(v.user_scn, r, cmd)
                v.sync()
                if time.time() - t_print > 0.5:
                    t_print = time.time(); bv = r.body_vel()
                    print(f"\r명령 vx {cmd[0]:+.2f} vy {cmd[1]:+.2f} w {cmd[2]:+.2f} | 실제(몸체) {bv[0]:+.2f} {bv[1]:+.2f} "
                          f"{bv[2]:+.2f} | 조향 {np.round(r.steer(), 0)}   ", end="", flush=True)
                time.sleep(max(0.0, CTRL_DT - (time.time() - t0)))
    finally:
        sock.close()
        if child is not None:
            child.terminate()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urdf", nargs="+")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-joy", action="store_true", help="화면 조이스틱 창 안 띄움 (키보드만, 또는 따로 띄운 조이스틱)")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--max-lin", type=float, default=0.15)
    ap.add_argument("--max-ang", type=float, default=0.8)
    a = ap.parse_args()
    if a.check:
        check(a.urdf)
    else:
        teleop(a.urdf[0], joy=not a.no_joy, port=a.port, max_lin=a.max_lin, max_ang=a.max_ang)
