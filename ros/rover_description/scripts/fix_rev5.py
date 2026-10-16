"""rocker_bogie_4wd_rover.urdf (rev5, 10-14) -> rover_rev5.urdf with our conventions.

Fixes (geometry/mass untouched):
  1. right steering joint names were swapped (rr_steering_joint drove the FRONT-right leg)
  2. steering axes were -z (positive = clockwise) -> +z (positive = CCW, as IK / firmware / old URDF)
  3. left wheel axes pointed -y in the base frame -> every wheel +y (positive = forward roll, as before)
  4. base_link origin moved 18.3 mm back onto the rocker-pivot / module center (old URDF convention:
     modules at x = +-0.1024, which the firmware IK and contract.MODULE_XY_M use)
  5. link names back to the old ones (code finds wheel bodies by name: *_wheel_link)
  6. joint effort/velocity/damping and steering limit +-90 deg as the old URDF
  7. wheel meshes re-centred: the exported wheel_N.stl origin is 4.5 mm off the tyre centre, so the tyre sat
     4.5 mm outboard of the steering axis.  The full assembly rev5.stl has the tread at |y| 114.8-150.2 mm,
     centre 132.5 = steering axis (2026-10-15).  Shift visual/collision/inertial along the wheel y axis so the
     tread (outer 2 mm band) is centred on the joint.
Meshes: meshes/rev5/*.stl (mm, scale 0.001).
"""
import sys, xml.etree.ElementTree as ET

src, dst, meshrel = sys.argv[1], sys.argv[2], sys.argv[3]
tree = ET.parse(src); r = tree.getroot()
DX = 0.01830665319          # rocker pivot x in the rev5 base frame (negative) -> new origin there

ren_link = {"left_rocker_arm": "left_rocker_link", "right_rocker_arm": "right_rocker_link",
            "fl_leg": "fl_steering_link", "fr_leg": "fr_steering_link",
            "rl_leg": "rl_steering_link", "rr_leg": "rr_steering_link",
            "fl_wheel": "fl_wheel_link", "fr_wheel": "fr_wheel_link",
            "rl_wheel": "rl_wheel_link", "rr_wheel": "rr_wheel_link"}
for l in r.findall("link"):
    l.set("name", ren_link.get(l.get("name"), l.get("name")))
    for m in l.iter("mesh"):
        m.set("filename", meshrel + "/" + m.get("filename").split("/")[-1])
for j in r.findall("joint"):
    for t in ("parent", "child"):
        e = j.find(t); e.set("link", ren_link.get(e.get("link"), e.get("link")))

def shift(el, dx):
    o = el.find("origin")
    x, y, z = [float(v) for v in o.get("xyz").split()]
    o.set("xyz", f"{x + dx:.11g} {y:.11g} {z:.11g}")

base = [l for l in r.findall("link") if l.get("name") == "base_link"][0]
for t in ("inertial", "visual", "collision"):
    for e in base.findall(t):
        shift(e, DX)

joints = {j.get("name"): j for j in r.findall("joint")}
# 1. names by the child leg (position), not as given
for j in r.findall("joint"):
    if j.get("type") == "revolute" and "steering" in j.get("name"):
        j.set("name", j.find("child").get("link").replace("_steering_link", "_steering_joint"))
for j in r.findall("joint"):
    n, c = j.get("name"), j.find("child").get("link")
    p = j.find("parent").get("link")
    if p == "base_link":
        shift(j, DX)                                   # rockers, imu, camera
    lim = j.find("limit")
    if "rocker" in n:
        lim.set("effort", "0"); lim.set("velocity", "3")
        d = j.find("dynamics"); d.set("damping", "0.01"); d.set("friction", "0")
    if "steering" in n:
        j.find("axis").set("xyz", "0 0 1")                                     # 2.
        lim.set("lower", "-1.5707963267948966"); lim.set("upper", "1.5707963267948966")
        lim.set("effort", "2.942"); lim.set("velocity", "4.0")
        d = j.find("dynamics") or ET.SubElement(j, "dynamics"); d.set("damping", "0.01"); d.set("friction", "0")
    if "wheel" in n:
        if n in ("fl_wheel_joint", "rl_wheel_joint"):                          # 3.
            ax = [-float(v) for v in j.find("axis").get("xyz").split()]
            j.find("axis").set("xyz", " ".join(f"{v:g}" for v in ax))
        lim = lim if lim is not None else ET.SubElement(j, "limit")
        lim.set("effort", "1.0"); lim.set("velocity", "8.0")
        d = j.find("dynamics") or ET.SubElement(j, "dynamics"); d.set("damping", "0.001"); d.set("friction", "0")
# 7. tread centre on the wheel joint (wheel frame y = rolling axis for all four in this export)
import os, struct
import numpy as np
mesh_root = os.path.join(os.path.dirname(os.path.abspath(dst)), meshrel)
for l in r.findall("link"):
    if not l.get("name").endswith("_wheel_link"):
        continue
    m = l.find("visual/geometry/mesh")
    fn = os.path.join(mesh_root, m.get("filename").split("/")[-1])
    raw = open(fn, "rb").read(); n = struct.unpack("<I", raw[80:84])[0]
    v = np.frombuffer(raw, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)["v"]
    v = v.reshape(-1, 3).astype(float) * float(m.get("scale", "1 1 1").split()[1])
    rad = np.hypot(v[:, 0], v[:, 2]); t = v[rad > rad.max() - 0.002]
    yc = 0.5 * (t[:, 1].min() + t[:, 1].max())
    for tag in ("inertial", "visual", "collision"):
        for e in l.findall(tag):
            o = e.find("origin"); x, y, z = [float(q) for q in o.get("xyz").split()]
            o.set("xyz", f"{x:.9g} {y - yc:.9g} {z:.9g}")
    print(f"  {l.get('name')}: tread centre y {yc * 1000:+.2f} mm in wheel frame -> shifted onto the joint")
r.insert(0, ET.Comment(" KSRC: generated by fix_rev5.py from rocker_bogie_4wd_rover.urdf (rev5 2026-10-14). "
                       "Fixes: right steering names swapped, steering axis +z, wheel axes +y, base origin at module "
                       "center (+18.3 mm), old link names, old joint limits (+-90 deg), wheel meshes re-centred on the steering "
                       "axis (export offset 4.5 mm; assembly rev5.stl has the tread centred). Masses unchanged. "))
ET.indent(tree)
tree.write(dst, encoding="unicode", xml_declaration=True)
print("wrote", dst)
