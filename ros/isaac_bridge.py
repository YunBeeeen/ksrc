"""MuJoCo(rev5) <-> ROS2 브리지, Isaac 학습 정책용 (2026-10-15).  RViz "2D Goal Pose" -> Nav2 -> /cmd_vel
-> (펌웨어 IK + 정책 잔차 + 정렬 게이트 + 바퀴 PI) -> MuJoCo.  예전 mujoco_bridge.py(SB3 정책) 대체.

로버·액추에이터·센서·관측은 sim/rl/sim2sim_isaac.py 와 같다 (rev5 원기둥 바퀴, 서보 모델, 펌웨어 PI·게이트,
엔코더·IMU 센서, 관측 지연·잡음).  지형은 대회 STL 높이맵(sand/rock, Nav2 맵과 같은 좌표) 또는 평지.
μ: --mu const:0.3 처럼 고정 (4바퀴 같은 μ = 고른 모래 가정) 또는 patch(학습 분포).  --soil sand 면 지반역학.
측위는 참값 (map->odom 항등), 시간은 wall time.

발행  /tf (map->odom 항등, odom->base_link), /odom, /joint_states (RViz 로버 모델), /rl_action, /bridge_status
구독  /cmd_vel
서비스 /set_rl (SetBool: 정책 잔차 켬/끔, 같은 주행 중 IK 와 바로 비교), /reset_sim (Trigger: 스폰으로)

  python3 ros/isaac_bridge.py --terrain sand --policy <run>/model_1000.pt --mu const:0.3 --view
"""
import argparse
import json
import math
import pathlib
import sys
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import SetBool, Trigger
from tf2_ros import TransformBroadcaster

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "sim" / "rl"))
import mujoco  # noqa: E402
import sim2sim_isaac as s2s  # noqa: E402
from sim2sim_isaac import (CONTROL_DT_S, CORNERS, ACTION_CLASSES, ActionMap, MjRover, ObsBuilder, Policy,  # noqa: E402
                           load_run_cfg, sample_mu, terr)

URDF_JOINTS = {"st": "{}_steering_joint", "wh": "{}_wheel_joint"}
STAND_V, STAND_W = 0.01, 0.02      # [m/s], [rad/s] 이하면 정지 명령 -> 잔차 0 (pi/policy/run_policy.py 와 같은 값)


class Jitter:
    """조향 떨림 (10-15): 4바퀴 조향의 스텝당 2차 차분 |θ[k] − 2θ[k−1] + θ[k−2]| 평균 [deg].
    등속 회전·일정 각도는 0 이고 스텝마다 뒤집히는 떨림만 커진다 (Isaac jerk_steer 와 같은 2차 차분).
    명령 = 정책+IK 가 낸 조향 목표 (지연 전), 실제 = MuJoCo 조향 관절각 (서보 지연·데드밴드·하중 후).
    IK 접기 반전처럼 한 스텝에 20° 넘게 바뀐 구간은 떨림이 아니라 빼고 센다."""
    FLIP = math.radians(20.0)

    def __init__(self):
        self.h = {"cmd": [], "meas": []}
        self.sum = {"cmd": 0.0, "meas": 0.0}
        self.n = {"cmd": 0, "meas": 0}

    def add(self, cmd, meas):
        for k, v in (("cmd", cmd), ("meas", meas)):
            h = self.h[k]
            h.append(np.asarray(v, float).copy())
            if len(h) > 3:
                h.pop(0)
            if len(h) == 3:
                d1a, d1b = h[1] - h[0], h[2] - h[1]
                ok = (np.abs(d1a) < self.FLIP) & (np.abs(d1b) < self.FLIP)
                if ok.any():
                    self.sum[k] += float(np.abs(d1b - d1a)[ok].mean()); self.n[k] += 1

    def take(self):
        """흔들림 진폭 ±[deg].  스텝마다 +a, −a 로 뒤집히는 떨림이면 2차 차분 = 4a 라서 /4 (로그를 각도로 바로 읽게)."""
        out = tuple(math.degrees(self.sum[k] / self.n[k]) / 4.0 if self.n[k] else 0.0 for k in ("cmd", "meas"))
        self.sum = {"cmd": 0.0, "meas": 0.0}; self.n = {"cmd": 0, "meas": 0}
        return out


class ArenaRover(MjRover):
    """MjRover on a fixed competition height map (or flat), spawning at a fixed pose."""

    def __init__(self, terrain, soil, rng, view, spawn, drive="pi", sensors="real", friction_duty=0.0, gate="new"):
        self.arena = terrain
        scen = ("rough_" if terrain != "flat" else "flat_") + ("sand" if soil == "sand" else "mu")
        # MjRover builds its hfield with terr.ARENA_X/Y (8 m training map): build it at the arena's size instead
        # (changing hfield_size after compiling leaves the collision geom at the old size).
        old = (terr.ARENA_X, terr.ARENA_Y)
        if terrain != "flat":
            terr.ARENA_X, terr.ARENA_Y = terr.eval_extent(terrain)
        try:
            super().__init__(scen, 0.005, rng, visual=view, drive=drive, sensors=sensors,
                             wheel_friction_duty=friction_duty, gate=gate)
        finally:
            terr.ARENA_X, terr.ARENA_Y = old
        self.spawn = spawn
        if terrain != "flat":
            ex, ey = terr.eval_extent(terrain)
            self.ex, self.ey = ex, ey
            self.Z = terr.load_arena(terrain, int(self.m.hfield_nrow[0]))
            if spawn is None:     # deepest point of the flat drivable area (far from steps and walls)
                from scipy.ndimage import distance_transform_edt
                drive_ok, _ = terr.drivable_mask(self.Z, ex, ey, max_slope_deg=3.0, max_step=self.cfg.max_step)
                xs, ys = terr.cell_centers(self.Z, ex, ey)
                dist = distance_transform_edt(np.pad(drive_ok, 1))[1:-1, 1:-1]   # map edge counts as a wall
                iy, ix = np.unravel_index(np.argmax(dist), dist.shape)
                self.spawn = (float(xs[ix]), float(ys[iy]), 0.0)
        elif spawn is None:
            self.spawn = (0.0, 0.0, 0.0)
        self.rock = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("rock_l", "rock_r")]

    def reset(self):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        x, y, yaw = self.spawn
        ground = 0.0
        if self.arena != "flat":
            z_off = terr.to_hfield(m, self.Z)
            ground = terr.local_top(self.Z, self.ex, self.ey, np.array([x, y])) + z_off
        d.qpos[0:2] = (x, y)
        d.qpos[2] += ground + 0.01
        d.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        if self.sand:
            self.tm.p = s2s.sample_terrain(self.rng, 0.5)
            self.tm.reset()
        mujoco.mj_forward(m, d)
        self.servo.reset(); self.pi.reset(); self.gate.reset()
        self.gate_hold = False
        self.gyro_imu = np.zeros(3)
        self.duty = np.zeros(4)
        self.ticks_prev = self._ticks()
        self.wheel_w_enc = np.zeros(4)
        self.gfilt.reset()
        self.q_prev = self.v_prev = None
        self.grav_imu = np.array([0.0, 0.0, -1.0])
        for _ in range(100):                      # settle on the ground with zero commands
            self.step(np.zeros(8))


class Bridge(Node):
    def __init__(self, a):
        super().__init__("isaac_bridge")
        self.a = a
        self.rng = np.random.default_rng(a.seed)
        spawn = tuple(a.spawn) if a.spawn else None
        self.rover = ArenaRover(a.terrain, a.soil, self.rng, a.view, spawn, friction_duty=a.friction_duty, gate=a.gate)
        self.env_cfg, agent_cfg = (None, None) if not a.policy else load_run_cfg(a.policy)
        action = ACTION_CLASSES[self.env_cfg["actions"]["rover"]["class_type"].split(":")[-1]] if self.env_cfg else "Res8"
        rov = ((self.env_cfg or {}).get("actions", {}) or {}).get("rover", {}) or {}
        # 학습 때 잔차 후처리 (변화율 제한) 그대로, 정지 명령 잔차 0 은 옛 정책에도 적용
        self.amap = ActionMap(action, a.act_delay, rov.get("rate_limit", 0.0), not a.no_stand_gate)
        self.get_logger().info(f"잔차 변화율 제한 {rov.get('rate_limit', 0.0)}/스텝, 정지 잔차 0 {not a.no_stand_gate}")
        self.policy = self.obs = None
        if a.policy:
            self.policy = Policy(a.policy, agent_cfg)
            self.obs = ObsBuilder(self.env_cfg, agent_cfg["obs_groups"]["policy"], a.obs_delay, a.noise, self.rng)
        self.rl_on = bool(a.policy) and not a.start_ik
        self.mu_state = {}
        self.cmd = np.zeros(3); self.cmd_t = 0.0
        self.lock = threading.Lock()
        self.viewer = None
        if a.view:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.rover.m, self.rover.d)
            self.viewer.cam.distance, self.viewer.cam.elevation = 1.5, -35
        self.do_reset()
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, qos)
        self.pub_odom = self.create_publisher(Odometry, "/odom", qos)
        self.pub_js = self.create_publisher(JointState, "/joint_states", qos)
        self.pub_act = self.create_publisher(Float32MultiArray, "/rl_action", qos)
        self.pub_stat = self.create_publisher(String, "/bridge_status", qos)
        self.create_service(SetBool, "/set_rl", self.on_set_rl)
        self.create_service(Trigger, "/reset_sim", self.on_reset)
        self.tf = TransformBroadcaster(self)
        self.acc = {"n": 0, "exy": 0.0, "eyaw": 0.0, "sq": 0.0, "hold": 0}
        self.jit = Jitter()
        self.t_print = time.monotonic()
        self.create_timer(CONTROL_DT_S, self.tick)
        self.get_logger().info(
            f"Isaac 브리지: rev5 MuJoCo, 지형 {a.terrain} (흙 {a.soil}), μ {a.mu}, 스폰 {np.round(self.rover.spawn, 2).tolist()}, "
            f"정책 {a.policy or '없음(IK)'} -> {'RL' if self.rl_on else 'IK'}, 게이트 {a.gate}.  /set_rl 로 전환")

    # ---------------------------------------------------------------- services / topics
    def on_cmd(self, m):
        with self.lock:
            self.cmd = np.array([m.linear.x, m.linear.y, m.angular.z]); self.cmd_t = time.monotonic()

    def on_set_rl(self, req, res):
        if req.data and self.policy is None:
            res.success, res.message = False, "--policy 없이 실행됨"
            return res
        self.rl_on = bool(req.data)
        res.success, res.message = True, "RL 켬" if self.rl_on else "RL 끔 (순수 IK)"
        self.get_logger().info(res.message)
        return res

    def on_reset(self, req, res):
        self.do_reset()
        res.success, res.message = True, "스폰으로 리셋"
        return res

    def do_reset(self):
        self.rover.reset(); self.amap.reset(); self.jit = Jitter()
        self.rover.set_mu(sample_mu(self.rng, self.a.mu, self.mu_state, True))
        self.t_mu = time.monotonic() + self.a.mu_resample_s
        if self.policy is not None:
            self.policy.reset(); self.obs.reset()
        if self.viewer is not None and self.rover.rough:
            self.viewer.update_hfield(0)

    # ---------------------------------------------------------------- 50 Hz
    def tick(self):
        now = time.monotonic()
        with self.lock:
            cmd = self.cmd.copy() if now - self.cmd_t < self.a.cmd_timeout else np.zeros(3)
        if self.a.mu_resample_s > 0 and now >= self.t_mu and not self.a.mu.startswith("const:"):
            self.rover.set_mu(sample_mu(self.rng, self.a.mu, self.mu_state, False))
            self.t_mu = now + self.a.mu_resample_s
        r = self.rover
        s = r.state()
        a = np.zeros(self.amap.dim)
        if self.policy is not None:
            o = self.obs(dict(cmd=cmd, s=s, amap=self.amap, mu_eff=r.mu_effective(),
                              act_delay=self.a.act_delay, obs_delay=self.a.obs_delay))
            pa = self.policy(o)               # always run (keeps the history/RNN state warm)
            # 정지 명령에서는 잔차 0 (10-15): 학습에서 정지는 5% 뿐이고 바퀴가 서 있으면 조향 잔차가 보상에
            # 영향이 없어, 정책이 서 있는 동안 조향을 ±3° 로 흔든다 (MuJoCo 실측).  순수 IK 가 각도를 유지.
            if self.rl_on:
                a = np.asarray(pa, float)          # 정지 잔차 0·변화율 제한은 ActionMap 이 학습과 같게 적용
        tgt = self.amap.process(a, cmd)
        r.step(tgt)
        s = r.state()
        self.jit.add(tgt[:4], s["steer"])
        self.publish(s, a)
        # running stats (IK vs RL comparison in the terminal)
        v = np.array([s["lin_b"][0], s["lin_b"][1], s["ang_b"][2]])
        ac = self.acc
        ac["n"] += 1; ac["exy"] += float(np.linalg.norm(cmd[:2] - v[:2])); ac["eyaw"] += abs(cmd[2] - v[2])
        surf = s["wheel_w"] * s2s.WHEEL_RADIUS_M
        slip = s2s.wheel_slip(s) * np.maximum(np.abs(surf), 0.05)          # back to m/s
        ac["sq"] += float((np.clip(np.abs(slip) - 0.03, 0, None) ** 2).sum()); ac["hold"] += int(r.gate_hold)
        if now - self.t_print > 2.0 and ac["n"]:
            n = ac["n"]
            jc, jm = self.jit.take()
            msg = (f"[{'RL' if self.rl_on else 'IK'}] 2s 평균 xy오차 {ac['exy']/n:.3f} m/s  요오차 {ac['eyaw']/n:.3f} rad/s  "
                   f"큰헛돎 {ac['sq']/n:.4f}  게이트 {100*ac['hold']/n:.0f}%  조향 흔들림 명령 ±{jc:.2f}° 실제 ±{jm:.3f}°  "
                   f"μ {np.round(r.mu_effective(), 2).tolist()}")
            self.get_logger().info(msg)
            self.pub_stat.publish(String(data=json.dumps({"mode": "RL" if self.rl_on else "IK", "xy": ac["exy"]/n,
                                                          "yaw": ac["eyaw"]/n, "sq": ac["sq"]/n, "hold": ac["hold"]/n,
                                                          "jit_cmd_amp_deg": jc, "jit_meas_amp_deg": jm})))
            self.acc = {"n": 0, "exy": 0.0, "eyaw": 0.0, "sq": 0.0, "hold": 0}
            self.t_print = now
        if self.viewer is not None:
            if self.viewer.is_running():
                self.viewer.cam.lookat[:] = s["pos"]; self.viewer.sync()

    def publish(self, s, a):
        d = self.rover.d
        now = self.get_clock().now().to_msg()
        w, qx, qy, qz = d.qpos[3:7]
        t1 = TransformStamped(); t1.header.stamp = now; t1.header.frame_id = "map"; t1.child_frame_id = "odom"
        t1.transform.rotation.w = 1.0
        t2 = TransformStamped(); t2.header.stamp = now; t2.header.frame_id = "odom"; t2.child_frame_id = "base_link"
        t2.transform.translation.x, t2.transform.translation.y, t2.transform.translation.z = map(float, d.qpos[0:3])
        t2.transform.rotation.w, t2.transform.rotation.x = float(w), float(qx)
        t2.transform.rotation.y, t2.transform.rotation.z = float(qy), float(qz)
        self.tf.sendTransform([t1, t2])
        o = Odometry(); o.header.stamp = now; o.header.frame_id = "odom"; o.child_frame_id = "base_link"
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = map(float, d.qpos[0:3])
        o.pose.pose.orientation = t2.transform.rotation
        o.twist.twist.linear.x, o.twist.twist.linear.y = float(s["lin_b"][0]), float(s["lin_b"][1])
        o.twist.twist.angular.z = float(s["ang_b"][2])
        self.pub_odom.publish(o)
        js = JointState(); js.header.stamp = now
        r = self.rover
        names, pos = [], []
        for i, c in enumerate(CORNERS):
            names += [URDF_JOINTS["st"].format(c), URDF_JOINTS["wh"].format(c)]
            pos += [float(d.qpos[r.steer_q[i]]), float(d.qpos[r.wheel_q[i]])]
        names += ["left_rocker_joint", "right_rocker_joint"]
        pos += [float(d.qpos[r.rock[0]]), float(d.qpos[r.rock[1]])]
        js.name, js.position = names, pos
        self.pub_js.publish(js)
        self.pub_act.publish(Float32MultiArray(data=[float(x) for x in a]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terrain", choices=("sand", "rock", "flat"), default="sand")
    ap.add_argument("--soil", choices=("rigid", "sand"), default="rigid", help="sand = MuJoCo 지반역학(슬립-침하)")
    ap.add_argument("--policy", default="", help="Isaac 체크포인트 (params/ 같이). 없으면 IK 만")
    ap.add_argument("--start-ik", action="store_true", help="정책을 불러오되 IK 로 시작 (/set_rl true 로 켬)")
    ap.add_argument("--mu", default="const:0.5", help="const:μ | uniform:lo,hi | patch")
    ap.add_argument("--mu-resample-s", type=float, default=3.0)
    ap.add_argument("--friction-duty", type=float, default=53.0, help="바퀴 관절 마찰 [duty] (지면 하중 실측)")
    ap.add_argument("--obs-delay", type=int, default=1)
    ap.add_argument("--act-delay", type=int, default=1)
    ap.add_argument("--noise", action="store_true", default=True)
    ap.add_argument("--cmd-timeout", type=float, default=0.5)
    ap.add_argument("--no-stand-gate", action="store_true", help="정지 명령에서도 정책 잔차 적용 (떨림 확인용)")
    ap.add_argument("--gate", choices=("old", "new", "off"), default="new",
                    help="구동 정렬 게이트: old = 10-13 이전 펌웨어 (기본) / new = 현 펌웨어·Isaac 학습 / off")
    ap.add_argument("--spawn", type=float, nargs=3, default=None, metavar=("X", "Y", "YAW"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view", action="store_true")
    a, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = Bridge(a)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
