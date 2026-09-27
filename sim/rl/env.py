"""KSRC 스워브 로버 Gymnasium 환경.

액션은 **IK 위의 제한된 잔차** 8차원이다 (raw 바퀴 명령이 아님):
    theta_i = IK_theta_i(vx,vy,w) + a[i]   * CLIP_STEER
    duty_i  = IK_speed_i(vx,vy,w) + a[4+i] * CLIP_DUTY
IK 는 펌웨어에 실제로 올라가는 C 코드(firmware/common/swerve_kinematics.c)를
ctypes 로 호출한다.  파이썬으로 재구현하면 언젠가 반드시 어긋나고, 그러면
잔차 정책이 통째로 무의미해진다.

구동 명령을 duty 로 주는 것도 펌웨어와 맞춘 것이다 -- 실기체는 바퀴 속도를
폐루프로 제어하지 않고 PWM duty 를 열어준다 (mdd3a_normalize_speed).
"""
import sys, pathlib
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # sim/
import ksrc_common as kc

from config import RoverCfg, NoiseCfg, sample
from mjcf import build
from terramech import Terramechanics, PRESETS as TERRAIN_PRESETS, sample_terrain
from reward import RewardCfg
import terrain as terr
import path as pth

# MJCF 의 바퀴 순서와 펌웨어 C 의 모듈 순서가 다르다. 여기서 한 번만 맞춘다.
MJ_WHEELS = ("fl", "rl", "fr", "rr")          # mjcf.CORNERS 순서
C_ORDER   = ("fl", "fr", "rl", "rr")          # ksrc_common._make_modules 순서
C2MJ = [MJ_WHEELS.index(w) for w in C_ORDER]  # C 인덱스 -> MJCF 인덱스

OBS_PER_FRAME = 27      # 23 + lookahead 웨이포인트 2점(차체좌표, 4)
# 경로를 관측에 넣지 않으면 횡이탈을 **원리적으로** 관측할 수 없어 경로추종
# 보상이 학습 불가능하다 (기존 23차원에는 자기 위치도 경로도 없었다).
# 실기에서는 Nav2 local plan + TF 로 얻는다 -> 측위 오차를 노이즈로 섞는다.
# 특권 관측 (teacher 전용).  실기체에서는 절대 못 얻는 값들이다.
#   흙 8 + 로버 3 + 차체속도 3 + 슬립 4 + 침하 4 + 수직항력 4 + 배걸림 2 + 지형스캔 15
PRIV_DIM = 43
SCAN_X = (0.00, 0.25, 0.50, 0.75, 1.00)   # 전방 [m] (차체 기준)
SCAN_Y = (-0.35, 0.0, 0.35)               # 좌우 [m]
HISTORY = 50          # 1.0초. 흙 다짐 상태를 이력에서 추정하려면 0.2초로는
                      # 과도응답도 안 끝난다 (Lee 2020 은 2.0초, RMA 는 ~1.0초)
CTRL_HZ = 50


class RoverEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: RoverCfg = None, rew: RewardCfg = None,
                 difficulty: float = 0.0, episode_s: float = 20.0,
                 randomize: bool = True, arena_eval: bool = False,
                 eval_kind: str = "sand", privileged: bool = False,
                 noise: NoiseCfg = None, seed: int = None,
                 perturb: float = 0.0):
        self.base_cfg = cfg or RoverCfg()
        self.rew = rew or RewardCfg()
        self.difficulty = difficulty
        self.randomize = randomize
        self.arena_eval = arena_eval          # True 면 실제 경기장 지형으로 평가
        self.eval_kind = eval_kind            # "sand"(규사 경사지형) / "rock"(착륙지)
        # 대회맵으로 **학습**할 때 쓴다: 거시 구조는 그대로 두고 미세 요철만
        # 매 에피소드 교란한다 (terrain.perturb 주석 참고).  평가 스크립트는
        # 0 으로 둬서 맵 원본 그대로 본다.
        self.perturb = float(perturb)
        # privileged=True 면 관측 뒤에 특권정보를 붙인다 (teacher 학습용).
        # student 는 앞의 OBS_PER_FRAME*HISTORY 만 본다 -- 잘라 쓰면 그대로 호환된다.
        self.privileged = privileged
        # 구동·센싱 노이즈. randomize=False 면 끈다 (검증 스크립트용).
        self.nz = noise or NoiseCfg()
        self.rng = np.random.default_rng(seed)
        self.episode_steps = int(episode_s * CTRL_HZ)

        self.clip_steer = np.radians(10.0)    # 잔차 상한 -- ablation 손잡이
        self.clip_duty = 0.60          # 0.2 로는 모래에서 감속 여유가 부족했다

        n_obs = OBS_PER_FRAME * HISTORY + (PRIV_DIM if privileged else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (8,), np.float32)
        self._build()

    # ------------------------------------------------------------------ 모델
    def _build(self):
        self.cfg = sample(self.base_cfg, self.rng) if self.randomize else self.base_cfg
        c = self.cfg
        xml = build(c, terrain="hfield", hfield_n=terr.HF_N,
                    hfield_size=(terr.ARENA_X / 2, terr.ARENA_Y / 2))
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.n_sub = int(round((1.0 / CTRL_HZ) / self.m.opt.timestep))

        aid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        jid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.a_st = [aid(f"a_st_{w}") for w in MJ_WHEELS]
        self.a_wh = [aid(f"a_wh_{w}") for w in MJ_WHEELS]
        self.j_st = [self.m.jnt_qposadr[jid(f"st_{w}")] for w in MJ_WHEELS]
        self.j_wh = [self.m.jnt_dofadr[jid(f"wh_{w}")] for w in MJ_WHEELS]
        self.sid = {n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, n)
                    for n in ("s_quat", "s_gyro", "s_acc")}
        self.sadr = {k: self.m.sensor_adr[v] for k, v in self.sid.items()}
        self.ground_gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "ground")

        # 펌웨어 C 에 넘길 모듈 좌표 (C 순서)
        self.c_modules = kc._make_modules(half_w=c.track / 2, half_l=c.axle_x)
        self.c_state = (kc.SwerveModuleState * 4)()
        self.v_max = c.mot_w_noload * c.wheel_r

    # ------------------------------------------------------------------ 명령
    def _new_path(self):
        """에피소드용 기준경로를 뽑는다.

        경로 하나에 고정하면 그 경로에 과적합되고, 직선만 주면 나중에 실제
        플래너가 내는 곡선 경로에서 무너진다.  그래서 매 에피소드 새로 뽑고
        곡률을 섞는다 (path.make_path 의 주석 참고).
        """
        self.path = pth.make_path(self.drive, self.hf_Z, self.hf_ex, self.hf_ey,
                                  self.rng, self.d.qpos[:2].copy(),
                                  slope_deg=self.slope_deg, d_ep=self.d_ep)
        # 같은 꼭짓점의 두 번째 사본.  단조 투영 인덱스를 따로 들고 가야 하므로
        # 컨트롤러용(추정치)과 보상용(참값)을 분리한다.
        self.path_c = pth.RefPath(self.path.P)
        self.s_path = 0.0
        self.s_est = 0.0
        self.e_y = 0.0
        self.goal_dist = float(self.path.total)

    def _yaw(self):
        w, x, y, z = self.d.qpos[3:7]
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _project_truth(self):
        """**참값** 위치로 경로 진행도·횡이탈·완주를 갱신한다.

        보상과 종료 판정은 전부 여기서 나온 값을 쓴다.  관측만 센서로 얻을 수
        있으면 되고 보상은 특권정보를 써도 된다.  컨트롤러(추정치)와 섞어 쓰다가
        `prog` 에 측위 잡음의 시간미분이 들어간 적이 있다.
        """
        pos = self.d.qpos[:2].copy()
        # 진행도 증가는 **실제 변위로 상한**된다 (path.project 주석 참고).
        self.s_path, self.e_y, _ = self.path.project(pos)
        # 완주 = 종방향 잔여 + 목표점 실거리 둘 다.  잔여거리만 보면 경로를 크게
        # 벗어난 상태에서도 완주로 잡힌다 (직선 끝에서 1.02m 벗어난 점이 완주였다).
        self.goal_dist = float(np.linalg.norm(pos - self.path.P[-1]))
        if (self.path.total - self.s_path < 0.25) and (self.goal_dist < 0.25):
            self.goals += 1
            self.goal_hit = True

    def _update_cmd(self):
        """Nav2 컨트롤러가 하는 일을 흉내낸다: 기준경로 pure pursuit -> cmd_vel.

        **컨트롤러는 측위 추정치만 본다.**  실기에서는 map->odom TF 로 얻으므로
        측위 정확도가 그대로 실린다.  시뮬에서 참 위치를 쓰면 lookahead 를 극단적으로
        짧게 잡는 게 항상 유리하다고 나오는데(e_y 중앙 38.8 -> 4.0mm) 실기에서는
        잡음 30mm 에 lookahead 0.08m 면 발진한다.
        잡음은 저역통과(tau=0.2s) -- 50Hz 백색이면 조향이 지터를 따라 떨어
        scrub 이 0.012 -> 0.21 로 17배가 된다.
        """
        pos = self.d.qpos[:2].copy()
        yaw = self._yaw()
        if self.randomize:
            n, r = self.nz, self.rng
            self.pos_bias = np.clip(
                self.pos_bias + r.normal(0, n.wp_drift / CTRL_HZ, 2),
                -n.wp_drift_max, n.wp_drift_max)
            a = self.nz_a
            g = np.sqrt((2.0 - a) / a)
            self.pos_lp += a * (r.normal(0, n.wp_noise * g, 2) - self.pos_lp)
            self.yaw_lp += a * (float(r.normal(0, n.wp_yaw_noise * g)) - self.yaw_lp)
            pos_est = pos + self.pos_bias + self.pos_lp
            yaw_est = yaw + self.yaw_lp
        else:
            pos_est, yaw_est = pos, yaw
        self.s_est, _, _ = self.path_c.project(pos_est)
        cmd = pth.pursue(self.path_c, self.s_est, pos_est, yaw_est, self.v_cruise,
                         k_v=self.cfg.pp_k_v, L_min=self.cfg.pp_l_min,
                         L_max=self.cfg.pp_l_max)
        # 실현 가능하도록 스케일 -- IK 가 포화하면 완벽한 정책도 추종을 못 한다
        need = max(np.hypot(cmd[0] - cmd[2] * mm.y, cmd[1] + cmd[2] * mm.x)
                   for mm in self.c_modules)
        if need > self.v_max:
            cmd *= self.v_max / need
        self.cmd = cmd
        # 관측용 lookahead 도 같은 추정치에서 나온다 (같은 map->odom TF).
        self.wp_b = pth.lookahead_body(self.path_c, self.s_est, pos_est, yaw_est)

    def _compute_ik(self):
        """cmd_vel -> 바퀴 4개 조향각·듀티.  **펌웨어와 같은 C 코드**를 쓴다.

        파이썬으로 재구현하면 언젠가 어긋나는데, 정책이 이 해 위의 잔차를
        학습하므로 어긋나는 순간 정책이 통째로 무의미해진다.
        """
        out = kc.swerve_ik_compute(self.cmd[0], self.cmd[1], self.cmd[2],
                                   self.c_modules, self.v_max, self.c_state)
        # 조향 가동범위(±180도) 안으로 접기 -- swerve_fold_to_limit.
        kc.swerve_fold_to_limit(float(self.cfg.steer_lim), self.c_state, out,
                                unwind_rad=0.0, slow_mps=0.0)
        th = np.zeros(4); du = np.zeros(4)
        for ci, mi in enumerate(C2MJ):
            th[mi] = out[ci].angle_rad
            du[mi] = np.clip(out[ci].speed_mps / self.v_max, -1, 1)
        self.th_ik = th
        self.duty_ik = du
        self.th_last = th.copy()          # 진단용

    def set_difficulty(self, d: float):
        """커리큘럼 콜백이 SubprocVecEnv.env_method 로 호출한다."""
        self.difficulty = float(np.clip(d, 0.0, 1.0))
        return self.difficulty

    def set_clip(self, steer_deg: float, duty: float):
        """ablation 손잡이: (10,0.2)=8-D / (0,0.2)=속도만 / (0,0)=순수 IK."""
        self.clip_steer = np.radians(steer_deg); self.clip_duty = duty

    # ------------------------------------------------------------ 구동/센싱 노이즈
    def _sample_noise(self):
        """에피소드마다 고정되는 것(바이어스·전압·지연)과 매 스텝 바뀌는 것을 나눈다."""
        n, r = self.nz, self.rng
        if not self.randomize:                    # 검증 스크립트는 무잡음
            self.volt = 1.0; self.gyro_b = np.zeros(3); self.acc_b = np.zeros(3)
            self.servo_a = 1.0; self.delay = 0
            self.act_buf = []
            self.wp_bias = np.zeros(2); self.pos_bias = np.zeros(2)
            self.pos_lp = np.zeros(2); self.yaw_lp = 0.0; self.nz_a = 1.0
            return
        self.volt = float(r.uniform(*n.volt_scale))
        self.gyro_b = r.uniform(*n.gyro_bias, size=3)
        self.acc_b = r.uniform(*n.acc_bias, size=3)
        tau = float(r.uniform(*n.servo_lag))
        self.servo_a = float(np.clip((1.0 / CTRL_HZ) / max(tau, 1e-3), 0.0, 1.0))
        self.delay = int(r.integers(n.act_delay[0], n.act_delay[1] + 1))
        self.act_buf = []
        self.wp_bias = np.zeros(2)          # (미사용, 호환용)
        self.pos_bias = np.zeros(2)         # 측위 드리프트 (랜덤워크)
        self.pos_lp = np.zeros(2); self.yaw_lp = 0.0      # 저역통과 잡음 상태
        self.nz_a = float(np.clip((1.0 / CTRL_HZ) / max(n.wp_tau, 1e-3), 0.0, 1.0))

    def _motor_scale(self, duty):
        """배터리 전압 강하.  DC 모터는 정지토크와 무부하속도가 **둘 다** 전압에
        비례한다.  부하가 클수록 더 떨어진다 (3S 11.1V 가 78A 피크에서 셀당
        3.2V 까지 간다)."""
        if not self.randomize:
            return 1.0
        load = float(np.mean(np.abs(duty)))
        return float(np.clip(self.volt * (1.0 - self.nz.volt_sag_k * load), 0.3, 1.0))

    # ------------------------------------------------------------------ 관측
    def _frame(self):
        c = self.cfg
        q = self.d.sensordata[self.sadr["s_quat"]:self.sadr["s_quat"] + 4]
        w, x, y, z = q
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        gyro = self.d.sensordata[self.sadr["s_gyro"]:self.sadr["s_gyro"] + 3]
        acc = self.d.sensordata[self.sadr["s_acc"]:self.sadr["s_acc"] + 3]
        wheel_meas = np.array([self.d.qvel[j] for j in self.j_wh]) * c.wheel_r
        steer = np.array([self.d.qpos[j] for j in self.j_st])
        # --- 센싱 노이즈 -----------------------------------------------------
        # 관측이 완벽하게 깨끗하면 이력만으로 다 풀려서 정책이 특권정보를 안 쓴다.
        # 실기체에서 못 없애는 것들만 넣는다.
        if self.randomize:
            n, r = self.nz, self.rng
            # 엔코더: 50Hz 에서는 양자화가 지배적이다 (8384 tick/rev).
            tick = 2 * np.pi / c.ticks_per_rev * CTRL_HZ * c.wheel_r
            if n.enc_quant > 0:
                wheel_meas = np.round(wheel_meas / (tick * n.enc_quant)) * tick * n.enc_quant
            wheel_meas = wheel_meas * (1.0 + r.normal(0, n.enc_noise, 4))
            roll += float(r.normal(0, n.rp_noise)); pitch += float(r.normal(0, n.rp_noise))
            gyro = gyro + self.gyro_b + r.normal(0, n.gyro_noise, 3)
            acc = acc + self.acc_b + r.normal(0, n.acc_noise, 3)
        # 물리 스케일로 정규화 -- 파라미터가 바뀌어도 분포가 안 흔들리게
        return np.concatenate([
            self.cmd / np.array([self.v_max, self.v_max, 3.0]),
            self.duty_ik,                      # 이미 [-1,1]
            wheel_meas / self.v_max,
            steer / np.pi,
            [roll / (np.pi / 2), pitch / (np.pi / 2)],
            gyro / 5.0, acc / 9.81,
            np.clip(self.wp_b.ravel() / 1.0, -2.0, 2.0),   # lookahead 2점 (m)
        ]).astype(np.float32)

    def _obs(self):
        o = np.concatenate(self.hist).astype(np.float32)
        if self.privileged:
            o = np.concatenate([o, self._privileged()]).astype(np.float32)
        return o

    # -------------------------------------------------------------- 특권 관측
    def _terrain_scan(self):
        """차체 기준 격자점의 지형 높이 (현재 차체 높이 대비).  D435 depth 로
        언젠가 student 에 줄 수도 있지만, 지금은 teacher 전용으로 둔다."""
        Z, ex, ey = self.hf_Z, self.hf_ex, self.hf_ey
        ny, nx = Z.shape
        yaw = self._yaw(); c, s_ = np.cos(yaw), np.sin(yaw)
        px, py, pz = self.d.qpos[0], self.d.qpos[1], self.d.qpos[2]
        gx = np.array([[c * a - s_ * b + px for b in SCAN_Y] for a in SCAN_X]).ravel()
        gy = np.array([[s_ * a + c * b + py for b in SCAN_Y] for a in SCAN_X]).ravel()
        j = np.clip(((gx + ex / 2) / ex * nx).astype(int), 0, nx - 1)
        i = np.clip(((gy + ey / 2) / ey * ny).astype(int), 0, ny - 1)
        h = Z[i, j] + self.hf_off - pz          # 차체보다 얼마나 높은가 [m]
        return np.clip(h / 0.30, -1.0, 1.0)     # ±300mm 로 정규화

    def _privileged(self):
        """실기체에서 못 얻는 값 전부.  teacher 만 본다."""
        p, c = self.terr_p, self.cfg
        soil = np.array([p.mu_max / 0.9, p.K / 0.3, p.alpha / 0.12, p.beta / 0.9,
                         p.c_r / 0.1, p.k_z / 13.0, p.z_max / 0.06, p.mu_lat])
        rover = np.array([c.m_base / 1.0, c.wheel_r / 0.045, c.pivot_z / 0.06])
        v_w = self.d.qvel[:3].copy()
        q = self.d.qpos[3:7]
        Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, q); Rm = Rm.reshape(3, 3)
        v_b = (Rm.T @ v_w) / self.v_max                    # 차체좌표 실속도
        slip = self.tm.slip.copy()                          # 바퀴별 슬립비
        sink = self.tm.z / max(self.tm.p.z_max, 1e-6)       # 바퀴별 침하
        nominal = self.m.body_mass.sum() * 9.81 / 4.0
        fn = np.clip(self.tm.Fn / max(nominal, 1e-6), 0.0, 3.0) / 3.0
        belly = np.array([float(self.n_belly > 0 and self.k > 0),
                          float(np.sum(self.tm.Fn < 0.25 * nominal)) / 4.0])
        return np.concatenate([soil, rover, v_b, slip, sink, fn, belly,
                               self._terrain_scan()]).astype(np.float32)

    # ------------------------------------------------------------------ 리셋
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.randomize:
            self._build()
        # 난이도를 한 점에 고정하지 않고 U(0.6*d_max, d_max) 로 뽑는다.
        # 고정하면 정책이 그 난이도에만 맞춰지고, 커리큘럼이 정체하면 어려운
        # 지형을 영영 못 본다.  범위로 주면 항상 일부는 어렵다.
        # arena_eval 이어도 흙 난이도는 **학습과 같은 분포**에서 뽑는다.
        # 예전엔 d_ep=difficulty 로 고정했는데, 그러면 흙이 제일 무른 쪽으로
        # 고정되어 지형 기하의 차이와 흙의 차이가 뒤섞인다 (실제로 실제 경기장
        # 평가에서 세 변형이 전부 무너져 비교가 불가능했다).
        # arena_eval 에서 d_ep 는 흙에만 영향을 준다 -- 지형은 항상 실제 STL.
        self.d_ep = float(self.rng.uniform(0.6 * self.difficulty, self.difficulty))
        # 흙도 난이도에 연동한다. firm -> sand 로 보간되므로 고난이도 에피소드는
        # 실제로 모래가 된다 (예전 거부 샘플링은 모래만 골라서 버렸다).
        self.terr_p = (sample_terrain(self.rng, self.d_ep)
                       if self.randomize else TERRAIN_PRESETS["sand"])
        self._sample_noise()
        Z = (terr.load_arena(self.eval_kind) if self.arena_eval
             else terr.generate(self.d_ep, self.rng)[0])
        if self.arena_eval and self.perturb > 0:
            Z = terr.perturb(Z, *terr.eval_extent(self.eval_kind), self.rng,
                             amp=self.perturb * self.d_ep)
        # to_hfield 는 -Z.min() 을 돌려준다. MuJoCo 표면높이 = Z - Z.min() 이므로
        # 스폰 높이에 이 오프셋을 반드시 더해야 한다 (안 더하면 지형에 파묻힌다).
        # hfield 의 물리 범위는 학습(8x8)과 평가(실제 4x5)가 다르다. 런타임에 맞춘다.
        ex, ey = terr.eval_extent(self.eval_kind)
        hx, hy = ((ex / 2, ey / 2) if self.arena_eval
                  else (terr.ARENA_X / 2, terr.ARENA_Y / 2))
        self.m.hfield_size[0, 0] = hx; self.m.hfield_size[0, 1] = hy
        z_off = terr.to_hfield(self.m, Z)

        # 스폰/목표는 **주행 가능 영역**에서만 뽑는다.  맵 중앙 고정 스폰은 실제
        # 규사 STL 에서 절벽 단차(한 셀에 231mm) 위에 로버를 놓아 리셋 직후
        # 전복시켰다 (전복률 90%).  마스크는 국소 경사 25도 이하 + 풋프린트 안
        # 높이차 50mm 이하로 정의한다.
        self.hf_Z = Z; self.hf_ex = 2 * hx; self.hf_ey = 2 * hy; self.hf_off = z_off
        # 경사 상한은 로버 견인력 예산에서 유도된다 (config.max_slope_deg 주석).
        self.drive, self.slope_deg = terr.drivable_mask(
            Z, self.hf_ex, self.hf_ey, max_slope_deg=self.cfg.max_slope_deg)
        spawn = terr.sample_drivable(self.drive, Z, self.hf_ex, self.hf_ey,
                                     self.rng, 1)[0]
        z_top = terr.local_top(Z, self.hf_ex, self.hf_ey, spawn) + z_off

        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[0] = spawn[0]; self.d.qpos[1] = spawn[1]
        self.d.qpos[2] += z_top + 0.01
        # 자세는 뒤에서 경로 시작방향에 맞춰 넣는다 (_new_path 이후).
        self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.tm = Terramechanics(self.m, self.cfg.wheel_r, self.terr_p)
        self.c_state = (kc.SwerveModuleState * 4)()

        # 명령은 **목표점 추종**으로 만든다.  고정 명령을 쓰면 20초 에피소드에서
        # 6.4m 를 직진해 3x5m 지형 밖으로 나가버린다 (평지에서도 성공률이 15%
        # 밖에 안 나온 원인).  목표점 방식은 그걸 막으면서 동시에 nav2 가 실제로
        # 주는 "계속 변하는 속도 명령"을 재현한다.
        self.goals = 0
        self.v_cruise = self.rng.uniform(0.45, 1.0) * self.v_max
        self._new_path()
        # 스폰 자세를 경로 시작방향에 맞춘다 (+-60도 오차).  완전 랜덤이면
        # 에피소드 초반을 제자리 회전에 쓰고, omega 명령 통계를 그게 지배한다.
        # 실기에서도 nav2 는 rotation_shim_controller 로 큰 초기 요 오차를
        # 먼저 없앤 뒤 경로추종에 들어간다.
        yaw0 = pth.start_heading(self.path) + float(
            self.rng.uniform(-np.pi / 3, np.pi / 3))
        self.d.qpos[3:7] = [np.cos(yaw0 / 2), 0.0, 0.0, np.sin(yaw0 / 2)]
        self.wp_b = np.zeros((2, 2))
        self.cmd = np.zeros(3)
        self.duty_ik = np.zeros(4)
        self.a_prev = np.zeros(8)
        self.a_prev2 = np.zeros(8)
        self.st_prev = np.array([self.d.qpos[j] for j in self.j_st])
        self.st_cmd = self.st_prev.copy()        # 서보 1차 지연 상태
        self.th_last = self.st_prev.copy()       # 등가각 선택의 기준 (직전 명령각)
        self.k = 0
        self.prog_hist = []
        self.goal_hit = False
        self.path_len = 0.0
        self.rsum = {}                           # 항별 누적 보상 (에피소드 단위)
        self.n_belly = 0; self.n_lifted = 0; self.n_blocked = 0
        self.e_sum = 0.0                     # 누적 전기에너지 [J]
        # **에피소드 전 구간 집계.**  예전엔 eval/train 이 종료 스텝의 순간값을
        # 모았다 (slip/sink/e_y/yaw/energy).  f_belly 같은 누적 비율과 의미가
        # 달라 한 표에 섞여 있었다.  평균·p90·최대를 전부 남긴다.
        self.acc = {k: [] for k in ("slip", "sink", "e_y", "yaw", "pw", "prog")}
        self.n_duty_sat = 0.0; self.duty_lost = 0.0; self.duty_use = 0.0
        self.n_steer_sat = 0.0; self.steer_use = 0.0
        self.stuck_cause = "-"
        self.last_xy = self.d.qpos[:2].copy()
        self.x0 = self.d.qpos[:2].copy()

        for _ in range(int(0.3 / self.m.opt.timestep)):     # 정착
            self.tm.apply(self.d, self.m.opt.timestep); mujoco.mj_step(self.m, self.d)
        self.tm.reset()
        # **바퀴별 흙 차이.**  난이도에 비례해 비대칭을 키운다 (d=0 이면 균일).
        # 상관길이는 트랙폭(236.7mm)보다 조금 작게 둬서 좌우 바퀴가 서로 다른 값을
        # 밟고, 패치를 넘어갈 때 과도 외란이 생긴다.  **tm.reset() 이 mu_field 를
        # 지우므로 반드시 그 뒤에 넣어야 한다** (처음에 앞에 뒀다가 안 먹었다).
        if self.randomize and self.cfg.soil_amp > 0:
            self.tm.mu_field = terr.soil_field(
                self.hf_ex, self.hf_ey, self.rng,
                amp=self.cfg.soil_amp * self.d_ep, lam=self.cfg.soil_lam)
            self.tm.f_ex, self.tm.f_ey = self.hf_ex, self.hf_ey
        # 정착으로 위치가 조금 움직이지만 project 가 단조로 흡수한다.
        # 여기서 _new_path 를 다시 부르면 위에서 자세를 맞춘 경로와 **다른**
        # 경로가 되어 초기 요 오차 정렬이 무의미해진다.
        self._project_truth()            # 참값 경로 상태
        self.goal_hit = False            # 정착 중 완주로 잡히지 않게
        self._update_cmd()               # 첫 명령
        self._compute_ik()               # 첫 IK (정책이 관측할 값)
        self.hist = [self._frame() for _ in range(HISTORY)]
        return self._obs(), {}

    # ------------------------------------------------------------------ 스텝
    def step(self, a):
        """제어 한 주기.  순서가 중요하다.

        예전 순서는 `진행도 저장 -> 진행도 갱신 -> 행동 -> 물리 -> 보상` 이어서
        행동 a_t 에 대해 **a_(t-1) 의 이동**을 보상했다 (실측: 스텝0 의 실제 진행
        0.204 가 스텝1 에 반환됐다).  게다가 e_y 는 물리 전, yaw 는 물리 후 값을
        써서 한 보상 안에 두 시점이 섞였다.

        바른 순서:
          1) 정책이 관측한 그 명령에 잔차를 적용한다 (th_ik/duty_ik 는 직전 스텝
             끝에서 계산된 것이고, 관측에 들어간 것과 **같은 값**이다)
          2) 물리 진행
          3) 물리 후 **참값**으로 경로 상태·보상·종료 계산
          4) 다음 명령과 관측 생성
        """
        a = np.clip(np.asarray(a, np.float64), -1, 1)
        c = self.cfg
        th_ik, du_ik = self.th_ik, self.duty_ik

        # 1) 잔차 적용
        # 포화 계측: 잔차는 "요청" 이고, 클립을 거친 뒤가 "실제 인가" 다.
        th_req = th_ik + a[:4] * self.clip_steer
        du_req = du_ik + a[4:] * self.clip_duty
        th = np.clip(th_req, -c.steer_lim, c.steer_lim)
        du = np.clip(du_req, -1, 1)
        if self.clip_duty > 1e-9:
            self.n_duty_sat += float(np.mean(np.abs(du_req) > 1.0))
            self.duty_lost += float(np.mean(np.abs(du_req - du)) / self.clip_duty)
            self.duty_use += float(np.mean(np.abs(du - du_ik)) / self.clip_duty)
        if self.clip_steer > 1e-9:
            self.n_steer_sat += float(np.mean(np.abs(th_req) > c.steer_lim))
            self.steer_use += float(np.mean(np.abs(th - th_ik)) / self.clip_steer)
        # 제어 지연: Pi -> STM32 -> 구동부 왕복. 50Hz 스텝 단위로 큐잉.
        self.act_buf.append((th.copy(), du.copy()))
        th_a, du_a = self.act_buf[0] if len(self.act_buf) > self.delay else (th, du)
        if len(self.act_buf) > self.delay:
            self.act_buf.pop(0)
        # 조향 서보 1차 지연 + 데드밴드 (STS3215)
        if self.randomize:
            err = th_a - self.st_cmd
            err = np.where(np.abs(err) < self.nz.servo_dead, 0.0, err)
            self.st_cmd = self.st_cmd + self.servo_a * err
            th_a = self.st_cmd
        # 배터리 전압: 구동 전압에만 곱한다.  **감쇠(역기전력)에는 곱하지 않는다.**
        # 실제 DC 모터는 tau = (V*kt/R) - (kt*ke/R)*w 로 역기전력 항이 V 와 무관하다.
        # 예전엔 dof_damping 도 vs 배 해서 무부하 평형 w = duty*tau_stall/b 에서
        # vs 가 상쇄됐다 -- 전압 랜덤화가 무부하속도에 **전혀 효과가 없었다**
        # (vs 1.05/1.00/0.83 전부 3.980 rad/s).
        vs = self._motor_scale(du_a)
        for i in range(4):
            self.m.dof_damping[self.j_wh[i]] = c.wheel_damping
        for i in range(4):
            self.d.ctrl[self.a_st[i]] = th_a[i]
            self.d.ctrl[self.a_wh[i]] = du_a[i] * vs

        # 2) 물리
        for _ in range(self.n_sub):
            self.tm.apply(self.d, self.m.opt.timestep)
            mujoco.mj_step(self.m, self.d)
        self.k += 1

        # 3) 물리 후 참값으로 경로 상태 -> 보상 -> 종료
        self.s_prev = self.s_path
        self.goal_hit = False
        self._project_truth()
        r, term, info = self._reward(a, du_a, vs)

        # 4) 다음 명령과 관측
        self._update_cmd()
        self._compute_ik()
        self.hist.pop(0); self.hist.append(self._frame())
        self.a_prev2 = self.a_prev
        self.a_prev = a
        trunc = self.k >= self.episode_steps
        return self._obs(), r, term, trunc, info

    # ------------------------------------------------------------- 고착 원인 분해
    # 셋 다 "바퀴는 도는데 안 나간다" 로 보이지만 원인도 대응도 다르다.
    #   배 걸림  차체가 지형에 얹혀 바퀴 하중 상실 -> 토크 조절로는 해결 안 됨
    #   막힘     바위에 물려 바퀴가 안 돔          -> 토크를 더 줘야 함
    #   모래고착  슬립-침하 되먹임                  -> 토크를 빼야 함
    # 실기체에선 수직항력/전류/침하를 못 재서 구분이 불가능하지만, 시뮬에서는
    # 전부 관측된다. **정책 입력에는 넣지 않고** 평가 리포트에만 쓴다.
    def _diagnose(self, w_ang, du):
        """(배 접촉, 뜬 바퀴 수, 막힘) 을 돌려준다.

        배 접촉과 '진짜 배 걸림' 은 다르다. 차체가 지형을 스치기만 하면 바퀴
        하중은 그대로라 주행에 지장이 없다. 문제는 차체가 **하중을 받아 바퀴가
        뜨는** 경우다. 그래서 접촉 여부와 별개로 수직항력이 떨어진 바퀴 수를 센다.
        """
        c = self.cfg
        belly = False
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            if self.ground_gid not in (con.geom1, con.geom2):
                continue
            other = con.geom2 if con.geom1 == self.ground_gid else con.geom1
            if other not in self.tm.wg:          # 타이어가 아닌 것이 지면에 닿음
                belly = True; break
        nominal = self.m.body_mass.sum() * 9.81 / 4.0
        lifted = int(np.sum(self.tm.Fn < 0.25 * nominal))
        v_meas = np.abs(w_ang) * c.wheel_r
        v_cmd = np.abs(du) * self.v_max
        act = v_cmd > 0.15 * self.v_max          # 의미있는 명령을 받은 바퀴만
        blocked = bool(act.any() and
                       np.mean(v_meas[act] < 0.15 * v_cmd[act]) > 0.5)
        return belly, lifted, blocked

    # ------------------------------------------------------------------ 보상
    def _reward(self, a, du, vs=1.0):
        R, c = self.rew, self.cfg
        # --- 경로추종 3항 ---------------------------------------------------
        # 속도추종을 **경로좌표로 분해**한 것이다 (reward.py 주석 참고):
        #   접선 성분 -> prog,  법선 성분 -> xt,  요 -> yaw.
        dt_ctrl = 1.0 / CTRL_HZ
        den = max(self.v_cruise * dt_ctrl, 1e-6)
        # 경로 호길이 진행량.  투영이 단조라 뒤로 점프하지 않는다.
        prog = float(np.clip((self.s_path - self.s_prev) / den, -1.0, 1.0))
        # 횡이탈.  xt_sat 에서 포화 -- 실측 90분위(110mm)에 맞췄다.
        xt = float(np.clip(abs(self.e_y) / R.xt_sat, 0.0, 1.0))
        goal_bonus = R.r_goal if self.goal_hit else 0.0

        # 요속도 추종오차를 **비용**으로.  예전엔 가우시안 양의 보상이라
        # 오차가 작으면 매 스텝 점수가 쌓였고, 그게 완주를 늦출 유인이 됐다
        # (yaw +180 누적 vs 완주 보너스 일회성 +3).
        om_err = abs(float(self.d.qvel[5]) - self.cmd[2])
        yaw_c = float(np.clip(om_err / R.yaw_sigma, 0.0, 1.0))
        yawr = 1.0 - yaw_c                      # 진단용 (기존 지표와 호환)

        # --- 모든 벌점을 [0,1] 로 정규화한다 ------------------------------
        # 안 하면 가중치 숫자와 실제 영향이 따로 논다.  예전 값으로는 sink 가
        # 최대 0.02(단위가 m 라서), smooth 가 최대 1.60(track 보다 큼) 이었다.
        w_ang = np.array([self.d.qvel[j] for j in self.j_wh])     # 바퀴 각속도
        slip = float(np.abs(self.tm.slip).mean())                  # 이미 [0,1]
        sink = float(self.tm.z.mean() / self.tm.p.z_max)           # -> [0,1]

        # 실제 인가 토크.  du 는 **지연·전압이 반영된 인가 듀티** 다
        # (예전엔 요청 듀티와 공칭 계수를 써서 지연·전압이 빠져 있었다).
        tau = du * vs * c.mot_tau_stall - c.wheel_damping * w_ang
        # 배터리 소비전력.  예전엔 |tau*omega| (기계 출력) 를 썼는데, 그러면
        # **정지 상태로 힘을 쓰면 0** 이 나온다 -- 전류는 최대인데.
        # 브러시 DC:  P_elec = P_mech + I^2 R,   I = tau/kt
        #   tau_stall = kt*V/R,  w_noload = V/kt  ->  R/kt^2 = w_noload/tau_stall
        #   따라서  I^2 R = tau^2 * w_noload / tau_stall   (알려진 파라미터만 씀)
        # 회생은 배터리로 돌아오지 않으므로 기계 출력은 양수만 센다.
        p_mech = np.maximum(tau * w_ang, 0.0)
        p_cu = tau ** 2 * c.mot_w_noload / c.mot_tau_stall
        self.p_elec = float(np.sum(p_mech + p_cu))                  # W (4개 합)
        P_REF = 4.0 * c.mot_tau_stall * c.mot_w_noload / 4.0        # 정규화 기준 [W]
        energy = float(np.clip(self.p_elec / P_REF, 0.0, 1.0))      # -> [0,1]
        TAU_RATED = 0.49                                           # 기어박스 정격
        over = float(np.mean(np.clip(np.abs(tau) - TAU_RATED, 0, None)) /
                     (c.mot_tau_stall - TAU_RATED))                # -> [0,1]

        # scrub 은 **실제 조향 각속도** 로 재야 한다. 잔차 a[:4] 로 재면 IK 가
        # 크게 조향해도 벌점이 0 이 된다.
        st = np.array([self.d.qpos[j] for j in self.j_st])
        dst = np.abs(st - self.st_prev) * CTRL_HZ                  # rad/s
        self.st_prev = st
        stopped = np.clip(1 - np.abs(w_ang * c.wheel_r) / self.v_max, 0, 1)
        scrub = float(np.mean(np.clip(dst / c.steer_vmax, 0, 1) * stopped))

        # 2차 차분.  1차 차분으로 재면 일정한 램프 조향에도 벌점이 붙는다.
        # 2차 차분은 고주파 진동만 잡고 부드러운 변화는 허용한다.
        d2 = a - 2.0 * self.a_prev + self.a_prev2
        smooth = float(np.sum(d2 ** 2) / 128.0)                    # -> [0,1]

        self.e_sum += self.p_elec / CTRL_HZ        # 줄(J) 누적 -- 물리 단위
        belly, lifted, blocked = self._diagnose(w_ang, du)
        self.n_belly += belly
        self.n_lifted += (belly and lifted >= 1)
        self.n_blocked += blocked
        resid = float(np.sum(a ** 2) / 8.0)                        # -> [0,1]

        # 항별 기여를 그대로 남긴다.  원시값(slip=0.6)만 보면 그게 보상에서
        # 얼마나 힘을 쓰는지 알 수 없다 -- 가중치를 곱한 값이 비교 가능한 단위다.
        # 목적함수 = "목표까지의 비용 최소화".  매 스텝 양의 보상은 prog 하나뿐이고
        # 그것도 실제 변위로 상한돼 있다 (정지 상태에서 0).
        terms = {
            "prog":   +R.w_prog * prog,
            "time":   -R.w_time,
            "xt":     -R.w_xt * xt,
            "yaw":    -R.w_yaw * yaw_c,
            "goal":   +goal_bonus,
            "slip":   -R.w_slip * slip,
            "sink":   -R.w_sink * sink,
            "scrub":  -R.w_scrub * scrub,
            "energy": -R.w_energy * energy,
            "torque": -R.w_torque * over,
            "smooth": -R.w_smooth * smooth,
            "resid":  -R.w_resid * resid,
        }
        rew = float(sum(terms.values()))

        # 종료 조건
        roll = self.hist[-1][15] * (np.pi / 2); pitch = self.hist[-1][16] * (np.pi / 2)
        tip = abs(roll) > np.radians(60) or abs(pitch) > np.radians(60)
        # 누적 경로길이. 출발점으로부터의 거리로 재면 목표를 찍고 되돌아올 때
        # 값이 줄어들어 "정상 주행 중인데 고착" 으로 오판한다.
        self.path_len += float(np.linalg.norm(self.d.qpos[:2] - self.last_xy))
        self.last_xy = self.d.qpos[:2].copy()
        self.prog_hist.append(self.path_len)
        stuck = False
        if len(self.prog_hist) > 2 * CTRL_HZ:                  # 최근 2초
            stuck = (self.prog_hist[-1] - self.prog_hist[-2 * CTRL_HZ]) < 0.02 and slip > 0.5
        # 벽이 물리적으로 막으므로 이건 안전망일 뿐 (벽을 넘었거나 떨어진 경우)
        ex, ey = terr.eval_extent(self.eval_kind)
        hx, hy = ((ex / 2, ey / 2) if self.arena_eval
                  else (terr.ARENA_X / 2, terr.ARENA_Y / 2))
        oob = bool(abs(self.d.qpos[0]) > hx - 0.1 or abs(self.d.qpos[1]) > hy - 0.1
                   or self.d.qpos[2] < -0.5)
        if stuck:
            # 귀속 순서가 곧 우선순위다. 배가 하중을 받아 바퀴가 뜬 상태가
            # 가장 치명적이고(토크 조절로 해결 불가), 그다음이 바퀴 스톨,
            # 나머지는 전부 슬립 고착이다 -- 고착 판정 자체가 slip>0.5 를
            # 요구하므로 "셋 중 아무것도 아님" 은 정의상 슬립 고착이다.
            # (예전엔 침하 임계로 갈랐는데, 침하는 시정수 2초로 쌓이고 고착
            #  판정도 2초라 대부분이 "기타" 로 빠졌다 -- 전체의 30%.)
            self.stuck_cause = ("배걸림-하중상실" if (belly and lifted >= 1) else
                                "배걸림-접촉"     if belly else
                                "막힘"            if blocked else
                                "슬립고착")
        if tip:   rew += R.r_tip;   terms["tip"] = R.r_tip
        if stuck: rew += R.r_stuck; terms["stuck"] = R.r_stuck
        for kk, vv in terms.items():
            self.rsum[kk] = self.rsum.get(kk, 0.0) + vv
        A = self.acc
        A["slip"].append(slip); A["sink"].append(sink); A["e_y"].append(abs(self.e_y))
        A["yaw"].append(yawr); A["pw"].append(self.p_elec); A["prog"].append(prog)
        # 경로를 완주하면 거기서 끝난다 -- 다음 목표를 이어붙이면 "경로를
        # 따라갔나" 라는 판정이 흐려진다.
        term = bool(tip or stuck or oob or self.goal_hit)
        # 성공 = **지정된 경로를 끝까지 따라갔나**.  예전 기준(간 거리 / 명령속도로
        # 갈 수 있었던 거리 >= 0.35)은 두 군데가 틀렸다:
        #   - 0.35 는 너무 관대했다 (3분의 1만 가도 성공)
        #   - path_len 이 누적 경로길이라 **헤매는 것에 보상**이 됐다
        # 경로 호길이 진행률은 둘 다 없다.  헤매면 s_path 가 안 늘어난다.
        frac = float(self.s_path / max(self.path.total, 1e-6))
        # 완주 판정은 goal_hit 을 그대로 쓴다.  frac >= 0.9 로 재면 goal_hit 이
        # "남은 거리 0.25m" 에서 뜨므로 frac = 1 - 0.25/total 이 되어 **경로 길이가
        # 성공을 결정**한다 (2.15m 경로 -> 0.885 실패, 3.13m -> 0.920 성공).
        completed = bool(self.goal_hit and not (tip or stuck or oob))
        k = max(self.k, 1)
        info = dict(prog=prog, xt=xt, yaw=yawr, e_y=float(abs(self.e_y)),
                    slip=slip, sink=sink, energy=energy,
                    over=over, scrub=scrub,
                    f_belly=self.n_belly / k, f_lifted=self.n_lifted / k,
                    f_blocked=self.n_blocked / k, cause=self.stuck_cause,
                    duty_sat=self.n_duty_sat / k, duty_lost=self.duty_lost / k,
                    duty_use=self.duty_use / k,
                    steer_sat=self.n_steer_sat / k, steer_use=self.steer_use / k,
                    tip=tip, stuck=stuck, oob=oob,
                    # 에피소드 전 구간 집계 (평균 / 90분위 / 최대)
                    **{f"{k}_{st}": float(f(v)) for k, v in self.acc.items()
                       if v for st, f in (("m", np.mean),
                                          ("p90", lambda x: np.percentile(x, 90)),
                                          ("max", np.max))},
                    goals=self.goals, dist=self.path_len, frac=float(frac),
                    # 배터리에 중요한 것은 순간 전력이 아니라 **이동거리당 소비**다.
                    # 느리게 가면 순간 전력은 낮아도 미터당은 더 나쁠 수 있다.
                    # 물리 단위.  e_J 는 누적 전기에너지, e_per_m 은 **완주한 경로
                    # 호길이당** 소비다 (헤맨 거리를 분모에 넣으면 헤매는 게 유리해진다).
                    # e_J 는 **구동 모터의 기계일 + 구리손** 추정치다.  무부하 전류,
                    # 기어 손실, 드라이버 손실, 조향 서보, 전장 소비가 빠져 있어
                    # 배터리 지속시간 예측에는 못 쓴다 (모델 내부 비교용).
                    e_J=float(self.e_sum),
                    e_per_m=float(self.e_sum / self.s_path) if self.s_path > 0.3 else float("nan"),
                    goal_dist=float(self.goal_dist),
                    **{f"rterm_{kk}": vv for kk, vv in self.rsum.items()},
                    completed=completed, success=completed,
                    route_m=float(self.path.total))
        return float(rew), term, info
