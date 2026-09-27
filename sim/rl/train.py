#!/usr/bin/env python3
"""PPO 학습.

  python3 train.py --steps 2000000 --envs 16 --reward balanced

알고리즘은 SB3 기성품을 쓴다.  성능을 가르는 건 알고리즘이 아니라 환경
(관측/보상/랜덤화)이고, 마감이 있는 상황에서 PPO 를 직접 구현해서 디버깅할
이유가 없다.  "학습이 안 되는 게 내 PPO 버그 때문인가"를 의심하지 않아도 된다.
"""
import argparse, dataclasses, os, pathlib
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from env import RoverEnv
from reward import PRESETS as REWARD_PRESETS


def make_env(rank, args):
    def _f():
        rew = REWARD_PRESETS[args.reward]
        if args.w_energy is not None:
            rew = dataclasses.replace(rew, w_energy=args.w_energy)
        env = RoverEnv(rew=rew, difficulty=args.d0,
                       episode_s=args.episode_s, privileged=args.teacher,
                       seed=args.seed + rank,
                       # 대회맵으로 학습할 때: 거시 구조 고정 + 미세 요철 교란.
                       # 맵 하나에 고정하면 그 한 장의 요철까지 외우는데, 모래는
                       # 대회 당일까지 움직인다 (다른 팀 주행/갈퀴질/습도).
                       arena_eval=(args.terrain != "proc"),
                       eval_kind=(args.terrain if args.terrain != "proc" else "sand"),
                       perturb=(0.0 if args.terrain == "proc" else args.perturb))
        # Monitor 가 있어야 SB3 가 rollout/ep_rew_mean, ep_len_mean 을 찍는다.
        # make_vec_env() 헬퍼는 자동으로 씌워주지만 SubprocVecEnv 를 직접 만들면
        # 안 씌워진다 -- 그래서 터미널에 손실만 나오고 보상이 안 나왔다.
        return Monitor(env)
    return _f


class CurriculumCallback(BaseCallback):
    """에피소드 성공률이 임계를 넘으면 지형 난이도를 올린다.

    30도 경사에서 시작하면 정책이 단 한 번도 성공을 못 해 보상 신호가 0 이고
    영원히 학습이 안 된다.  쉬운 데서 시작해 서서히 올리는 게 필수다."""

    def __init__(self, d0=0.0, step=0.05, thresh=0.60, window=100, verbose=0):
        super().__init__(verbose)
        self.d = d0; self.step = step; self.thresh = thresh
        self.window = window; self.buf = []

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.buf.append(float(info.get("success", 0.0)))
        if len(self.buf) >= self.window:
            sr = float(np.mean(self.buf)); self.buf.clear()
            if sr >= self.thresh and self.d < 1.0:
                self.d = min(1.0, self.d + self.step)
                self.training_env.env_method("set_difficulty", self.d)
                if self.verbose:
                    print(f"[curriculum] 성공률 {sr:.0%} -> 난이도 {self.d:.2f}")
            self.logger.record("curriculum/difficulty", self.d)
            self.logger.record("curriculum/success_rate", sr)
        return True


class MetricsCallback(BaseCallback):
    """에피소드가 끝날 때 info 를 모아 텐서보드에 올린다.

    SB3 기본 스칼라(loss/entropy/...)만으로는 "정책이 왜 못 가는지"를 못 본다.
    슬립·침하·배걸림·고착원인까지 같이 봐야 지형이 어려운 건지 정책이 못 하는
    건지 구분된다.
    """

    # **에피소드 전 구간 집계만 쓴다.**  예전엔 종료 스텝의 순간값을 모았고
    # (slip/sink/e_y/yaw/energy), 그러면 누적 비율인 f_belly 와 의미가 섞인다.
    # 실측 차이: 종료스텝 slip 0.149 vs 에피소드 평균 0.482 (3.2배).
    KEYS = ("success", "stuck", "tip", "oob", "goals", "dist", "frac", "goal_dist",
            "e_y_m", "e_y_p90", "e_y_max", "slip_m", "slip_p90", "sink_m",
            "yaw_m", "prog_m", "pw_m", "e_J", "e_per_m",
            "f_belly", "f_lifted", "duty_sat", "duty_use", "steer_use")
    CAUSES = ("배걸림-하중상실", "배걸림-접촉", "막힘", "슬립고착")
    TAGS = {"배걸림-하중상실": "lost_load", "배걸림-접촉": "belly",
            "막힘": "blocked", "슬립고착": "slip_stuck"}

    def __init__(self, window=200, verbose=0):
        super().__init__(verbose)
        self.window = window
        self.buf = {k: [] for k in self.KEYS}
        self.rterm = {}          # 보상 항별 에피소드 합
        self.cause = []

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if not done:
                continue
            for k in self.KEYS:
                if k in info:
                    self.buf[k].append(float(info[k]))
            for k, v in info.items():
                if k.startswith("rterm_"):
                    self.rterm.setdefault(k[6:], []).append(float(v))
            self.cause.append(info.get("cause", "-"))
        if len(self.cause) >= self.window:
            for k, v in self.buf.items():
                if v:
                    self.logger.record(f"rollout/{k}", float(np.mean(v)))
                    v.clear()
            for k, v in self.rterm.items():
                if v:
                    self.logger.record(f"reward/{k}", float(np.mean(v)))
                    v.clear()
            n = len(self.cause)
            for c in self.CAUSES:
                self.logger.record(f"stuck_cause/{self.TAGS[c]}",
                                   self.cause.count(c) / n)
            self.cause.clear()
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4_000_000)
    p.add_argument("--envs", type=int, default=os.cpu_count())
    p.add_argument("--reward", default="balanced", choices=list(REWARD_PRESETS))
    p.add_argument("--d0", type=float, default=0.0, help="시작 난이도")
    p.add_argument("--episode-s", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/ppo")
    p.add_argument("--clip-steer-deg", type=float, default=10.0)
    p.add_argument("--clip-duty", type=float, default=0.60)   # 0.2 로는 모래 감속 여유 부족
    p.add_argument("--w-energy", type=float, default=None,
                   help="에너지 벌점 가중치 (기본 0.05). RL 정책이 순수 IK 대비 "
                        "에너지를 2.4배 쓰므로(650mAh 팩에서 주행시간 절반) 올려본다. "
                        "0.05 는 전체 보상의 0.5%% 로 사실상 꺼져 있다")
    # gamma 는 **시간 지평**으로 정한다.  gamma = exp(-dt/tau).
    # 0.99 는 50Hz 에서 시상수 2초라 20초 과제의 완주 보너스가 739 스텝 뒤
    # 0.0006 배로 보인다 -- 사실상 없는 항이 된다.  0.998 = 시상수 10초.
    p.add_argument("--gamma", type=float, default=0.998)
    p.add_argument("--terrain", default="proc", choices=["proc", "sand", "rock"],
                   help="학습 지형. proc=절차생성, sand/rock=대회맵 고정(미세 요철 교란)")
    p.add_argument("--perturb", type=float, default=0.020,
                   help="대회맵 학습 시 미세 요철 진폭(표준편차) [m], 난이도에 비례")
    p.add_argument("--teacher", action="store_true",
                   help="특권 관측으로 teacher 학습 (이후 distill.py 로 student 증류)")
    args = p.parse_args()

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    venv = SubprocVecEnv([make_env(i, args) for i in range(args.envs)])
    venv.env_method("set_clip", args.clip_steer_deg, args.clip_duty)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    # teacher 는 특권정보를 z 로 압축하는 인코더를 앞에 단다 (nets.TeacherExtractor).
    # student 는 이 z 를 고유수용감각 이력만으로 맞히도록 나중에 증류한다.
    policy_kwargs = {}
    if args.teacher:
        from nets import TeacherExtractor, LATENT_DIM
        policy_kwargs = dict(features_extractor_class=TeacherExtractor,
                             features_extractor_kwargs=dict(latent_dim=LATENT_DIM))

    # MLP 정책은 CPU 가 더 빠르다 (GPU 전송 오버헤드가 연산보다 큼). SB3 도 경고함.
    try:
        import tensorboard  # noqa: F401
        tb = str(out / "tb")
        print(f"[tb] tensorboard --logdir {out/'tb'}  ->  http://localhost:6006")
    except ImportError:
        tb = None
        print("[warn] tensorboard 없음 (pip install tensorboard) -> stdout 만")

    model = PPO("MlpPolicy", venv, verbose=1, seed=args.seed, device="cpu",
                n_steps=512, batch_size=4096, n_epochs=6,
                learning_rate=3e-4, gamma=args.gamma, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.002,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                                   **policy_kwargs),
                tensorboard_log=tb)
    model.learn(total_timesteps=args.steps, callback=[
        CurriculumCallback(d0=args.d0, verbose=1),
        MetricsCallback(),
        # save_vecnormalize 를 켜야 중간 체크포인트를 평가할 수 있다.  안 켜면
        # 관측 정규화 통계가 학습 끝에만 저장돼서, 중단하면 체크포인트가 있어도
        # 쓸 수 없다 (정책이 스케일 어긋난 입력을 받는다).
        # 20k 마다 남기면 4M 스텝에 200개(수 GB)다.  200k 마다로 충분하다.
        CheckpointCallback(save_freq=max(200_000 // args.envs, 1),
                           save_path=str(out), name_prefix="ppo",
                           save_vecnormalize=True),
    ])
    model.save(out / "final")
    venv.save(str(out / "vecnorm.pkl"))
    print("saved ->", out)


if __name__ == "__main__":
    main()
