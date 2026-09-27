# CLI

모든 명령은 `sim/rl/` 에서 실행한다.

```bash
cd ~/ksrc/sim/rl
```

---

## 0. 빌드 / 검증

펌웨어 C 코드를 공유 라이브러리로 빌드한다. **기구학을 수정했으면 반드시 먼저 실행한다** —
시뮬과 STM32 가 같은 `firmware/common/*.c` 를 쓰기 때문에, 안 하면 정책이 학습한 기구학과
실기체 기구학이 어긋난다.

```bash
bash ../build.sh
```

모델 물리 검증 9종. 인자 없음.

```bash
python3 check_model.py
```

| | 검증 |
|---|---|
| [1] | 평지 정착 — 차체 높이, 로커각, 접촉력 합 = 무게 |
| [2] | 디퍼렌셜 — `rock_L + rock_R = 0` |
| [3] | 풀스로틀 — 차체속도가 이론 무부하와 일치 |
| [4] | 비틀림 접지 — 40mm 턱에서 네 바퀴 유지 |
| [5] | 횡경사 유지 — 슬립모델 ON, 횡방향 접지력 |
| [6] | 방향별 추종 — 전진/후진/게걸음/대각/제자리회전 |
| [7] | 요 외란 저항 |
| [8] | 조향 스텝 응답 |
| [9] | 누적 조향 드리프트 — 180° 등가 접기 검증 |

견인력 곡선과 고착 재현 검증. 인자 없음.

```bash
python3 test_terramech.py
```

---

## 1. 학습

### teacher (특권 관측)

특권정보 43차원(흙 8 / 로버 3 / 차체속도 3 / 슬립 4 / 침하 4 / 수직항력 4 / 배걸림 2 /
지형스캔 15)을 인코더로 `z(8)` 에 압축한다. 실기체에서는 못 쓰고, 증류의 교사로만 쓴다.

```bash
python3 train.py --teacher --steps 4000000 --envs 16 --out runs/teacher
```

### blind (대조군)

같은 관측 길이인데 특권정보가 없다. teacher-student 구조가 **실제로 값어치가 있는지**
가르는 기준선이다.

```bash
python3 train.py --steps 4000000 --envs 16 --out runs/blind
```

### 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--steps` | 4000000 | 총 스텝 |
| `--envs` | CPU 코어 수 | 병렬 환경. 16 코어면 16 |
| `--reward` | `balanced` | `speed` / `balanced` / `safe` |
| `--d0` | 0.0 | 시작 난이도 |
| `--episode-s` | 20.0 | 에피소드 길이 [초] |
| `--seed` | 0 | |
| `--out` | `runs/ppo` | 저장 경로 |
| `--clip-steer-deg` | 10.0 | 조향 잔차 상한 [도] |
| `--clip-duty` | 0.60 | 듀티 잔차 상한 |
| `--teacher` | off | 특권 관측 + 인코더 |

`--clip-steer-deg 0 --clip-duty 0` 으로 주면 순수 기구학 주행이 된다 (ablation).

### 산출물

```
runs/<이름>/
  final.zip            최종 정책
  vecnorm.pkl          관측 정규화 통계  ← 평가할 때 반드시 같이 준다
  ppo_*_steps.zip      200k 마다 체크포인트
  ppo_vecnormalize_*.pkl
  tb/                  텐서보드 로그
```

> `vecnorm.pkl` 없이 정책만 돌리면 스케일이 어긋난 입력을 받아 결과가 무의미하다.

---

## 2. 증류 (teacher → student)

student 는 고유수용감각 이력만으로 `ẑ` 를 맞히도록 배운다. 정책 머리는 **동결**이고
`AdaptationModule` 만 학습한다. 데이터는 student 가 직접 굴러다니며 모은다 (DAgger).

```bash
python3 distill.py --teacher runs/teacher/final --vecnorm runs/teacher/vecnorm.pkl
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--out` | `runs/teacher/student.pt` | |
| `--iters` | 30 | DAgger 반복 |
| `--steps-per-iter` | 6000 | 반복당 수집 스텝 |
| `--epochs` | 4 | 반복당 회귀 에폭 |
| `--batch` | 512 | |
| `--lr` | 1e-3 | |
| `--w-act` | 0.0 | 행동 손실 가중치. 0 = RMA 설정, >0 = Lee 2020 설정 |
| `--difficulty` | 1.0 | |
| `--buffer` | 200000 | 집계 버퍼 |

---

## 3. 진단 — 증류를 시작해도 되는지

**성공률만 보면 안 된다.** PPO 가 인코더를 무시하고 고유수용감각만으로 풀어버리면
`z` 가 죽은 변수가 되고, 성공률은 멀쩡한데 증류할 내용이 없다.

```bash
python3 probe_teacher.py --model runs/teacher/final --vecnorm runs/teacher/vecnorm.pkl
```

| | 재는 것 | 판정 |
|---|---|---|
| [1] | `z` 의 에피소드 간 표준편차 | max < 0.02 면 인코더가 상수 → 여기서 멈춤 |
| [2] | `z → 흙 파라미터` 선형 R² | 진단용. 판정용 아님 |
| [3] | `z` 를 **다른 에피소드 것으로 바꿔치기** 했을 때 낙폭 | 주 판정 |

[3] 은 같은 씨드(= 같은 지형·흙)로 쌍대 비교하고, 목표 도달 수·이동거리 같은
연속 지표를 주로 본다. 이진 성공률은 24 에피소드에서 표준오차가 10%p 라 검정력이 낮다.
`z=0` 은 정책이 본 적 없는 분포 밖 입력이라 보조로만 쓴다.

`--episodes` (기본 24), `--difficulty` (기본 1.0) 로 조절한다.

---

## 4. 평가

**학습 지형이 아니라 실제 경기장 STL 에서 재는 게 핵심이다.**

```bash
# 실제 규사 경사지형 (주 평가 대상)
python3 eval.py --arena --terrain sand --episodes 30 \
                --model runs/teacher/final --vecnorm runs/teacher/vecnorm.pkl

# 기준선만 (모델 없이)
python3 eval.py --arena --terrain sand --episodes 30

# 절차생성 랜덤 지형
python3 eval.py --difficulty 1.0 --episodes 30
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--model` / `--vecnorm` | 없음 | 없으면 기준선만 |
| `--arena` | off | 실측 STL 로 평가 |
| `--terrain` | `sand` | `sand`(규사 경사지형, 주) / `rock`(암석 착륙지, 후순위) |
| `--episodes` | 20 | |
| `--difficulty` | 0.8 | `--arena` 에서는 **흙에만** 영향 (지형은 항상 실측 STL) |
| `--reward` | `balanced` | |
| `--episode-s` | 20.0 | |

### 비교군

| 이름 | 무엇 |
|---|---|
| 풀스로틀(peak1.0) | 명령 무시, 최대 스로틀 |
| 고정스로틀(peak0.7) | 명령 무시, 스로틀 0.7 고정 |
| 일률감속(×0.7) | 명령 존중, 일률 30% 감속 |
| 경사스케줄 | IMU 피치로 듀티 상한 (고전 제어) |
| 순수 IK | 보정 없음 |
| RL 8-D / 4-D | 조향+속도 잔차 / 속도 잔차만 |

출력은 표 세 개다: 성능(성공/고착/전복/슬립/침하), 멈춘 원인 분해, 잔차 권한 사용률.

---

## 5. 보기

### MuJoCo 뷰어

창이 떠야 하므로 **터미널에서 `!` 를 붙여 직접 실행한다.**

```bash
! cd ~/ksrc/sim/rl && python3 view.py
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--terrain` | `sand` | `sand` / `rock` / `proc`(절차생성) |
| `--d` | 1.0 | 난이도 |
| `--model` / `--vecnorm` | 없음 | 없으면 순수 IK |
| `--seed` | 0 | 바꾸면 다른 맵 |
| `--static` | off | 물리 끄고 지형만 (제일 가벼움) |
| `--realtime` | 1.0 | 재생 배속. 0.3 이면 슬립 관찰용 |

마우스: 좌드래그 회전 / 우드래그 이동 / 휠 줌. 스페이스 일시정지.
에피소드가 끝나면 결과를 찍고 자동 리셋한다.

### 텐서보드

```bash
! cd ~/ksrc/sim/rl && tensorboard --logdir runs/teacher/tb
# http://localhost:6006
```

여러 실행을 한 번에 비교하려면 상위 폴더를 준다:

```bash
! cd ~/ksrc/sim/rl && tensorboard --logdir runs
```

올라가는 스칼라:

| 그룹 | 내용 |
|---|---|
| `rollout/` | success, stuck, tip, oob, slip, sink, goals, dist, frac, f_belly, f_lifted, duty_sat, duty_use, steer_use |
| `stuck_cause/` | belly, lost_load, blocked, slip_stuck (멈춘 원인 비율) |
| `curriculum/` | difficulty, success_rate |
| `train/` | SB3 기본 (loss, entropy, explained_variance, approx_kl …) |

### 텐서보드 없이 돌린 학습

stdout 로그를 사후에 그림으로 만든다.

```bash
python3 plotlog.py train.log -o trainlog.png
```

### 모델 2D 단면

```bash
python3 draw.py
```

---

## 전형적인 흐름

```bash
bash ../build.sh                                    # 기구학 C 빌드
python3 check_model.py                              # 물리 9종 통과 확인

python3 train.py --teacher --envs 16 --out runs/teacher      # ~50분
python3 probe_teacher.py --model runs/teacher/final \
        --vecnorm runs/teacher/vecnorm.pkl          # z 를 쓰는가? 아니면 여기서 멈춤

python3 distill.py --teacher runs/teacher/final \
        --vecnorm runs/teacher/vecnorm.pkl          # student 증류
python3 train.py --envs 16 --out runs/blind                  # 대조군 ~50분

python3 eval.py --arena --terrain sand --episodes 30 \
        --model runs/teacher/final --vecnorm runs/teacher/vecnorm.pkl
```

---

## 주의

- **`pkill -f` 로 학습을 죽이지 말 것.** 패턴이 자기 셸까지 잡아 세션이 같이 죽는다
  (이 프로젝트에서 네 번 발생). 실행할 때 PID 를 받아두고 그 PID 로 `kill` 한다.
- `--envs` 는 물리 코어 수를 넘기지 않는다. 16 코어에서 16 이 최대다.
- 학습 중에는 CPU 가 포화되므로 뷰어·평가를 같이 돌리면 둘 다 느려진다.
- 기하(`config.py`)를 바꾸면 이전 정책은 못 쓴다. teacher 부터 다시 돌려야 한다.
