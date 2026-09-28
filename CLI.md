# CLI

0~5절의 학습·평가 명령은 `sim/rl/`에서 실행한다. 6절의 런치는 `ros/`, 조이스틱은
저장소 루트에서 실행한다.

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

조향 목표각이 한 주기에 60° 넘게 바뀌는 큰 방향 전환에서는 네 바퀴가 목표의
10° 이내에 들어올 때까지 구동을 함께 보류하는 규칙이 `RoverEnv`의 기본값이다.
학습·평가·RViz 순수 IK·RViz RL·텔레옵에 동일하게 적용된다. 평소 작은
조향 오차마다 멈추지 않도록 큰 전환에만 대기를 시작한다.
기존 `runs/s2`는 이 규칙과 새 경로추종 보상을 넣기 전에 학습했으므로 새
보상의 성능 기준으로 간주하지 않는다. 새 정책은 별도 디렉터리에 학습하고
순수 IK와 짝비교한다. 보상만 비교하려면 같은 명령·시드에
`--reward position_only`(이동방향 항 없음) 또는 `--reward legacy`(차체 헤딩 항)를 주고
각각 별도 디렉터리에 학습한다.

```bash
cd ~/ksrc/sim/rl
python3 train.py --steps 4000000 --envs 16 --terrain sand --stage1 --out runs/s3_path_reward
python3 compare_policy.py --model runs/s3_path_reward/final \
  --vecnorm runs/s3_path_reward/vecnorm.pkl --episodes 240
```

두 명령의 `--drive-align-gate-deg` 기본값은 대기 해제 오차 10°다. 이전 동작과 비교할 때만
둘 다 `--drive-align-gate-deg 0`으로 맞춘다.

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
| `--reward` | `balanced` | `speed` / `balanced` / `safe` / `position_only` / `legacy` |
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

# s2 정책을 순수 IK와 같은 시드·미학습 경로에서 직접 짝비교
python3 compare_policy.py --model runs/s2/final --vecnorm runs/s2/vecnorm.pkl \
                          --episodes 100
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
| PP+IK | 잔차 0 |
| 일률감속 ×0.8 / ×0.6 | 기본 차체 명령의 속도를 낮춤 |
| 경사스케줄 | IMU 피치로 속도 상한 조정 |
| RL 3-D | 차체속도 `(Δvx, Δvy, Δω)` 잔차 |

`eval.py`는 성능·고착 원인·경로오차·잔차 사용량을 출력한다. 정책 대 순수 IK의
직접 판정은 `compare_policy.py`의 짝비교와 95% 신뢰구간을 사용한다.

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

## 6. 대회맵 주행: RViz 목표 또는 텔레옵

세 모드는 아래처럼 실행한다. **한 번에 하나의 런치만** 띄운다. 기본 지형은 규사
대회맵이고, 암석 대회맵은 세 명령 모두 `terrain:=rock`으로 바꾼다.

```bash
cd ~/ksrc/ros
source /opt/ros/humble/setup.bash
python3 make_maps.py  # 맵 파일이 없거나 지형 생성 코드를 바꿨을 때만

ros2 launch nav2_ksrc.launch.py terrain:=sand                  # RViz 2D Goal Pose + 순수 IK
ros2 launch nav2_ksrc.launch.py terrain:=sand rl:=true         # RViz 2D Goal Pose + s2 RL 잔차
ros2 launch nav2_ksrc.launch.py terrain:=sand mode:=teleop     # 대회맵 + 수동 텔레옵
```

**텔레옵 모드**는 런치를 켜 둔 채 두 번째 터미널에서 기존 마우스 조이스틱을
ROS 모드로 실행한다. 왼쪽 패드는 전후·게걸음, 오른쪽 패드 또는 Q/E는 회전,
스페이스바는 정지다. 패드에서 손을 떼거나 창 포커스를 잃으면 0 명령을 보낸다.
텔레옵에서 큰 방향 전환 중에는 차체가 거의 멈출 때까지 최대 0.5초 감속한다.
조향 목표가 한 주기에 60° 넘게 바뀌면 네 바퀴가 새 조향각의 10° 이내에
들어온 뒤 구동하는 규칙은 모든 모드의 `RoverEnv`에 공통 적용된다.

```bash
cd ~/ksrc
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
python3 pi/teleop/teleop_joystick.py --ros --max-lin 0.25 --max-ang 0.8
```

텔레옵 런치는 MuJoCo 뷰어와 RViz의 대회맵·로버 위치를 함께 띄우며,
`map_server`만 활성화한다. Nav2 컨트롤러가 없어 `/cmd_vel` 발행자가 충돌하지
않는다. 이 모드에서 RViz의 2D Goal Pose는 주행 명령이 아니다. 조이스틱이
직접 `/cmd_vel`을 낸다. MuJoCo 창이 부담되면 `view:=false`를 런치에 추가해
RViz 맵만 볼 수 있다. 종료 후에도 명령이 끊기면 브리지의 0.5초 워치독이 정지한다.
규사맵의 최초 스폰과 `/reset_sim` 복귀 위치는 3° 이하 평탄 구역으로 제한하며,
경사 경계에서 로버 바퀴가 걸치지 않도록 풋프린트만큼 여유를 둔다. 같은 시드로
리셋하므로 복귀 위치는 매번 같다. 학습·평가 에피소드의 시작 위치 분포는 바꾸지 않았다.
텔레옵의 사전 감속만 끄려면 `teleop_guard:=false`, 조향각 구동 게이트만
끄려면 `drive_align_gate_deg:=0`을 런치에 추가한다. 이전 두 기능 모두 없던
동작과 비교하려면 둘 다 지정한다.
ROS 발행 옵션은 현재 `teleop_joystick.py`에 추가했다. `teleop_keyboard.py`와
`teleop_terminal.py`는 기존 Nucleo 직렬 프로토콜용이다.

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
ros2 topic echo /cmd_vel --once          # 조이스틱을 움직인 상태에서 실행
ros2 topic echo /cmd_vel_applied --once  # 실제 IK 입력
ros2 topic echo /odom --once             # 맵 위 로버 위치
```

현재 저장소의 ROS 런치 파일은 `ros/nav2_ksrc.launch.py` 하나다. 기본 `nav2`
모드에서는 MuJoCo 브리지, `map_server`, `planner_server`, `controller_server`
(MPPI Omni), `behavior_server`, `bt_navigator`, `lifecycle_manager_navigation`,
RViz를 실행한다. `teleop` 모드에서는 브리지·맵 서버·RViz만 실행한다.
`amcl`과 `velocity_smoother`는 파라미터 파일에 항목이 있지만 현재 런치에서는 실행하지
않는다. 측위 대신 브리지가 `map → odom` 항등 변환과 참값 `odom → base_link`를 낸다.
Nav2 컨트롤러의 `/cmd_vel`은 브리지가 받아 공유 스워브 IK를 거쳐 **MuJoCo 조향·구동
액추에이터**에 적용한다. 브리지는 기본적으로 IK 전에 차체 명령을 0.10초 시정수로
저역통과하고 그 결과를 `/cmd_vel_filtered`에도 발행한다. 0 명령은 지연 없이 정지한다.
기본은 순수 IK(잔차 0)다. `rl:=true`로 띄우면 `runs/s2/final.zip`과 짝인
`vecnorm.pkl`을 로드하고 Nav2 `/plan` 기준 관측으로 3D 잔차를 계산한다.
`s2`는 공통 조향각 게이트 적용 전에 학습했으므로 이 런치는 보정 적용 확인용이다.
재학습한 모델은 `rl_model:=... rl_vecnorm:=...`로 지정한다.
`/rl_action`은 정규화 행동, `/rl_delta`는 요청된 차체속도 보정,
`/cmd_vel_applied`는 포화 처리 후 실제 IK 입력이다.
이 런치만으로 실제 STS/Cytron 모터에 명령이 나가지는 않는다.
STM32의 현재 `Swerve_Drive()`는 조향 명령과 구동 PWM을 바로 보내므로,
실기에서도 같은 정렬 대기를 쓰려면 서보 각 피드백·캘리브레이션을 확인한 뒤
펌웨어 구동 제어에 별도로 반영해야 한다.

```bash
cd ~/ksrc/ros
source /opt/ros/humble/setup.bash
ros2 launch nav2_ksrc.launch.py terrain:=sand rl:=true
# 필요하면 잔차만 절반으로 줄여 확인: rl_scale:=0.5
```

RViz의 **2D Goal Pose**로 목표를 준 뒤 아래 토픽을 보면 정책이 적용됐는지
확인할 수 있다. `topic echo`는 계속 실행되므로 관심 있는 토픽마다 별도 터미널을
쓴다. 목표에 도착해 `/cmd_vel`이 0이면 정책 잔차도 0이다.

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
ros2 topic echo /plan --once             # Nav2 경로 수신
ros2 topic echo /rl_action                # [Delta vx, Delta vy, Delta omega] 정규화 행동
ros2 topic echo /rl_delta                 # 요청 보정: m/s, m/s, rad/s
ros2 topic echo /cmd_vel_filtered         # 보정 전 Nav2 명령
ros2 topic echo /cmd_vel_applied          # 보정 후 IK 입력
ros2 service call /set_rl std_srvs/srv/SetBool '{data: false}'
ros2 service call /set_rl std_srvs/srv/SetBool '{data: true}'
```

필터 명령과 최종 명령은 포화 처리 때문에 `/rl_delta`만큼 정확히 차이 나지 않을
수 있다. 정책은 pure pursuit 출력으로 학습됐으므로 Nav2 MPPI에서 성능 향상을
보장하는 실험은 아니다. 실제 주행 결과는 순수 IK와 같은 목표에서 비교한다.

### 왜 한 런치로 합쳤나

브리지와 Nav2를 각각 실행하던 구성에서는 브리지 누락, 서로 다른 지형·맵, 서로 다른
ROS 도메인 때문에 Nav2가 `/odom`·TF를 못 받거나 다른 실행의 노드와 섞일 수 있었다.
실패 당시 로그에는 `map_server` 활성화 실패, 오래된 `/odom` 타임스탬프, 현재 맵과
크기가 다른 맵이 함께 보였다. **다른 ROS 실행과의 충돌은 유력한 원인이지만 로그만으로
단정할 수는 없다.** 통합 런치는 같은 프로세스 묶음에서 지형에 맞는 맵을 고르고,
모든 자식에 `ROS_DOMAIN_ID=77`, `ROS_LOCALHOST_ONLY=1`을 설정해 이 문제를 피한다.
ROS 자체가 한 런치를 요구하는 것은 아니다. 아래처럼 분리 실행해도 같은 환경·맵·
라이프사이클 설정을 맞추면 된다. 기존 맵 파일 누락은 `make_maps.py`로 생성한다.

### 왜 2D Goal Pose 뒤에 조향이 튀었나

RViz의 **2D Goal Pose는 차체 속도나 바퀴각을 직접 보내지 않는다.** 클릭 위치가 목표
`x, y`, 드래그 방향이 최종 yaw이며, Nav2가 경로와 `/cmd_vel`을 계산한다. 이전 시뮬
조향 액추에이터는 큰 목표각을 즉시 받는 위치 제어(`kp=12`, `kv=0`)였다. 고정된
−90° 목표각 시험에서도 실제 바퀴가 약 −132°까지 지나치고 최대 35.6 rad/s로 움직였다.
따라서 MPPI 출력이 변하기 전에 조향 모델 자체가 과도하게 반응했다. 현재 코드는
`steer_kv=0.30` 감쇠를 넣고 목표각을 제어 주기마다 최대 `5.2 rad/s ÷ 50 Hz`만큼
진행시킨다. 같은 스텝 시험에서 초과각은 0.14°, 최대 속도는 약 4.7 rad/s였다.
빠르게 변하는 명령 시험에서는 최대 5.18 rad/s였다. 이는 시뮬 단위 시험 결과이며
RViz GUI에서 연속 주행한 결과를 뜻하지 않는다. 저장된 RL 정책의 성능도 변경된
조향 물리에서 다시 평가해야 한다.

RViz에서 **차체 위치가 순간이동하던 문제**도 따로 있었다. 전복 후 브리지가 자동으로
환경을 리셋해 스폰 위치를 다시 발행했기 때문이다. 지금은 전복 시 구동을 멈추고
현재 위치를 유지하며, `/reset_sim`을 호출할 때만 원래 스폰으로 리셋한다.

목표를 눌러도 움직이지 않는 현상은 별개로 확인한다. Nav2 노드가 `active`가 아니면
목표가 수행되지 않는다. 또 목표가 시작 위치에서 0.15 m 이내이고 방향 차이가 0.50 rad
이내면 현재 목표 허용오차 안이라 즉시 `Reached the goal!`이 된다. 실제 시험 중 한
목표는 시작 위치에서 약 0.04 m 떨어져 있었다. 먼저 충분히 떨어진 통과 가능 지점을
클릭하고 드래그한다. RViz의 **2D Pose Estimate**는 이 브리지의 스폰 위치를 바꾸지
않으며, `Publish Point`는 현재 웨이포인트 명령으로 연결되지 않았다.

### 통합 실행 (권장)

```bash
source /opt/ros/humble/setup.bash
cd ~/ksrc/ros
python3 make_maps.py                         # 맵이 없거나 지형/맵 생성 코드를 바꿨을 때
ros2 launch nav2_ksrc.launch.py terrain:=sand
```

`terrain:=rock`으로 암석 맵·지형을 선택한다. `rviz:=false`는 RViz만,
`view:=false`는 MuJoCo 창만 끈다. `cmd_filter_tau:=0`이면 이번 차체 명령 필터를
꺼서 기존 동작과 비교할 수 있다. 이 필터는 **브리지 안에만** 있으며 Nav2의
`velocity_smoother` 노드를 실행하는 것은 아니다. `domain_id:=78`처럼 도메인을 바꾸면 별도
터미널의 `ROS_DOMAIN_ID`도 같은 값으로 맞춘다. **통합 런치의 도메인 설정은 그
런치가 띄운 자식 프로세스에만 적용된다.** 세 번째 진단 터미널에는 자동으로
전파되지 않는다. 종료는 런치를 실행한 터미널에서
Ctrl-C를 누른다. 동일한 Nav2나 브리지를 중복 실행하지 않는다.

### 브리지와 Nav2 런치를 분리 실행

두 터미널 **각각** 먼저 다음 환경을 설정한다. 별도 `ros2` 진단 터미널도 같다.

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77
export ROS_LOCALHOST_ONLY=1
cd ~/ksrc/ros
```

터미널 A: 브리지. `--view`를 빼면 MuJoCo 창 없이 실행된다. 기본 `--seed 0`,
`--d 0.5`, `--cmd-filter-tau 0.10`이며, 지형은 Nav2 런치와 같아야 한다.
필터 비교 시 `--cmd-filter-tau 0`을 준다. 정책 시험은 아래 둘째 명령처럼
모델과 vecnorm을 함께 지정한다.

```bash
python3 mujoco_bridge.py --terrain sand --view
python3 mujoco_bridge.py --terrain sand --view \
  --model ../sim/rl/runs/s2/final.zip --vecnorm ../sim/rl/runs/s2/vecnorm.pkl
```

터미널 B: 같은 지형의 Nav2와 RViz. 이미 브리지를 띄웠으므로 `bridge:=false`를 준다.
맵이 없다면 먼저 `python3 make_maps.py`를 실행한다.

```bash
ros2 launch nav2_ksrc.launch.py terrain:=sand bridge:=false
```

### 런치 없이 노드를 하나씩 실행

각 명령을 **서로 다른 터미널**에서 실행한다. 모든 터미널에 위의 `source`,
`ROS_DOMAIN_ID`, `ROS_LOCALHOST_ONLY`, `cd`를 반복한다. 맵 파일이 없다면 먼저
`python3 make_maps.py`를 실행한다. 실행 순서는 브리지 → Nav2 서버 5개 →
라이프사이클 매니저 → RViz다. `map_server`의 `yaml_filename`은 런치가 덮어쓰던
값이므로 수동 실행에서는 반드시 실제 맵 파일을 준다. 암석 지형이면 브리지의
`sand`와 맵 파일명의 `sand`를 함께 `rock`으로 바꾼다.

```bash
# 터미널 1: MuJoCo 브리지
python3 mujoco_bridge.py --terrain sand --view

# 터미널 2: 정적 맵 서버
ros2 run nav2_map_server map_server --ros-args \
  --params-file "$PWD/nav2_params.yaml" \
  -p yaml_filename:="$PWD/maps/arena_sand.yaml"

# 터미널 3: 전역 경로 계획
ros2 run nav2_planner planner_server --ros-args \
  --params-file "$PWD/nav2_params.yaml"

# 터미널 4: MPPI Omni 경로 추종 → /cmd_vel
ros2 run nav2_controller controller_server --ros-args \
  --params-file "$PWD/nav2_params.yaml"

# 터미널 5: 복구 행동
ros2 run nav2_behaviors behavior_server --ros-args \
  --params-file "$PWD/nav2_params.yaml"

# 터미널 6: 목표 행동 트리
ros2 run nav2_bt_navigator bt_navigator --ros-args \
  --params-file "$PWD/nav2_params.yaml"

# 터미널 7: 위 5개 Nav2 서버를 configure → activate
ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args \
  -r __node:=lifecycle_manager_navigation \
  -p use_sim_time:=false -p autostart:=true \
  -p 'node_names:=[map_server,planner_server,controller_server,behavior_server,bt_navigator]'

# 터미널 8: RViz
ros2 run rviz2 rviz2 -d "$PWD/rviz/ksrc.rviz" --ros-args -p use_sim_time:=false
```

라이프사이클 매니저 없이 서버 실행 명령만 입력하면 Nav2 서버는 `unconfigured`에
머물러 주행할 수 없다. 현재 런치와 동일하게 `use_sim_time:=false`를 유지한다.

### 목표를 보내기 전 확인

통합 런치를 썼더라도 진단용 터미널에서 다음 환경을 다시 설정한다.
`domain_id:=`를 바꿨다면 `77` 대신 그 값을 쓴다.

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77
export ROS_LOCALHOST_ONLY=1
ros2 node list --no-daemon
ros2 topic info /cmd_vel -v --no-daemon
```

`node list`가 비어 있으면 도메인이 다르거나 런치가 종료된 것이다. 노드는
보이는데 `/cmd_vel` 발행자가 없다면 `controller_server`가 실행·활성화됐는지
확인한다. 발행자가 있어도 목표 주행 중이 아니면 `topic echo`는 새 메시지가
올 때까지 기다린다. 다음 명령을 한 번에 하나씩 실행한다.
`topic hz`와 끝나지 않는 `topic echo`는 Ctrl-C로 종료한다. `map_server`와
`bt_navigator`가 `active`이고 `/odom`이 계속 갱신돼야 한다. RViz에서
**2D Goal Pose**를 눌러 목표점에서 클릭·드래그한다. 전복으로 구동이 잠겼을 때는
`/reset_sim`을 호출하면 처음 `--seed`의 스폰으로 돌아간다.

**클릭한 지점이 목적지이고, 드래그는 목적지에서의 최종 방향만 정한다.** 로버 위를
클릭한 뒤 멀리 드래그해도 목적지는 로버 근처다. 최근 로그에서도 현재 위치
`(1.36, 0.14)`에 대해 목표가 `(1.34, 0.14)`여서 15 cm 목표 허용오차 안이었다.
구동 확인은 로버에서 0.5 m 이상 떨어진 주행 가능 지점을 먼저 클릭한다.

```bash
ros2 lifecycle get /map_server
ros2 lifecycle get /bt_navigator
ros2 topic hz /odom
ros2 topic echo /goal_pose --once             # 실행한 다음 RViz에서 목표를 찍는다
ros2 topic echo /odom --once                  # 현재 로버 좌표와 비교
ros2 topic echo /cmd_vel --no-daemon         # Nav2 원본 명령. 목표를 찍기 전에 켠다
ros2 topic echo /cmd_vel_filtered --no-daemon  # RL 전 필터 명령
ros2 topic echo /cmd_vel_applied --no-daemon   # IK 에 들어간 최종 명령
ros2 topic echo /rl_action --no-daemon         # 정책 행동 (rl:=true 시)
ros2 service call /reset_sim std_srvs/srv/Trigger '{}'
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
