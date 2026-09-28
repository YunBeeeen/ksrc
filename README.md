# KSRC 2026 — 4륜 독립조향 우주로버

2026 대한민국 우주로버 챌린지(우주항공청) 출전 로버의 **저수준 제어기 · 시뮬레이터 ·
강화학습** 코드베이스임.

바퀴 4개를 각각 독립 **조향** + **구동** 하는 스워브 방식임. 중앙 베벨 디퍼렌셜이
좌우 로커를 1:1 역방향으로 묶어 울퉁불퉁한 지형에서 네 바퀴 접지를 유지함.

| 항목 | 값 |
|---|---|
| 총질량 | 2.26 kg (규정 3 kg 이하) |
| 크기 | 284 × 263 × 173 mm (규정 300×300×200 이하) |
| 휠베이스 / 트랙 | 183.6 / 236.7 mm |
| 바퀴 | 지름 79.9 mm, 폭 35 mm, 그라우저 약 28개 |
| 지상고 | 9.6 mm (다리 하우징 최저점 기준) |
| 구동 | SPG30E-GR131 ×4 (12V, 131:1, 76 rpm 무부하, 20 kg·cm 정지토크) |
| 조향 | FEETECH STS3215 ×4 (반이중 UART 버스, 1 Mbaud, ±180°) |
| 최고 속도 | 0.318 m/s |
| 배터리 | 3S 650mAh 120C 리포 |
| MCU / SBC | STM32 Nucleo-F446RE / Raspberry Pi 5 + AI HAT |

---

## 시스템 구조

```
        조작자 / 자율
             │
    ┌────────┴────────┐
    │  Raspberry Pi 5 │   pi/teleop        조이스틱 → cmd_vel
    │                 │   Nav2 (MPPI/Omni) 경로 → cmd_vel   ← 시뮬 검증 완료
    │                 │   (계획) D435 ICP  대회장 맵 정합 → 절대 위치
    │                 │   RL 정책          cmd_vel 위의 잔차 보정
    └────────┬────────┘
             │  아스키/바이너리 시리얼 (115200)
    ┌────────┴────────┐
    │ Nucleo-F446RE   │   swerve_kinematics.c   cmd_vel → 바퀴 4개 각·속도
    │                 │   swerve_fold_to_limit  ±180° 등가해 선택
    │                 │   mdd3a_driver.c        PWM
    │                 │   encoder.c             엔코더 4채널
    └────────┬────────┘
             │
      구동모터 ×4      조향서보 ×4
```

명령 인터페이스는 `cmd_vel = (vx, vy, ω)` 한 지점으로 모임. 조이스틱 · Nav2 ·
RL 잔차가 모두 같은 지점에 붙으므로 서로 교체 가능함.

**시뮬레이터가 펌웨어 C 코드를 그대로 호출함.** `sim/build.sh` 가
`firmware/common/*.c` 를 `libksrc_common.so` 로 빌드하고 파이썬이 ctypes 로 부름.
기구학을 파이썬으로 재구현하면 언젠가 반드시 어긋나는데, 정책이 기구학 위의
**잔차**를 학습하는 구조라 어긋나는 순간 정책이 통째로 무의미해짐.

---

## 저장소 구조

```
firmware/
  common/                   MCU 비의존 순수 C + 호스트 유닛테스트
  src2026-rover-firmware/   보드에 올라가는 CubeIDE 프로젝트 (별도 git 저장소)
pi/teleop/                  조이스틱 텔레옵 (Python)
ros/                        ROS2 / Nav2 통합
  make_maps.py              주행가능 마스크 -> Nav2 정적 맵
  maps/                     arena_{sand,rock}.{pgm,yaml}
  mujoco_bridge.py          MuJoCo <-> ROS2 (/odom /tf 발행, /cmd_vel 구독)
  nav2_params.yaml          Nav2 파라미터 (MPPI motion_model=Omni)
  nav2_ksrc.launch.py       최소 구성 런치
  rviz/ksrc.rviz            RViz 설정
sim/
  ksrc_common.py            펌웨어 C 를 ctypes 로 호출
  rl/                       MuJoCo 물리 + 강화학습
CLI.md                      전체 실행 명령 모음
```

### `firmware/common/` — 펌웨어 공용 로직

MCU 의존성이 없는 순수 C 라 **호스트 PC 에서 컴파일·테스트** 가능하고
**시뮬레이터가 같은 코드를 호출**함.

| 파일 | 역할 |
|---|---|
| `swerve_kinematics.c` | 스워브 역기구학 + ±180° 등가해 선택(`swerve_fold_to_limit`) |
| `mdd3a_driver.c` | Cytron MDD3A PWM 정규화 |
| `teleop_protocol.c` | 텔레옵 바이너리 프레임 인코딩/디코딩 |
| `sts3215_protocol.c` | STS3215 패킷 조립 |
| `tests/` | 위 모듈 유닛테스트 |

### `sim/rl/` — MuJoCo 물리 + 강화학습

| 파일 | 역할 |
|---|---|
| `config.py` | 로버 물리 파라미터 + 센서/구동 노이즈. 출처를 [확정]/[추정]/[미정] 3등급 표기 |
| `mjcf.py` | config → MuJoCo MJCF 모델 생성 |
| `terramech.py` | 슬립-침하 모델. 고착 재현의 핵심 |
| `terrain.py` | 절차 지형 생성 · 대회장 맵 로드 · 주행가능 마스크 · 미세요철 교란 |
| `path.py` | 기준경로 생성 + 홀로노믹 pure pursuit (Nav2 컨트롤러 미러링) |
| `reward.py` | 보상 가중치와 프리셋 (speed / balanced / safe / position_only / legacy) |
| `env.py` | Gymnasium 환경 |
| `train.py` | PPO 학습 (stable-baselines3) |
| `eval.py` | 대회장 지형 평가. 고전 제어 기준선 4종과 비교 |
| `traction.py` | 고전 제어 기준선 (경사스케줄 듀티 제한 / 엔코더 TCS) |
| `nets.py` | teacher-student 신경망 (특권 인코더 / 적응 모듈) |
| `distill.py` | teacher → student 증류 (DAgger + 잠재변수 회귀) |
| `probe_teacher.py` | teacher 가 특권정보를 실제로 쓰는지 진단 |
| `view.py` | MuJoCo 뷰어 (정책 재생). `--path` 로 지정 경로, `--stage1` 로 학습 조건 재현 |
| `draw_path.py` | 지형 지도에서 마우스로 경로 찍기 |
| `check_model.py` | 모델 물리 검증 13종 |
| `test_terramech.py` | 견인력 곡선·고착 재현 검증 |
| `plotlog.py` | 학습 로그 → 그림 |
| `draw.py` | 모델 2D 단면 렌더 |

---

## 실행 명령

전체 인자는 **[CLI.md](CLI.md)** 에 있음. 아래는 자주 쓰는 것만임.

### 빌드 · 검증

```bash
cd sim && bash build.sh          # 기구학 C → libksrc_common.so
cd rl  && python3 check_model.py # 물리 검증 13종
python3 test_terramech.py        # 견인력 곡선 검증
```

### 학습

```bash
cd sim/rl

# Stage 1 — 경로추종 sanity check (경로 세트 고정, 랜덤화·교란·커리큘럼 OFF)
python3 train.py --steps 4000000 --envs 16 --terrain sand --stage1 --out runs/s3_path_reward

# Stage 2 — 랜덤화·커리큘럼 켜고 특권 관측(teacher) 붙임
python3 train.py --steps 4000000 --envs 16 --teacher --terrain sand --out runs/foo
```

| 옵션 | 뜻 |
|---|---|
| `--stage1` | 경로 세트 고정(10개) + 랜덤화·노이즈·교란·커리큘럼 전부 OFF. **3D 잔차가 경로추종을 배우는지만** 본다 |
| `--terrain sand\|rock\|proc` | `sand`/`rock` 은 대회장 맵 고정, `proc` 은 절차 생성 |
| `--teacher` | 특권 관측을 붙임 (Stage 2) |
| `--path-seed` / `--n-paths` | Stage 1 경로 세트. **학습과 평가는 다른 seed** 를 써야 함 |
| `--gamma` | 기본 0.998 (시상수 10초). 0.99 는 50Hz 에서 2초라 완주 보너스가 안 보임 |
| `--drive-align-gate-deg` | 기본 10°. 60° 초과 큰 방향 전환 후 구동 재개 오차, 0이면 이전 동작과 비교 |
| `--w-energy` | 현재 미지원: 차축 부하 모델 검증 전에는 오류를 내고 학습을 시작하지 않음 |

### 현재 학습 환경 (`sim/rl/env.py`, `train.py`)

| 구성 | 현재 구현 |
|---|---|
| 물리·제어 주기 | MuJoCo 2 ms 물리 스텝 10회당 정책 1회(50 Hz). 에피소드 최대 20초/1000스텝 |
| 기준 제어 | 경로 위 pure pursuit가 `cmd_nom=(vx, vy, ω)` 생성 → 정책의 3차원 행동 `a∈[-1,1]^3`에 `(0.03 m/s, 0.03 m/s, 0.10 rad/s)`를 곱해 가산 → 공유 C 스워브 IK → 바퀴별 개루프 PWM 듀티·조향각 |
| 저수준 보호 | 바퀴 속도 한계를 넘으면 차체 명령을 공통 비율로 축소. 목표 조향각이 한 주기에 60° 넘게 바뀌면 네 바퀴가 목표의 10° 이내로 정렬될 때까지 구동 보류 |
| 경로·시작 | 주행가능 마스크(풋프린트 경사 최대 21°, 설명되지 않는 단차 최대 40 mm) 위에서 경로 생성. 고정 경로 세트에서는 경로 길이의 앞 0~60% 위치에서 시작, 초기 기수는 그 지점 접선의 ±30°. 순항속도는 무부하 최고속도의 45~100%에서 매 에피소드 추출 |
| 관측 | 프레임당 33개(`cmd_nom`, 직전 바퀴 명령·엔코더 속도·조향각, 자세·IMU, lookahead 2점, 횡오차, 기수오차의 cos/sin, 직전 행동) × 50프레임 = 1650차원(1초 이력). `--teacher`는 흙·슬립·침하 등 특권값 43개를 추가해 1693차원 |
| 종료 | 목표까지 경로 잔여길이와 실제 거리 **모두** 0.25 m 미만이면 성공. 전복(roll/pitch 60° 초과), 최근 2초 이동 0.02 m 미만이면서 슬립 0.5 초과, 경계 이탈은 실패 종료. 20초 도달은 시간 만료 |

위 Stage 1 실행 예시(`--terrain sand --stage1`)는 고정 경로 10개(seed 1234), 원본 대회 높이맵,
고정 규사 파라미터를 쓴다. **센서·구동 노이즈, 물리 파라미터 랜덤화, 높이맵
교란, 커리큘럼은 꺼진다.** 경로 선택·시작 위치·기수·순항속도는 여전히 매
에피소드 달라진다. `train.py`의 `--terrain` 기본값은 `proc`이므로 Stage 1에서는
`--terrain sand`를 명시한다. `--terrain proc --stage1`은 **경로만 sand 맵에서
생성하고 지형은 proc으로 만드는 불일치**가 있어 현재 비교에 쓰지 않는다.
평가 경로는 별도 seed(기본 4321)로 생성해 같은 경로 암기를 구분한다.

Stage 2(`--stage1` 없음)는 `--terrain sand|rock`이면 해당 대회 높이맵을 쓰고
매 에피소드 미세요철을 교란한다(기본 표준편차 `0.020 m × 난이도`). `proc`이면
8×8 m 절차 지형을 새로 만든다. 흙 다짐·두께와 바퀴별 마찰 패치, 미측정
질량·로커 스토퍼, 배터리 전압, 엔코더·IMU 오차, 서보 응답, 0~2스텝 제어
지연, 저역통과된 측위 오차를 랜덤화한다. 최근 100개 에피소드 성공률이
60% 이상이면 난이도를 0.05씩 올린다. 학습기는 SB3 PPO, 관측·보상
`VecNormalize`, 기본 `γ=0.998`, 512스텝/환경 rollout, batch 4096,
6 epoch, 학습률 `3e-4`다. `--teacher`는 이 Stage 2의 선택 사항이다.

**현재 관측 제약:** pure pursuit 명령과 lookahead는 잡음이 있는 추정 위치로
계산하지만, 33차원 관측의 횡오차 `e_y`와 기수오차 `e_psi`는 아직 참 위치·자세로
계산한다. 실기 투입 전 이 두 입력도 Nav2 경로와 추정 자세에서 산출해야 하며,
지금의 Stage 2가 측위 오차를 관측 전체에 일관되게 적용한다고 해석하면 안 된다.

### 현재 보상 함수 (`sim/rl/reward.py`, 기본 `balanced`)

정책 행동을 적용해 **물리를 진행한 뒤** 참 위치로 보상을 계산한다. 경로 호길이
진행량을 `p=clip(Δs/(v_cruise·Δt), 0, 1)`, 실제 이동량을
`m=clip(‖Δxy‖/(v_cruise·Δt), 0, 1)`로 정규화한다. `Δs`는 한 주기의 실제
이동거리로 상한을 두어 코너를 잘라도 가짜 진행 보상을 받지 않는다.
`e_y`는 경로 횡오차, `e_course`는 **실제 이동 방향과 경로 접선** 사이의 오차다.
2 cm/s 이하에서는 이동 방향이 불안정해 `e_course=0`으로 둔다.

```text
G = exp(-(e_y / 0.10 m)^2) · max(0, cos(e_course))
r_step = 0.15 p + 0.85 p G
       - 0.05 m min((e_y / 0.10 m)^2, 4)
       - 0.02 m exp(-(e_y / 0.10 m)^2) min((e_course / 0.35 rad)^2, 4)
       - 0.10 mean((a_t - 2a_(t-1) + a_(t-2))^2)
       - 0.03 mean(a_t^2)
```

완주 `+20`, 전복 `-10`, 고착 `-5`, 경계 이탈 `-10`은 해당 스텝에 한 번
더한다. 차체 기수 `e_psi`는 기본 보상에 없으므로 바퀴를 돌려 게걸음·후진으로
같은 경로를 따라도 벌점을 받지 않는다. `position_only`는 이동 방향 게이트·
횡오차/이동방향 벌점이 없는 비교군, `legacy`는 차체 기수 게이트를 쓰던
이전 비교군이다. `speed`, `safe`는 가중치·허용 폭을 바꾼 프리셋이다.
전력·에너지·슬립·침하는 **현재 진단 지표이지 보상 항이 아니다.** 차축 부하
모델이 미검증이라 에너지 가중치 옵션 `--w-energy`는 오류를 내고 중단한다.

로그는 `runs/<이름>/tb` 로 감. 실행마다 따로 쌓이므로 **전부 비교하려면 `--logdir runs`** 를 줌.
현재 `RoverEnv`는 조향 목표가 한 주기에 60° 넘게 바뀌는 큰 방향 전환에서만
네 바퀴 구동을 함께 보류하고, 모두 목표각의 10° 이내에 들면 다시 구동한다.
학습·평가·ROS 주행에 공통 적용된다. 저장된 `runs/s2`는 이 조향 대기와
새 경로추종 보상을 넣기 전에 학습했으므로 새 보상의 성능을 나타내지 않는다.
실제 STM32 펌웨어의 모터 출력에는 이 정렬 대기가 아직 연결되지 않았다.

### 재생 (play)

```bash
cd sim/rl

# 순수 기구학 (잔차 0) — 기준선 확인
python3 view.py --terrain sand --d 1.0 --realtime      # --realtime 0.5 로 느리게

# 학습된 정책.  vecnorm 을 빼면 관측 스케일이 어긋나 결과가 무의미함
python3 view.py --terrain sand --stage1 --realtime \
        --model runs/s2/final --vecnorm runs/s2/vecnorm.pkl

# 지형만 보기 (물리 정지)
python3 view.py --terrain rock --static
```

**`--stage1` 을 주면 학습과 같은 조건**(경로 세트 고정, 랜덤화 OFF)으로 재생함.
안 주면 경로가 매번 새로 생성되고 노이즈가 켜져서 학습 조건과 달라짐.

### 마우스로 경로 찍고 달려보기

MuJoCo 뷰어는 마우스 콜백이 없음(`launch_passive` 는 `key_callback` 만 받음).
그래서 위에서 본 2D 지도에서 찍고 그 경로를 재생함.

```bash
cd sim/rl
python3 draw_path.py --terrain sand --out mypath.npy
#   좌클릭 추가 · 우클릭/z 취소 · c 전체삭제 · Enter 저장 · q 취소
#   빨간 영역이 주행 불가 (경사 상한 21도).  구간별 통과 가능 여부를 표시해 줌

# 기준선 (잔차 0)
python3 view.py --terrain sand --path mypath.npy --realtime

# 정책
python3 view.py --terrain sand --path mypath.npy --realtime \
        --model runs/s2/final --vecnorm runs/s2/vecnorm.pkl
```

특정 경로에서 실패가 재현되므로 전복·고착 원인을 눈으로 확인할 때 씀.
나중에 Nav2 가 붙으면 RViz 의 "2D Goal Pose" 가 이 역할을 대체함.

### 평가 · 진단

```bash
cd sim/rl
# 학습에 쓰지 않은 경로 세트(seed 4321)에서 같은 스폰·흙 조건으로 짝비교
python3 compare_policy.py --model runs/s2/final --vecnorm runs/s2/vecnorm.pkl \
        --episodes 30
# 위 비교가 통과하면 같은 경로에서 구동·센싱 노이즈와 흙 랜덤화도 확인
python3 compare_policy.py --model runs/s2/final --vecnorm runs/s2/vecnorm.pkl \
        --episodes 30 --randomize

python3 eval.py --arena --terrain sand --episodes 30 \
        --model runs/foo/final --vecnorm runs/foo/vecnorm.pkl
python3 probe_teacher.py --model runs/foo/final --vecnorm runs/foo/vecnorm.pkl
tensorboard --logdir runs/foo/tb
```

`compare_policy.py` 는 순수 IK(잔차 0)와 학습 정책을 같은 시드로 실행하고,
완주율뿐 아니라 횡오차 평균·p90, 기수오차, 슬립, 시간, 잔차 사용률을 비교한다.
`runs/s2` 의 학습 경로 기본 seed 는 1234 이므로 평가 기본값 4321 과 다르다.
경로추종 보조라면 완주율만이 아니라 횡오차 평균·p90도 기준선보다 좋아져야 한다.
차체 기수오차는 기록하되, 게걸음·후진 중에는 경로추종 실패로 간주하지 않는다.
RViz/Nav2 브리지는 기본값에서 잔차 0(순수 IK)이고, `rl:=true` 일 때만 정책을
적용한다. 이 재생 평가는 Nav2 MPPI 명령과는 다른 pure pursuit 명령으로 잰 결과다.
브리지는 `/plan` 경로와 필터링된 Nav2 명령으로 정책 관측의 경로 관련 필드를
매 주기 채운다. 따라서 RViz 시험은 정책 적용 확인용이며, 성능 수치는 별도로 재야 한다.
현재 3D 잔차의 크기는 `config.py` 의 `d_vx_max/d_vy_max/d_om_max` 로 정해진다.
옛 `--clip-steer-deg`/`--clip-duty` 옵션과 `set_clip()` 은 현재 3D 행동에 영향이 없다.

### Nav2 + RViz 로 클릭 주행

```bash
# 브리지 + Nav2 + RViz 를 한 번에 실행한다 (기존 별도 브리지는 먼저 종료).
cd /home/eomyunbeen/ksrc/ros && source /opt/ros/humble/setup.bash
python3 make_maps.py                      # 맵이 없으면 한 번
# 아래 둘 중 하나만 실행
ros2 launch nav2_ksrc.launch.py terrain:=sand           # 순수 IK
ros2 launch nav2_ksrc.launch.py terrain:=sand rl:=true  # s2 정책 보정
```

브리지는 `/cmd_vel`을 0.10초 시정수로 필터링해 `/cmd_vel_filtered`로 내고,
`/rl_action`(정규화 행동), `/rl_delta`(요청 보정), `/cmd_vel_applied`(포화 후 IK 입력)를
발행한다. 보정 적용은 `/cmd_vel_filtered`와 `/cmd_vel_applied`를 비교하면 된다.
`ros2 service call /set_rl std_srvs/srv/SetBool '{data: false}'`로 주행 중 끄고
`'{data: true}'`로 다시 켤 수 있다. 정책 모델은 `rl_model`, `rl_vecnorm` 런치
인자로 바꿀 수 있다. `cmd_filter_tau:=0`은 명령 필터 비교용이다.

RViz 상단 **"2D Goal Pose"** 로 지도를 클릭하면 로버가 그쪽으로 감.
회색이 주행가능, 검정이 통과 불가(경사 21° 초과)임.
목표 허용오차가 0.15m 이므로 로버 바로 옆을 찍으면 이미 도착한 것으로 판정될 수 있다.
이 런치는 시뮬 전용 ROS 도메인 77과 localhost 통신을 사용해 다른 ROS 노드의
`/odom`·`/map_server`와 섞이지 않게 한다. 진단 명령도 같은 도메인에서 실행한다:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
ros2 lifecycle get /bt_navigator         # active 여야 목표를 받는다
ros2 topic hz /odom                       # 브리지 위치가 계속 발행되는지 확인
```

전복 시 브리지는 구동을 멈추되 물리는 계속 진행하며, 자동으로 스폰으로 순간이동하지 않는다. 다시 시작하려면 별도 터미널에서
위 두 환경변수를 설정하고 `ros2 service call /reset_sim std_srvs/srv/Trigger '{}'` 를 실행한다. 같은 초기 스폰으로 돌아간다.
RViz 의 "2D Pose Estimate" 는 시뮬 스폰을 바꾸지 않는다. RViz 주행에서 RL 잔차는
런치 인자 `rl:=true`를 준 경우에만 적용된다.

컨트롤러는 **MPPI (`motion_model: Omni`)** 를 씀. `nav2_regulated_pure_pursuit_controller`
는 differential/ackermann 용이라 `(v, ω)` 만 내고 스워브의 게걸음(`vy`)을 버림.
실측으로 MPPI 가 `vy = 0.057` 을 명령하는 것을 확인함.

측위는 브리지가 `map→odom` 을 **항등으로** 발행함 (참값). 그래서 "Nav2 가 우리 로버에
맞게 도는가" 만 남고 측위 문제가 섞이지 않음. 실제 측위(D435 지형 ICP)가 되면 그
항등 변환만 갈아끼우면 됨.

> **로버를 점으로 취급함** (`robot_radius 0.01`). 정적 맵이 이미 풋프린트를 반영하기
> 때문임 — `drivable_mask` 가 0.16m 창으로 경사·단차를 판정하므로 "그 셀에 로버 중심을
> 놓으면 괜찮다" 는 뜻임. Nav2 에서 반경을 다시 팽창시키면 이중계상이고, 실측으로
> `inflation 0.18` 에서 자유공간이 70.4% → 28.5% 로 깎여 경로 계획이 실패했음.

### 대회맵 + 기존 조이스틱 텔레옵

첫 터미널에서 대회맵, MuJoCo 로버, RViz를 띄운다. 텔레옵 모드에는 Nav2
컨트롤러를 띄우지 않아 `/cmd_vel` 명령이 충돌하지 않는다.

```bash
cd ~/ksrc/ros
source /opt/ros/humble/setup.bash
ros2 launch nav2_ksrc.launch.py terrain:=sand mode:=teleop
```

두 번째 터미널에서 **기존** 마우스 조이스틱을 ROS 모드로 실행한다. 왼쪽 패드로
전후·게걸음, 오른쪽 패드 또는 Q/E로 회전하고 스페이스바로 멈춘다.
급격히 주행 방향을 바꾸면 텔레옵 브리지가 먼저 감속한다. 네 바퀴가 새 방향으로
조향된 뒤 구동하는 규칙은 학습·평가·RViz 주행에도 공통 적용된다.
`teleop_guard:=false`는 사전 감속만 끄고,
`drive_align_gate_deg:=0`은 공통 조향각 구동 게이트를 끈다.

```bash
cd ~/ksrc
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
python3 pi/teleop/teleop_joystick.py --ros --max-lin 0.25 --max-ang 0.8
```

`terrain:=rock`은 암석 대회맵으로 바꾼다. 이 모드의 RViz는 맵과 위치를 보여주는
화면이고 목표 주행 명령은 받지 않는다. 기존 `--port /dev/ttyACM0` 방식은 실기
Nucleo용 직렬 경로로 그대로 유지된다. 규사맵에서는 최초 스폰과 `/reset_sim`
위치를 3° 이하 평탄 구역으로 제한한다. [세 모드의 전체 CLI](CLI.md#6-대회맵-주행-rviz-목표-또는-텔레옵).

### 펌웨어

```bash
cd firmware/src2026-rover-firmware
./build.sh                # 빌드
./build.sh --flash        # 빌드 + ST-Link 플래시
python3 -m serial.tools.miniterm /dev/ttyACM0 115200
```
터미널 명령: `V vx vy ω`(기구학 주행) · `T a1..a4`(조향각 직접) ·
`SCAN`/`ID`/`ZERO`(서보 설정) · `LOG`/`ECHO`(화면).

### 실기 직렬 텔레옵

```bash
python3 pi/teleop/teleop_joystick.py --port /dev/ttyACM0
```
마우스 조이스틱 2개(병진/회전)로 몸체 속도를 만들어 아스키 `V` 로 보냄.

---

## 설계상 중요한 선택

**정책은 기구학을 대체하지 않고 보정함.** 행동은 차체속도 잔차 3개
`(Δvx, Δvy, Δω)`이며, 기본 명령에 더한 뒤 공유 스워브 IK로 보낸다.
`config.py`의 `d_vx_max/d_vy_max/d_om_max`가 잔차 권한을 정한다.
RViz 주행에서는 `rl:=false` 또는 `/set_rl`로 잔차를 0으로 만들 수 있다.

**쿨롱 마찰로는 고착을 재현할 수 없음.** MuJoCo 기본 접촉은 미끄러지면 견인력이
`μN` 으로 고정이라 시뮬 로버가 자기 구덩이를 못 팜. 실제 모래는
`슬립↑ → 침하↑ → 저항↑ → 슬립↑` 의 양의 되먹임이 있고 그게 곧 고착임.
`terramech.py` 가 전단곡선을 마찰계수로 주고 침하를 상태로 적분해 재현함.

**기본 보상은 2026년 4WIS 논문의 진행·정렬·횡오차·헤딩오차·매끄러움 구조를
스워브에 맞게 적용한다.** 진행의 15%는 경로 밖에도 남기고, 85%는 횡오차와
**실제 이동 방향**의 게이트를 통과해야 받는다. 이동 중 횡오차·방향오차에도
작은 벌점을 준다. `sigma_y=0.10m`, 이동방향 오차 폭은 0.35rad이며 2cm/s 이하의
이동 방향은 평가하지 않는다. 차체가 바라보는 방향은 주행 보상에 넣지 않으므로
같은 경로를 게걸음·후진으로 따라가도 같은 점수를 받는다. 잔차 크기와 행동
2차 차분 벌점, 완주·전복·고착·경계 항도 유지한다. `--reward position_only`는
이동 방향 항을 빼던 직전 보상, `--reward legacy`는 차체 헤딩을 쓰던 기존 보상
비교군이다. 최종 목표 기수는 Nav2의 도착 자세 제어가 맡는다.

**컨트롤러는 측위 추정치를, 보상·판정은 참값을 씀.** 보상에 참값을 쓰는 것은
가능하지만 정책 관측은 실기에서 얻을 수 있어야 한다. 현재 `e_y`와 `e_psi` 관측은
아직 참값이라 위의 관측 제약을 해결해야 한다. 잡음을 넣으면 그 값을 쓰는 모든
소비자를 함께 확인해야 함.

**측위 잡음은 저역통과로 줌**(τ=0.2 s). 50 Hz 백색잡음을 주면 조향이 지터를 따라
떨어 `scrub` 이 17배가 됨. 실제 측위는 EKF 가 오도메트리와 ICP(5~15 Hz)를 융합해
매끄러운 출력을 내므로 관측 잡음 대역이 제어 대역보다 훨씬 낮음.

**플래너 경사 상한은 현재 흙 모델의 견인력 가정에서 유도함.**
`max_slope_deg = 21.0°` 임. 규사 프리셋의 최대 마찰계수 0.55에서 침하 후
굴림·압밀 저항 약 0.14를 빼면 `atan(0.41) ≈ 22°`이며, 플래너에는 21°를 쓴다.
실제 규사의 마찰계수·침하 저항은 아직 측정하지 않았으므로 이는 실기 등판한계
측정값이 아니다. 30° 구간은 현재 Nav2 맵에서 통과 불가로 표시한다.

**학습 지형은 대회장 맵 고정 + 미세요철 교란임.** 거시 구조(램프·평지 배치)는
건축물이라 안 변하고, 변하는 것은 바퀴자국 규모(수 cm)임. 맵 하나에 고정하면 그
한 장의 요철까지 외우는데 모래는 대회 당일까지 움직임(다른 팀 주행·갈퀴질·습도).
암반 지형은 학습에서 빼고 일반화 검증용으로 남김.

**흙은 규사 하나로 고정하고 상태만 랜덤화함.** 대회장이 나무 프레임 위 규사 구조라
암석·경질토를 섞을 이유가 없음. 대신 다짐/습도(`μ` 0.60~0.82)와 두께(8~60 mm)를
난이도에 따라 흔듦.

**로버 형태는 랜덤화하지 않음.** 치수는 측정으로 확정되는 값이라 흔들 이유가 없음.
대신 실기에서 매 순간 변하고 측정으로 없앨 수 없는 것을 흔듦 — 배터리 전압(0.83~1.05),
엔코더 양자화·잡음, IMU 바이어스, 서보 지연·데드밴드, 제어 지연, 측위 오차.

---

## 현재 상태

| 항목 | 상태 |
|---|---|
| 펌웨어 | 플래시·터미널·기구학 주행·워치독 실기체 검증 완료. ±180° 접기는 코드 수정 완료, **미플래시** |
| 시뮬 물리 | 검증 13종 통과. 슬립모델이 고착 재현 확인 |
| 지형 | 실측 규사 경사지형(기복 450 mm, 최대 30°) 통계에 맞춘 절차 생성 확인 |
| 경로추종 | 큰 방향 전환 시 공통 조향 대기 적용 후 미학습 경로 30쌍: 순수 IK→구형 `s2` 완주 83.3→90.0%(RL만 2회, IK만 0회, McNemar p=0.50), 평균 횡오차 40.2→45.3mm. `s2`는 이 제어 변경 전 학습됐으므로 재학습과 별도 검증 필요 |
| 측위 | **미해결.** 엔코더+IMU 데드레커닝은 1.7 m 주행에 845 mm 오차(슬립 때문). D435 + 대회장 맵 ICP 가 기하적으로 가능함을 확인, 미구현 |
| Nav2 | **시뮬에서 동작 확인.** MPPI(Omni) 주행과 선택적 `rl:=true` 잔차 적용을 `/goal_pose` 종단 시험으로 확인. 정책 적용은 성능 개선 검증과 별개이며 실기는 측위 선행 |
| 텔레옵 + 대회맵 | 기존 pygame 조이스틱의 ROS 모드에서 `/cmd_vel`→MuJoCo 이동·패드 해제 후 정지, `/map` 발행 종단 시험 통과 |
| teacher-student | teacher 가 특권 잠재변수를 거의 안 씀(ablation 노이즈 수준) → 증류 보류 |

미확정 파라미터는 `config.py` 에 `[미정]` 으로 표시돼 있고 전부 도메인 랜덤화
범위로 덮여 있음.

### 알려진 제약

- **현재 규사 프리셋에서는 30° 램프를 못 오름.** `mu_max=0.55`, 기본 저항
  `c_r=0.05`라 침하 전부터 순견인 상한 0.50이 `tan(30°)=0.577`보다 작다.
  이는 미보정 흙 모델의 결과이며 실기 한계나 기구 변경 필요성을 입증하지 않는다.
- **`terramech.py` 가 `wheel_w` 를 쓰지 않음.** 바퀴 폭·접지압 효과를 정량화할 수 없음
- **다리 하우징이 바퀴축 아래 30.35 mm 에 고정.** 지상고 9.6 mm 를 결정하는 값이고
  바퀴를 규정 한계(반경 58.25 mm)까지 키워도 지상고는 27.9 mm 가 상한임
