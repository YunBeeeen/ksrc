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
    │                 │   (계획) Nav2      경로 → cmd_vel
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
| `reward.py` | 보상 가중치와 프리셋 (speed / balanced / safe) |
| `env.py` | Gymnasium 환경 |
| `train.py` | PPO 학습 (stable-baselines3) |
| `eval.py` | 대회장 지형 평가. 고전 제어 기준선 4종과 비교 |
| `traction.py` | 고전 제어 기준선 (경사스케줄 듀티 제한 / 엔코더 TCS) |
| `nets.py` | teacher-student 신경망 (특권 인코더 / 적응 모듈) |
| `distill.py` | teacher → student 증류 (DAgger + 잠재변수 회귀) |
| `probe_teacher.py` | teacher 가 특권정보를 실제로 쓰는지 진단 |
| `view.py` | MuJoCo 뷰어 (정책 재생) |
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
python3 train.py --steps 4000000 --envs 16 --teacher --terrain sand --out runs/foo
```
`--terrain sand` 는 대회장 맵 고정 + 미세요철 교란임. `proc` 은 절차 생성임.
`--teacher` 는 특권 관측을 붙임. 로그는 `runs/foo/tb` 로 감.

### 재생 (play)

```bash
cd sim/rl
python3 view.py                                          # 순수 기구학 주행
python3 view.py --model runs/foo/final --vecnorm runs/foo/vecnorm.pkl
python3 view.py --terrain sand --d 1.0 --realtime         # 대회장 맵, 최고난이도
python3 view.py --terrain rock --static                   # 정지 상태로 지형만 보기
```
`vecnorm.pkl` 없이 정책만 넘기면 관측 스케일이 어긋나 결과가 무의미해짐.

### 평가 · 진단

```bash
cd sim/rl
python3 eval.py --arena --terrain sand --episodes 30 \
        --model runs/foo/final --vecnorm runs/foo/vecnorm.pkl
python3 probe_teacher.py --model runs/foo/final --vecnorm runs/foo/vecnorm.pkl
tensorboard --logdir runs/foo/tb
```

### 펌웨어

```bash
cd firmware/src2026-rover-firmware
./build.sh                # 빌드
./build.sh --flash        # 빌드 + ST-Link 플래시
python3 -m serial.tools.miniterm /dev/ttyACM0 115200
```
터미널 명령: `V vx vy ω`(기구학 주행) · `T a1..a4`(조향각 직접) ·
`SCAN`/`ID`/`ZERO`(서보 설정) · `LOG`/`ECHO`(화면).

### 텔레옵

```bash
python3 pi/teleop/teleop_joystick.py --port /dev/ttyACM0
```
마우스 조이스틱 2개(병진/회전)로 몸체 속도를 만들어 아스키 `V` 로 보냄.

---

## 설계상 중요한 선택

**정책은 기구학을 대체하지 않고 보정함.** 액션 8개는 바퀴 명령 자체가 아니라
역기구학 해에 더해지는 제한된 잔차임(`|Δ조향| ≤ 10°`, `|Δduty| ≤ 0.6`). 잔차를
0 으로 클램프하면 순수 기구학 주행으로 즉시 되돌아가므로 현장에서 플래그 하나로
정책을 끌 수 있음.

**쿨롱 마찰로는 고착을 재현할 수 없음.** MuJoCo 기본 접촉은 미끄러지면 견인력이
`μN` 으로 고정이라 시뮬 로버가 자기 구덩이를 못 팜. 실제 모래는
`슬립↑ → 침하↑ → 저항↑ → 슬립↑` 의 양의 되먹임이 있고 그게 곧 고착임.
`terramech.py` 가 전단곡선을 마찰계수로 주고 침하를 상태로 적분해 재현함.

**보상은 경로추종 3항으로 분해함.** `prog`(경로 접선방향 진행) ·
`xt`(횡이탈) · `yaw`(요속도 추종). 속도추종을 경로좌표로 분해한 형태라
lumped `|v − cmd|` 보다 해석이 명확하고 기울기가 죽지 않음.

**컨트롤러는 측위 추정치를, 보상·판정은 참값을 씀.** 관측만 센서로 얻을 수 있으면
되고 보상은 특권정보를 써도 됨. 부분 적용이 가장 위험함 — 잡음을 넣으면 그 값을 쓰는
모든 소비자를 확인해야 함.

**측위 잡음은 저역통과로 줌**(τ=0.2 s). 50 Hz 백색잡음을 주면 조향이 지터를 따라
떨어 `scrub` 이 17배가 됨. 실제 측위는 EKF 가 오도메트리와 ICP(5~15 Hz)를 융합해
매끄러운 출력을 내므로 관측 잡음 대역이 제어 대역보다 훨씬 낮음.

**플래너 경사 상한은 로버 견인력에서 유도함.** `max_slope_deg = 21.0°` 임.
가용 마찰 0.53 − 굴림/침하 저항 0.14 = 0.39 → `atan(0.39) = 21.3°`. 마찰만 보면
32.5° 지만 저항이 26% 를 먹음. 근거 없이 35° 로 두면 플래너가 로버가 못 오르는
30° 램프로 경로를 냄.

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
| 경로추종 | pure pursuit 미러링 동작. 규사 맵 성공률 순수 IK 80.3% / RL 85.7% |
| 측위 | **미해결.** 엔코더+IMU 데드레커닝은 1.7 m 주행에 845 mm 오차(슬립 때문). D435 + 대회장 맵 ICP 가 기하적으로 가능함을 확인, 미구현 |
| Nav2 | 미착수. 측위 선행 필요 |
| teacher-student | teacher 가 특권 잠재변수를 거의 안 씀(ablation 노이즈 수준) → 증류 보류 |

미확정 파라미터는 `config.py` 에 `[미정]` 으로 표시돼 있고 전부 도메인 랜덤화
범위로 덮여 있음.

### 알려진 제약

- **30° 램프는 물리적으로 못 오름** (등판한계 21°). 대회 코스에 30° 필수 통과 구간이
  있으면 기구 변경(바퀴 폭·러그) 없이는 못 넘음
- **`terramech.py` 가 `wheel_w` 를 쓰지 않음.** 바퀴 폭·접지압 효과를 정량화할 수 없음
- **다리 하우징이 바퀴축 아래 30.35 mm 에 고정.** 지상고 9.6 mm 를 결정하는 값이고
  바퀴를 규정 한계(반경 58.25 mm)까지 키워도 지상고는 27.9 mm 가 상한임
