# 10-17 (또는 18) 실측 절차 (모래 연습장)

목적: (1) 실물 조향이 정책 명령처럼 떨리는지, (2) 모래 평지·경사에서 IK 대비 RL 이득이 있는지, (3) 연습장 모래의 유효 견인 계수.
정책은 **student v1** `res8_mlp_v1_rc4000` (노트북 `sim/rl/runs/isaac/2026-10-15_13-06-32_student_mlp_v1_rc4000/model_1000.pt` 를 내보낸 것) 하나로 한다.
`smooth1015` (떨림 줄이는 학습) 는 S0 결과가 나쁠 때 쓰는 예비다.
준비·`RUN` 정의는 `MEASURE_1016.md` 와 같다. 로그 폴더만 `mkdir -p ~/ksrc/logs/1017`.

`RUN` = `~/venv/bin/python pi/policy/run_policy.py --policy pi/policy/res8_mlp_v1_rc4000 --rt 50 --cpu 3 --gc-off`

## 측정 순서 (우선순위: S0 → S1 → S3 → S2, 시간 없으면 S4 생략)

| # | 무엇 | 환경·방법 | 명령 | 보는 것 |
|---|---|---|---|---|
| S0 | **조향 떨림 실물 확인** | 같은 명령을 ① 바퀴 띄움 ② 땅(바닥 또는 모래 평지)에 놓고, 잔차 켬. 비교용으로 ③ 땅 + 잔차 끔 | `$RUN --cmd 0.15 0 0 --seconds 8 --log logs/1017/jit_air.npz` · 같은 명령 땅에서 `jit_ground.npz` · `--residual-off` 로 `jit_ground_ik.npz` | `steer_cmd` 흔들림 vs `steer_meas` 흔들림 (스텝당 변화·std), `steer_load` (부하), 서보 소리·발열 |
| S1 | **모래 평지 직진 A/B** | 평지 1.5 m, IK 와 RL 번갈아 각 3회 | `$RUN --residual-off --cmd 0.15 0 0 --seconds 10 --log logs/1017/flat_ik_1.npz` · 잔차 켜고 `flat_rl_1.npz` (…\_2, \_3) | 실제 이동 거리(영상+테이프), 방향 틀어짐, 헛돎(`wheel_mps` vs 실제 속도), 침하 흔적 |
| S2 | **연습장 경사 A/B** | 같은 출발점에서 IK, RL 번갈아 각 3회. 경사각을 휴대폰 경사계로 기록 | `$RUN --residual-off --cmd 0.15 0 0 --seconds 15 --log logs/1017/slope_ik_1.npz` · 잔차 켜고 `slope_rl_1.npz` | 끝까지 올랐는지, 어디서 멈췄는지(경사→평지 전환부 포함), 헛돎·요 틀어짐 |
| S3 | **유효 견인 계수** | IK 로 천천히(0.10 m/s) 경사를 오르며 헛돌기 시작하는 지점의 경사각 θ | `$RUN --residual-off --cmd 0.10 0 0 --seconds 15 --log logs/1017/mu_ik.npz` | μ_eff ≈ tan θ. **≤0.3 이면 RL 이 의미 있는 영역** (시뮬: μ≤0.3 큰 이득, ≥0.5 이득 없음) |
| S4 | VO 잡음·지연 (선택) | S1 직진 중 ORB 동시 기록 | ORB 띄운 채 S1 RL 1회 | VO 속도 잡음·지연 (student v3 용) |

- 모든 주행은 휴대폰으로 옆에서 촬영(테이프 기준).
- 이상하면 즉시 멈춤: 조이스틱/스페이스.

## 판정

**S0 (떨림)**
- 땅에서 `steer_meas` 와 `steer_load` 가 조용하면 (명령은 흔들려도 실제 각도는 하중·서보 응답으로 걸러짐) → **v1 그대로 사용**.
- 땅에서도 실제 각도가 흔들리거나 부하가 계속 출렁이고 서보가 웅웅거리면 → **smooth1015 증류본으로 교체** (같은 SET, 1001 iter, 몇 시간).
- 참고: 시뮬 v1 은 주행 중 조향 스텝당 ~0.3° (std ~1°). STS3215 한 칸 0.088° 라 데드밴드보다 크다.

**S1·S2 (A/B)**
- RL 이 IK 보다 더 멀리/끝까지 가고 방향 틀어짐이 작으면 이득. 3회 중 다수결.
- 시뮬 MuJoCo 모래(μ0.4) 에서는 IK 초반 실패, RL 은 경사→평지 전환부에서만 실패했다. 실물에서 같은 지점에서 멈추는지 본다.

## 끝나고 (노트북)
```bash
scp pi@raspberrypi.local:~/ksrc/logs/1017/*.npz ~/ksrc/sysid_logs/1017/
```
npz 경로와 경사각·영상 메모를 Claude 에게 주면 S0 떨림 비교, A/B 지표, μ_eff 를 정리한다.
결과로 정하는 것: v1 유지 / smooth1015 교체, MuJoCo sand 계수(μ_max 등) 교체, teacher 재학습 방향(추종 중심 보상, 침하 모델).
