# 10-17 밤 ~ 10-18 실물 주행 기록

Pi `run_policy.py` 주행 로그(npz)와 ORB-SLAM3 VO 실행 기록. 비닐 깐 경사(약 10–13°)와 평지.

## 주행 로그 (npz, 50 Hz)

| 파일 | 시각 | 정책 | 명령 |
|---|---|---|---|
| `v1_vo_1.npz` | 10-17 23:33 | student v1 (`res8_mlp_v1_rc4000`, 30°), VO 기록만 | 고정 0.15 m/s 직진 10 s |
| `v3_vo_1.npz` | 10-17 23:38 | student v3 (`res8_mlp_v3vo_rc4000`, 30°), VO 사용 | 고정 0.15 m/s 직진 10 s |
| `v3_tele_1.npz` | 10-17 23:47 | student v3, VO 사용 | 텔레옵 |
| `tele/ik_1.npz` | 10-18 11:59 | IK (잔차 끔) | 텔레옵, 상한 0.15 m/s |
| `tele/stv3_1.npz` | 10-18 12:08 | student v3, VO 사용 | 텔레옵, 상한 0.15 m/s |

주요 열: `cmd`(보낸 명령), `tlm_cmd`(Nucleo 적용 명령), `wheel_mps`, `enc_ticks`, `duty`, `steer_meas`/`steer_cmd`/`steer_load`/`steer_temp`,
`gyro_dps`/`acc_g`/`imu_raw`, `vo`(vx, vy, 나이 s), `slip`(바퀴별 헛돎 추정, VO 기준), `obs`/`action`, `flags`, `t_ms`/`host_ns`, `args`(실행 옵션).
`enc_ticks`·`imu_raw`·`slip`·`args` 는 10-18 11:35 이후 로그(`tele/`)에만 있다. 바퀴·조향 배열 순서는 FL, FR, RL, RR.

## ORB (`orb/vo1018_XX/`)

`base_map_<n>.tum` = 로버 궤적 (시각 x y z qx qy qz qw, 맵 번호마다 따로), `map_<n>.ply` = 지도 점군, `tracking.json` = 추적 통계.
`atlas.osa` (70–160 MB) 는 GitHub 용량 제한으로 뺐다 (노트북 `~/ksrc/sysid_logs/1018/orb/` 에 있음).
`vo1018_05`, `_14`, `_15` 는 시작 직후 끈 빈 실행. 카메라 영상·카메라 IMU 는 기록하지 않았다 (장착 카메라는 D435, IMU 없음).
`orb_probe/` = ORB 준비용 카메라 입력 확인 결과.
