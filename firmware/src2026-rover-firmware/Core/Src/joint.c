/**
 * @file    joint.c
 * @brief   관절(Joint) 각도 관리 모듈 구현
 *
 * 물리적 각도(도) ↔ 서보 위치값(0~4095) 변환을 처리하며,
 * on_target_angles_received() 콜백 함수를 통해 입력 소스(터미널/SPI)와
 * 모터 제어 로직을 완전히 분리합니다.
 *
 * 변환 공식:
 *   position = offset + dir × (theta × 4096.0 / 360.0)
 *   theta    = (position - offset) × 360.0 / 4096.0 × dir
 */

#include "joint.h"
#include "sts3215.h"
#include <stdio.h>
#include <stdlib.h>

/* ========================== 관절 설정 배열 ========================== */
/*
 * ★ 캘리브레이션 후 아래 값을 실제 로봇에 맞게 수정하세요.
 *
 * 캘리브레이션 방법:
 *   1. 터미널에서 "OFF" 입력 (토크 해제)
 *   2. 서보를 손으로 원하는 0도 위치(직진 방향)로 돌림
 *   3. "P" 입력하여 현재 위치값 읽기 (예: "Servo 1: pos=2100")
 *   4. 아래 offset 값을 해당 값으로 수정 (예: .offset = 2100)
 *   5. 재빌드 후 "ON" → "T 0 0" 입력하여 0도 위치 확인
 */
static JointConfig_t joint_config[SERVO_COUNT] = {
    /* idx 0: 서보 ID=1, 좌측 조향 (또는 원하는 위치) */
    {
        .id      = 1,
        .offset  = STS_POS_CENTER,  /* 0도 = 2048 (캘리브레이션 후 수정) */
        .dir     = +1,              /* 양수 각도 = 위치값 증가 방향 */
        .min_deg = -90.0f,
        .max_deg = +90.0f,
    },
    /* idx 1: 서보 ID=2, 우측 조향 (또는 원하는 위치) */
    {
        .id      = 2,
        .offset  = STS_POS_CENTER,  /* 0도 = 2048 (캘리브레이션 후 수정) */
        .dir     = +1,              /* 양수 각도 = 위치값 증가 방향 */
        .min_deg = -90.0f,
        .max_deg = +90.0f,
    },
    /* idx 2: 서보 ID=3 */
    {
        .id      = 3,
        .offset  = STS_POS_CENTER,
        .dir     = +1,
        .min_deg = -90.0f,
        .max_deg = +90.0f,
    },
    /* idx 3: 서보 ID=4 */
    {
        .id      = 4,
        .offset  = STS_POS_CENTER,
        .dir     = +1,
        .min_deg = -90.0f,
        .max_deg = +90.0f,
    },
};

/** 이동 속도 (step/s, 0=최대속도). "S 200" 명령으로 변경 가능 */
static uint16_t joint_goal_speed = 0;

/* ========================== 내부 헬퍼 ========================== */

/**
 * @brief  각도(도) → 서보 위치값(0~4095) 변환
 */
static uint16_t angle_to_position(const JointConfig_t *cfg, float deg)
{
    float pos_f = (float)cfg->offset + (float)cfg->dir * (deg * 4096.0f / 360.0f);

    /* 위치값 클램프 (0~4095) */
    if (pos_f < (float)STS_POS_MIN) pos_f = (float)STS_POS_MIN;
    if (pos_f > (float)STS_POS_MAX) pos_f = (float)STS_POS_MAX;

    return (uint16_t)(pos_f + 0.5f);  /* 반올림 */
}

/**
 * @brief  서보 위치값(0~4095) → 각도(도) 역변환
 */
static float position_to_angle(const JointConfig_t *cfg, uint16_t pos)
{
    return (float)cfg->dir * ((float)pos - (float)cfg->offset) * 360.0f / 4096.0f;
}

/* ========================== 공개 API 구현 ========================== */

void joint_init(void)
{
    printf("[Joint] 관절 모듈 초기화 (서보 %d개)...\r\n", SERVO_COUNT);

    for (int i = 0; i < SERVO_COUNT; i++)
    {
        uint8_t id = joint_config[i].id;

        /* 가속도 설정 (50 = 부드러운 가감속) */
        STS_Status ret = sts_set_acceleration(id, 50);
        if (ret != STS_OK)
        {
            printf("[Joint] 경고: 서보 ID=%d 가속도 설정 실패 (err=%d)\r\n", id, ret);
        }

        /* 토크 ON */
        ret = sts_set_torque(id, 1);
        if (ret != STS_OK)
        {
            printf("[Joint] 경고: 서보 ID=%d 토크 ON 실패 (err=%d)\r\n", id, ret);
        }
        else
        {
            printf("[Joint] 서보 ID=%d 토크 ON 완료\r\n", id);
        }
    }

    printf("[Joint] 관절 모듈 초기화 완료!\r\n");
}

void on_target_angles_received(float *theta, int count)
{
    /*
     * ★ 핵심 콜백 함수 ★
     *
     * 현재: PC 터미널 파서가 "T 10.5 -20" 명령 수신 시 호출
     * 나중: SPI 수신 콜백이 라즈베리파이로부터 각도 수신 시 호출
     *
     * 동작:
     *   1. 각도 → 위치값 변환 (범위 클램프 포함)
     *   2. SYNC_WRITE로 모든 서보를 한 사이클에 동시 이동
     */

    if (count > SERVO_COUNT) count = SERVO_COUNT;

    uint8_t  ids[SERVO_COUNT];
    uint16_t positions[SERVO_COUNT];
    uint16_t speeds[SERVO_COUNT];

    for (int i = 0; i < count; i++)
    {
        float deg = theta[i];
        const JointConfig_t *cfg = &joint_config[i];

        /* 각도 클램프 (허용 범위 초과 시 경고) */
        if (deg < cfg->min_deg)
        {
            printf("[Joint] 경고: 서보 %d 각도 %.1f° → 클램프 %.1f°\r\n",
                   cfg->id, deg, cfg->min_deg);
            deg = cfg->min_deg;
        }
        if (deg > cfg->max_deg)
        {
            printf("[Joint] 경고: 서보 %d 각도 %.1f° → 클램프 %.1f°\r\n",
                   cfg->id, deg, cfg->max_deg);
            deg = cfg->max_deg;
        }

        ids[i]       = cfg->id;
        positions[i] = angle_to_position(cfg, deg);
        speeds[i]    = joint_goal_speed;
    }

    /* SYNC_WRITE로 한 사이클에 모든 서보 동시 이동 */
    sts_sync_write_position(ids, positions, speeds, (uint8_t)count);
}

float joint_get_angle(int idx)
{
    if (idx < 0 || idx >= SERVO_COUNT) return -999.0f;

    int16_t pos = sts_read_position(joint_config[idx].id);
    if (pos < 0)
    {
        printf("[Joint] 에러: 서보 ID=%d 위치 읽기 실패\r\n", joint_config[idx].id);
        return -999.0f;
    }

    return position_to_angle(&joint_config[idx], (uint16_t)pos);
}

void joint_set_speed(uint16_t speed)
{
    joint_goal_speed = speed;

    /* 각 서보에 개별적으로 Goal Speed 설정 */
    for (int i = 0; i < SERVO_COUNT; i++)
    {
        uint8_t data[2];
        data[0] = (uint8_t)(speed & 0xFF);
        data[1] = (uint8_t)((speed >> 8) & 0xFF);
        sts_write(joint_config[i].id, STS_REG_GOAL_SPEED, data, 2);
    }

    printf("[Joint] 이동 속도 설정: %u step/s\r\n", speed);
}

void joint_set_torque_all(uint8_t on)
{
    for (int i = 0; i < SERVO_COUNT; i++)
    {
        sts_set_torque(joint_config[i].id, on);
    }
    printf("[Joint] 전체 서보 토크 %s\r\n", on ? "ON" : "OFF");
}
