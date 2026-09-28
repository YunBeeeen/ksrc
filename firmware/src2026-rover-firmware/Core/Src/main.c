/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "motor.h"
#include "encoder.h"
#include "sts3215.h"
#include "joint.h"
#include "swerve_kinematics.h"
#include "mdd3a_driver.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
typedef struct {
    int32_t  last_raw_count;  /**< Previous raw encoder tick */
    int64_t  total_count;     /**< Cumulative tick count (no rollover) */
    float    rpm;             /**< Wheel RPM */
    float    rad_per_sec;     /**< Angular velocity [rad/s] */
    float    linear_vel;      /**< Linear velocity [m/s] = w * R */
} WheelState_t;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* Motor & Encoder Specifications */
#define GEAR_RATIO          131.0f             /**< 131:1 Gearbox reduction (SPG30E-GR131) */
/* 조향 가동범위 [rad].  서보(STS3215)는 다회전 가능하고 sts3215_protocol.c 가
 * 이음매를 처리하므로 **제약은 배선 꼬임뿐**이다 (슬립링 없음).
 * 실제 배선 여유를 재서 확정할 것. 최소 PI/2 여야 모든 방향을 표현할 수 있다. */
#define SWERVE_STEER_LIMIT_RAD  3.14159265f   /* +-180deg */

#define ENCODER_PPR         16.0f              /**< 16 pulses per motor rev (Cytron wire-ID chart: "16 x GR = resolution") */
#define ENCODER_CPR         (ENCODER_PPR * 4.0f) /**< 64 ticks in 4x mode (TIM_ENCODERMODE_TI12) */
#define TICKS_PER_REV       (ENCODER_CPR * GEAR_RATIO) /**< 8384.0 ticks per wheel rev */

/* Wheel radius R (Diameter 96mm -> Radius 48mm = 0.048m) */
/* 조립 STL 실측값.  이전 0.048f (지름 96mm) 는 설계 초기 추정치였고 실측과
 * 20.2% 차이가 났다.  개루프 duty 에서는 "보고되는 속도가 틀리다" 로 끝나지만,
 * 바퀴 속도 PI 폐루프를 넣으면 PI 가 틀린 측정값을 목표에 맞추려고 duty 를
 * 계속 왜곡하므로 치명적이다.  sim/rl/config.py 의 wheel_r 과 같은 값이어야 한다. */
#define WHEEL_RADIUS_M      0.03995f           /**< Radius R = 39.95mm (Diameter 79.9mm), STL 실측 */
#define PI                  3.1415926535f

/* --- Swerve module geometry & actuator mapping (bench-test placeholder) ---
 * x=forward, y=left, matches firmware/common/tests/test_swerve_kinematics.c
 * and sim/mujoco/ksrc_rover.urdf. Real offsets pending CAD; module<->MOTOR/
 * servo-ID mapping is also unconfirmed against actual wiring -- both
 * placeholders until bench-verified. Servo IDs use joint.c's array index
 * (0-based; joint_config[] maps index->bus ID internally). */
/* 조립 STL 실측값 (sim/rl/config.py: axle_x=0.0918, track=0.2367 -> track/2=0.11835).
 * 이전 (±0.12, ±0.12) 는 추정치였다.  순수 병진(vx, vy)에서는 모듈 위치가 결과에
 * 영향을 주지 않지만 omega != 0 이면 v_i = v_B + omega x r_i 로 오차가 남는다:
 *     제자리회전 w=0.8  -> 조향각 최대 7.20도
 *     선회 vx=0.2 w=0.5 -> 최대 5.15도
 *     실주행 중앙값     -> 0.90도
 * 시뮬과 반드시 같은 값이어야 한다 (같은 C 코드를 공유하는 의미가 없어진다). */
static const swerve_module_pos_t SWERVE_MODULES[SWERVE_NUM_MODULES] = {
    { 0.0918f,  0.11835f },  /* module 0: FL */
    { 0.0918f, -0.11835f },  /* module 1: FR */
    {-0.0918f,  0.11835f },  /* module 2: RL */
    {-0.0918f, -0.11835f },  /* module 3: RR */
};
static const MotorID SWERVE_MODULE_MOTOR[SWERVE_NUM_MODULES] = { MOTOR1, MOTOR2, MOTOR3, MOTOR4 };

/* Placeholder until bench-measured (free-spin m/s at 100% duty, see
 * mdd3a_driver.h's open item). Wrong magnitude only scales wheel speed
 * for a given vx/vy/omega, doesn't change direction/shape. */
/* 듀티 100% 에서의 바퀴 선속도.  시뮬은 mot_w_noload * wheel_r = 7.96 * 0.03995
 * = 0.318 m/s 로 계산한다.  이 값이 어긋나면 IK 의 desaturation 기준이 달라져
 * 같은 cmd_vel 에 대해 시뮬과 실기의 스로틀 스케일이 맞지 않는다.
 * 실기 벤치테스트로 확정할 값이고, 지금은 시뮬과 일치시킨다. */
#define SWERVE_MAX_WHEEL_MPS  0.318f

/* STS3215 레지스터 중 sts3215.h 에 없는 것 (공식 메모리표 기준) */
#define STS_REG_ID_ADDR    0x05  /**< 서보 버스 ID (EPROM, 0~253) */
#define STS_REG_LOCK_FLAG  0x37  /**< 잠금 플래그: 0 = EPROM 쓴 값이 전원 차단 후에도 유지 (기본값) */
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim3;
TIM_HandleTypeDef htim4;
TIM_HandleTypeDef htim5;
TIM_HandleTypeDef htim8;
TIM_HandleTypeDef htim12;

/* USER CODE BEGIN PV */

/* === 터미널 명령 수신 버퍼 (USART2 RX 인터럽트) === */
#define CMD_BUF_SIZE  64
static char     cmd_buf[CMD_BUF_SIZE];  /**< ISR 이 문자를 모으는 버퍼 */
static char     cmd_line[CMD_BUF_SIZE]; /**< 완성된 줄 (메인 루프가 읽음) */
static uint8_t  cmd_idx = 0;            /**< 현재 버퍼 인덱스 */
static volatile uint8_t cmd_ready = 0;  /**< 1이면 완성된 명령이 있음 */

/** 주기적 엔코더/속도 로그 출력 여부. 조이스틱이나 SCAN/ZERO 같은 대화형
 *  작업 중에는 10Hz x 5줄이 터미널을 덮어버려서 끌 수 있어야 한다. */
static uint8_t  s_log_enabled = 1;

/** 받은 문자를 되돌려 보낼지. 안 그러면 터미널에서 타이핑이 안 보인다.
 *  ISR 에서 송신 대기를 하므로 조이스틱이 연속 전송할 때는 수신 오버런을
 *  유발할 수 있어, 텔레옵이 시작되면 자동으로 꺼진다. */
static uint8_t  s_echo_enabled = 1;

/* === 텔레옵 상태 === */
/** 명령이 이 시간 이상 끊기면 모터 정지 (조이스틱 종료/USB 분리 대비) */
#define TELEOP_TIMEOUT_MS  500
static uint8_t  s_teleop_active = 0;      /**< V 명령을 한 번이라도 받으면 1 */
static uint32_t s_last_v_cmd_ms = 0;      /**< 마지막 V 명령 시각 */
static uint8_t  s_teleop_timed_out = 0;   /**< 워치독이 이미 정지시켰으면 1 */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_TIM1_Init(void);
static void MX_TIM2_Init(void);
static void MX_TIM3_Init(void);
static void MX_TIM4_Init(void);
static void MX_TIM5_Init(void);
static void MX_TIM8_Init(void);
static void MX_TIM12_Init(void);
/* USER CODE BEGIN PFP */
static void MX_USART2_Init(void);
static void MX_USART3_Init(void);
static void servo_boot_sequence(void);
static void process_command(const char *cmd);
static void print_help(void);
static void Swerve_Drive(float vx, float vy, float omega);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
static void MX_USART2_Init(void)
{
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_USART2_CLK_ENABLE();

  /* PA2: USART2_TX, PA3: USART2_RX (Connected to ST-LINK VCP) */
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  GPIO_InitStruct.Pin = GPIO_PIN_2 | GPIO_PIN_3;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF7_USART2;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /* Configure Baud rate 115200 (Auto-calculated based on APB1 clock) */
  uint32_t pclk1 = HAL_RCC_GetPCLK1Freq();
  USART2->BRR = (pclk1 + (115200 / 2)) / 115200;
  USART2->CR1 = USART_CR1_TE | USART_CR1_RE | USART_CR1_UE;
}

int __io_putchar(int ch)
{
  while (!(USART2->SR & USART_SR_TXE));
  USART2->DR = (uint8_t)ch;
  return ch;
}

int _write(int file, char *ptr, int len)
{
  (void)file;
  for (int i = 0; i < len; i++)
  {
    if (*ptr == '\n')
    {
      __io_putchar('\r');
    }
    __io_putchar(*ptr++);
  }
  return len;
}

/* Newlib-nano syscall stubs (required when using printf with nano.specs) */
int _read(int file, char *ptr, int len)   { (void)file; (void)ptr; (void)len; return 0; }
int _close(int file)                      { (void)file; return -1; }
int _isatty(int file)                     { (void)file; return 1; }
int _lseek(int file, int ptr, int dir)    { (void)file; (void)ptr; (void)dir; return 0; }
#include <sys/stat.h>
int _fstat(int file, struct stat *st)     { (void)file; st->st_mode = S_IFCHR; return 0; }
int _getpid(void)                         { return 1; }
int _kill(int pid, int sig)               { (void)pid; (void)sig; return -1; }

/* ================================================================== */
/*        USART3 초기화 — STS3215 서보 통신 (PB10/PB11, 1Mbps)        */
/* ================================================================== */
static void MX_USART3_Init(void)
{
  /* GPIO 클럭 및 USART3 클럭 활성화 */
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_USART3_CLK_ENABLE();

  /* PB10: USART3_TX (Single-Wire Half-Duplex 모드, 이 핀 하나로 송수신 동시 처리) */
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  GPIO_InitStruct.Pin = GPIO_PIN_10;                 /* PB10 하나만 사용 */
  GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;            /* ★ 반드시 Open-Drain으로 설정 */
  GPIO_InitStruct.Pull = GPIO_PULLUP;                /* 내부 풀업 활성화 */
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF7_USART3;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /* Configure Baud rate 1000000 (1Mbps, Auto-calculated based on APB1 clock) */
  uint32_t pclk1 = HAL_RCC_GetPCLK1Freq();
  USART3->BRR = (pclk1 + (1000000 / 2)) / 1000000;
  
  /* CR3: HDSEL (Half-Duplex Selection) 비트 세트 */
  USART3->CR3 |= USART_CR3_HDSEL;
  
  /* CR1: 송수신 활성화 및 USART 활성화 */
  USART3->CR1 = USART_CR1_TE | USART_CR1_RE | USART_CR1_UE;
}

/* ================================================================== */
/*        USART2 RX 인터럽트 초기화 및 바이트 처리 루틴               */
/* ================================================================== */
static void MX_USART2_RxInt_Init(void)
{
  /* USART2 RXNE 인터럽트 활성화 */
  USART2->CR1 |= USART_CR1_RXNEIE;
  HAL_NVIC_SetPriority(USART2_IRQn, 5, 0);
  HAL_NVIC_EnableIRQ(USART2_IRQn);
}

void USART2_Process_Rx_Byte(uint8_t byte)
{
  char ch = (char)byte;

  /* 엔터(\r 또는 \n) → 명령 완성
   *
   * 이전 구현은 cmd_ready 가 1인 동안 들어오는 문자를 전부 버렸다. 메인
   * 루프가 직전 명령을 처리하는 사이에 도착한 줄이 통째로 사라져서,
   * 조이스틱처럼 20Hz 로 계속 보내면 명령이 끊기고 조각만 남았다.
   * 완성된 줄은 별도 버퍼로 옮기고 수집은 계속한다 — 아직 안 읽힌 줄이
   * 있으면 새 줄이 덮어쓴다(최신 우선). 오래된 속도 명령을 실행하는 쪽이
   * 최신 명령을 놓치는 것보다 위험하므로 이 방향이 맞다. */
  if (ch == '\r' || ch == '\n')
  {
    /* 빈 줄(CRLF 의 두 번째 문자 등)은 무시 — 에코도, 명령 확정도 하지 않는다 */
    if (cmd_idx > 0)
    {
      if (s_echo_enabled) { __io_putchar('\r'); __io_putchar('\n'); }
      cmd_buf[cmd_idx] = '\0';
      memcpy(cmd_line, cmd_buf, (size_t)cmd_idx + 1);
      cmd_ready = 1;
      cmd_idx = 0;
    }
  }
  else if (ch == '\b' || ch == 0x7F)  /* Backspace / DEL — 오타 수정 */
  {
    if (cmd_idx > 0)
    {
      cmd_idx--;
      if (s_echo_enabled) { __io_putchar('\b'); __io_putchar(' '); __io_putchar('\b'); }
    }
  }
  else if (cmd_idx < CMD_BUF_SIZE - 1)
  {
    cmd_buf[cmd_idx++] = ch;
    if (s_echo_enabled) __io_putchar(ch);
  }
}

/* ================================================================== */
/*              서보 부팅 시퀀스 — PING → 토크 ON → 준비              */
/* ================================================================== */
static void servo_boot_sequence(void)
{
  printf("\r\n--------------------------------------------------\r\n");
  printf("       STS3215 Servo Bus Initialization            \r\n");
  printf("--------------------------------------------------\r\n");
  printf("USART3: PB10(TX)/PB11(RX), 1,000,000 bps\r\n");
  printf("Servo Count: %d\r\n\r\n", SERVO_COUNT);

  /* 1. 각 서보 PING */
  uint8_t servo_ids[] = {1, 2, 3, 4};  /* SERVO_COUNT=4 기준 */
  uint8_t all_ok = 1;

  for (int i = 0; i < SERVO_COUNT; i++)
  {
    STS_Status ret = sts_ping(servo_ids[i]);
    if (ret == STS_OK)
    {
      printf("  Servo ID=%d: OK (응답 정상)\r\n", servo_ids[i]);
    }
    else
    {
      printf("  Servo ID=%d: FAIL (err=%d)\r\n", servo_ids[i], ret);
      all_ok = 0;
    }
  }

  if (!all_ok)
  {
    printf("\r\n[경고] 일부 서보 응답 없음. 배선/전원/ID 확인 필요.\r\n");
    printf("  점검: TX/RX 교차, GND 공통, 1Mbps 보레이트, 어댑터 모드\r\n");
  }

  /* 2. 관절 모듈 초기화 (가속도 설정 + 토크 ON) */
  joint_init();

  printf("\r\n>>> Servo System Ready! <<<\r\n");
  printf("터미널 명령어: T / P / S / ON / OFF / PING / HELP\r\n");
  printf("--------------------------------------------------\r\n\r\n");
}

/* ================================================================== */
/*      스워브 기구학 연동 — vx,vy,omega -> 조향 서보 + 구동 모터      */
/* ================================================================== */

/* last_angle_rad per module persists across calls (needed for the
 * <=90deg shortest-path steering optimization in swerve_ik_compute). */
static swerve_module_state_t s_swerve_state[SWERVE_NUM_MODULES];

static void Swerve_Drive(float vx, float vy, float omega)
{
  swerve_module_cmd_t cmd[SWERVE_NUM_MODULES];
  /* 조향 가동범위 안으로 접기 -- 시뮬과 **같은 C 함수**를 쓴다.
   * 예전에는 여기서 ±90도 고정 창에 접었는데, 연속각이 경계를 지날 때마다
   * 179도 강제 반전이 생겨 서보가 0.6초씩 엉뚱한 곳을 봤다 (실측: 에피소드의
   * 24%). swerve_fold_to_limit 은 직전 명령각에 가장 가까운 등가각을 고른다.
   * SWERVE_STEER_LIMIT_RAD 는 **배선이 견디는 범위**다 (슬립링 없음). */
  swerve_ik_compute(vx, vy, omega, SWERVE_MODULES, SWERVE_MAX_WHEEL_MPS,
                     s_swerve_state, cmd);
  /* unwind 는 끈다(0).  구동 중에는 바퀴가 쉬는 순간이 없어 한 번도 안 걸렸고,
   * ±180도 에서는 이음매 손실이 이미 에피소드의 3% 수준이다.  웨이포인트에서
   * 정지하는 구간이 생기면 그때 켜면 된다. */
  swerve_fold_to_limit(SWERVE_STEER_LIMIT_RAD, 0.0f, 0.0f, s_swerve_state, cmd);

  /* 조이스틱은 20~50Hz 로 계속 명령을 보낸다. 호출마다 모듈당 1줄씩 찍으면
   * 115200 baud(≈11.5KB/s)를 그대로 포화시켜 루프가 밀리므로 5Hz 로 제한. */
  static uint32_t last_print_ms = 0;
  uint32_t print_now = HAL_GetTick();
  int do_print = (print_now - last_print_ms >= 200);
  if (do_print) last_print_ms = print_now;

  float angles_deg[SWERVE_NUM_MODULES];
  for (int i = 0; i < SWERVE_NUM_MODULES; i++)
  {
    float deg = cmd[i].angle_rad * 180.0f / PI;
    float speed_mps = cmd[i].speed_mps;

    angles_deg[i] = deg;

    float norm = mdd3a_normalize_speed(speed_mps, SWERVE_MAX_WHEEL_MPS);
    int16_t duty = (int16_t)(norm * (float)PWM_MAX);
    Motor_SetSpeed(SWERVE_MODULE_MOTOR[i], duty);

    if (do_print)
    {
      printf("  module%d: angle=%.1f deg  speed=%.2f m/s  duty=%d\r\n",
             i, angles_deg[i], speed_mps, duty);
    }
  }

  /* Moves all 4 steering servos together (SYNC_WRITE under the hood). */
  on_target_angles_received(angles_deg, SWERVE_NUM_MODULES);
}

/* ================================================================== */
/*                     터미널 명령 보조 함수                           */
/* ================================================================== */

/** cmd 가 key 로 시작하는지 (대소문자 무시) */
static int cmd_is(const char *cmd, const char *key)
{
  while (*key)
  {
    char c = *cmd;
    if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
    if (c != *key) return 0;
    cmd++;
    key++;
  }
  return 1;
}

/**
 * @brief  현재 물리적 자세를 0도(위치값 2048)로 영점 기록
 *
 * 토크 스위치 레지스터(0x28)에 128을 쓰면 서보가 현재 위치를 2048로
 * 보정하고 EEPROM에 저장합니다 (공식 메모리표: "写128：任意当前位置较正为2048").
 * 쓰기만 하면 되므로 위치 읽기(sts_read)에 의존하지 않습니다.
 */
static void servo_zero_here(uint8_t id)
{
  uint8_t v = 128;
  STS_Status ret = sts_write(id, STS_REG_TORQUE_ENABLE, &v, 1);
  printf("  ID=%u 영점 기록 %s\r\n", id, (ret == STS_OK) ? "성공" : "실패/무응답");
}

/* ================================================================== */
/*              터미널 명령 파서 — 줄 단위 텍스트 처리                 */
/* ================================================================== */
static void process_command(const char *cmd)
{
  /* --- "V <vx> <vy> <omega>" : 스워브 기구학 명령 (body-frame m/s, rad/s) --- */
  if (cmd[0] == 'V' || cmd[0] == 'v')
  {
    /* nano.specs 에 -u _scanf_float 가 없어서 sscanf("%f") 는 동작하지 않음.
     * strtof 는 플래그 없이 쓸 수 있음 (T 명령과 동일한 방식). */
    float v[3] = {0.0f, 0.0f, 0.0f};
    int parsed = 0;
    const char *p = cmd + 1;
    for (int i = 0; i < 3; i++)
    {
      char *endp;
      float val = strtof(p, &endp);
      if (endp == p) break;
      v[i] = val;
      parsed++;
      p = endp;
    }

    if (parsed == 3)
    {
      /* 첫 V 명령에서 텔레옵으로 전환: 모터 자동 테스트 시퀀스를 멈추지
       * 않으면 그쪽이 Motor_SetSpeed 로 조이스틱 명령을 계속 덮어쓴다. */
      if (!s_teleop_active)
      {
        s_teleop_active = 1;
        Motor_StopAll();
        /* 연속 수신 중 ISR 이 송신을 기다리면 오버런이 나므로 에코를 끈다 */
        s_echo_enabled = 0;
        printf("[텔레옵] 시작 — 모터 자동 테스트 중단, 에코 OFF (복귀하려면 리셋)\r\n");
      }
      s_last_v_cmd_ms = HAL_GetTick();
      s_teleop_timed_out = 0;

      Swerve_Drive(v[0], v[1], v[2]);
    }
    else
    {
      printf("[CMD] 사용법: V <vx> <vy> <omega>  (예: V 0.1 0 0)\r\n");
    }
  }
  /* --- "T 10.5 -20" : 목표 각도 설정 --- */
  else if (cmd[0] == 'T' || cmd[0] == 't')
  {
    float angles[SERVO_COUNT] = {0};
    int parsed = 0;

    /* "T" 이후의 숫자들을 파싱 */
    const char *p = cmd + 1;
    for (int i = 0; i < SERVO_COUNT; i++)
    {
      char *endp;
      float val = strtof(p, &endp);
      if (endp == p) break;  /* 더 이상 숫자 없음 */
      angles[i] = val;
      parsed++;
      p = endp;
    }

    if (parsed > 0)
    {
      printf("[CMD] 목표 각도:");
      for (int i = 0; i < parsed; i++)
      {
        printf(" %.1f", angles[i]);
      }
      printf(" (도)\r\n");

      on_target_angles_received(angles, parsed);
    }
    else
    {
      printf("[CMD] 사용법: T <각1> <각2> <각3> <각4>  (예: T 45 -45 45 -45)\r\n");
    }
  }
  /* --- "PING" : 연결 확인 (ID 1~SERVO_COUNT) ---
   * 주의: 반드시 단일 문자 "P" 분기보다 먼저 와야 함. 뒤에 두면 'P'가
   * "PING"을 먼저 잡아채서 이 분기에 영영 도달하지 못함. */
  else if (cmd_is(cmd, "PING"))
  {
    for (int i = 0; i < SERVO_COUNT; i++)
    {
      STS_Status ret = sts_ping(i + 1);
      printf("  Servo ID=%d: %s\r\n", i + 1, (ret == STS_OK) ? "OK" : "FAIL");
    }
  }
  /* --- "P" : 현재 각도 출력 --- */
  else if (cmd[0] == 'P' || cmd[0] == 'p')
  {
    printf("[현재 각도]\r\n");
    for (int i = 0; i < SERVO_COUNT; i++)
    {
      float angle = joint_get_angle(i);
      int16_t pos = sts_read_position(i + 1);  /* ID = i+1 */
      printf("  Servo %d: %.1f deg (raw pos=%d)\r\n", i + 1, angle, pos);
    }
  }
  /* --- "SCAN" : 버스 전체 스캔 (ID 1~253) ---
   * PING 과 마찬가지로 단일 문자 "S"(속도) 분기보다 먼저 와야 함. */
  else if (cmd_is(cmd, "SCAN"))
  {
    /* 응답 없는 ID마다 수신 타임아웃(15ms)을 기다리므로 최대 ~4초간
     * 이 루프가 메인 루프를 막음. 그동안 모터가 계속 돌지 않도록 정지. */
    Motor_StopAll();

    printf("[SCAN] 버스 스캔 (ID 1~253), 약 4초 소요...\r\n");
    int found = 0;
    for (int id = 1; id <= 253; id++)
    {
      if (sts_ping((uint8_t)id) == STS_OK)
      {
        printf("  ID=%3d : 응답 OK\r\n", id);
        found++;
      }
      if ((id % 64) == 0) printf("  ... %d 까지 검사\r\n", id);
    }
    printf("[SCAN] 완료 — 서보 %d개 발견\r\n", found);
    if (found == 0)
    {
      printf("  배선(PB10/GND), 서보 전원(12V), 보레이트 1Mbps 확인 필요\r\n");
    }
  }
  /* --- "S 200" : 이동 속도 설정 --- */
  else if (cmd[0] == 'S' || cmd[0] == 's')
  {
    int speed = 0;
    if (sscanf(cmd + 1, "%d", &speed) == 1 && speed >= 0)
    {
      joint_set_speed((uint16_t)speed);
    }
    else
    {
      printf("[CMD] 사용법: S <속도>  (예: S 200, S 0=최대속도)\r\n");
    }
  }
  /* --- "ON" : 토크 ON --- */
  else if ((cmd[0] == 'O' || cmd[0] == 'o') && (cmd[1] == 'N' || cmd[1] == 'n'))
  {
    joint_set_torque_all(1);
  }
  /* --- "OFF" : 토크 OFF --- */
  else if ((cmd[0] == 'O' || cmd[0] == 'o') && (cmd[1] == 'F' || cmd[1] == 'f'))
  {
    joint_set_torque_all(0);
  }
  /* --- "LOG" : 주기적 엔코더 로그 on/off (대화형 작업 시 필수) --- */
  else if (cmd_is(cmd, "LOG"))
  {
    s_log_enabled = !s_log_enabled;
    printf("[LOG] 주기 로그 %s\r\n", s_log_enabled ? "ON" : "OFF");
  }
  /* --- "ECHO" : 입력 문자 되돌려보내기 on/off --- */
  else if (cmd_is(cmd, "ECHO"))
  {
    s_echo_enabled = !s_echo_enabled;
    printf("[ECHO] 입력 에코 %s\r\n", s_echo_enabled ? "ON" : "OFF");
  }
  /* --- "ZERO <id>" / "ZERO ALL" : 현재 자세를 0도로 영점 기록 --- */
  else if (cmd_is(cmd, "ZERO"))
  {
    const char *arg = cmd + 4;
    while (*arg == ' ') arg++;

    if (cmd_is(arg, "ALL"))
    {
      printf("[ZERO] 서보 %d개 전체 영점 기록...\r\n", SERVO_COUNT);
      for (int i = 0; i < SERVO_COUNT; i++)
      {
        servo_zero_here((uint8_t)(i + 1));
      }
    }
    else
    {
      int id = 0;
      if (sscanf(arg, "%d", &id) == 1 && id >= 1 && id <= 253)
      {
        printf("[ZERO] ID=%d 영점 기록...\r\n", id);
        servo_zero_here((uint8_t)id);
      }
      else
      {
        printf("[ZERO] 사용법: ZERO <id>  또는  ZERO ALL\r\n");
        printf("       순서: OFF -> 손으로 바퀴 정면 정렬 -> ZERO ALL -> ON -> T 0 0 0 0\r\n");
        return;
      }
    }
    printf("[ZERO] 완료. 'ON' 으로 토크 켜고 'T 0 0 0 0' 으로 확인하세요.\r\n");
  }
  /* --- "ID <현재> <새>" : 서보 버스 ID 변경 --- */
  else if (cmd_is(cmd, "ID"))
  {
    int old_id = -1, new_id = -1;
    if (sscanf(cmd + 2, "%d %d", &old_id, &new_id) == 2 &&
        old_id >= 0 && old_id <= 254 && new_id >= 0 && new_id <= 253)
    {
      printf("[ID] %d -> %d 변경 시도\r\n", old_id, new_id);

      /* 잠금 플래그 0 = EPROM 쓴 값이 전원 차단 후에도 유지됨.
       * 기본값이 0이지만 누가 1로 바꿔놨을 수 있으므로 명시적으로 씀. */
      uint8_t unlock = 0;
      sts_write((uint8_t)old_id, STS_REG_LOCK_FLAG, &unlock, 1);

      uint8_t v = (uint8_t)new_id;
      STS_Status ret = sts_write((uint8_t)old_id, STS_REG_ID_ADDR, &v, 1);

      if (old_id == STS_BROADCAST_ID)
      {
        printf("[ID] 브로드캐스트 전송 완료 (응답 없는 것이 정상)\r\n");
      }
      else if (ret != STS_OK)
      {
        printf("[ID] 경고: 쓰기 응답 이상 (err=%d)\r\n", ret);
      }

      printf("[ID] 새 ID=%d 확인: %s\r\n", new_id,
             (sts_ping((uint8_t)new_id) == STS_OK) ? "OK" : "응답 없음");
    }
    else
    {
      printf("[ID] 사용법: ID <현재ID> <새ID>\r\n");
      printf("     ** 반드시 서보를 1개만 연결한 상태에서 실행 **\r\n");
      printf("     (여러 개 물린 채로 하면 전부 같은 ID가 됨)\r\n");
      printf("     254 = 브로드캐스트, 현재 ID를 몰라도 됨.  예) ID 254 3\r\n");
    }
  }
  /* --- "HELP" : 사용법 --- */
  else if (strncmp(cmd, "HELP", 4) == 0 || strncmp(cmd, "help", 4) == 0)
  {
    print_help();
  }
  /* --- 알 수 없는 명령 ---
   * 조이스틱이 연속 전송 중일 때 명령 처리에 밀려 바이트가 잘리면 여기로
   * 조각이 들어온다. 그때마다 도움말 전체를 찍으면 UART 가 마비되므로
   * 한 줄만 출력한다. */
  else
  {
    printf("[CMD] 알 수 없는 명령: '%s'  (HELP 입력 시 도움말)\r\n", cmd);
  }
}

static void print_help(void)
{
  printf("========== 서보 명령어 도움말 ==========\r\n");
  printf("  T <a1> <a2>  : 목표 각도 설정 (도)\r\n");
  printf("                 예) T 10.5 -20\r\n");
  printf("  V <vx> <vy> <omega> : 스워브 기구학 (m/s, m/s, rad/s)\r\n");
  printf("                 예) V 0.1 0 0  (직진)\r\n");
  printf("  P            : 현재 각도 출력\r\n");
  printf("  S <speed>    : 이동 속도 설정 (0=최대)\r\n");
  printf("                 예) S 200\r\n");
  printf("  ON           : 토크 ON\r\n");
  printf("  OFF          : 토크 OFF\r\n");
  printf("  PING         : 서보 연결 확인 (ID 1~4)\r\n");
  printf("  SCAN         : 버스 전체 스캔 (ID 1~253, 약 4초)\r\n");
  printf("  ID <현재> <새> : 서보 ID 변경 [1개만 연결한 상태에서!]\r\n");
  printf("                 예) ID 254 3   (254=브로드캐스트)\r\n");
  printf("  ZERO <id|ALL>: 현재 자세를 0도로 영점 기록 [EEPROM 저장]\r\n");
  printf("                 순서) OFF -> 정면 정렬 -> ZERO ALL -> ON\r\n");
  printf("  LOG          : 주기 엔코더 로그 on/off (기본 ON)\r\n");
  printf("  ECHO         : 입력 문자 에코 on/off (기본 ON)\r\n");
  printf("  HELP         : 이 도움말\r\n");
  printf("========================================\r\n");
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_TIM1_Init();
  MX_TIM2_Init();
  MX_TIM3_Init();
  MX_TIM4_Init();
  MX_TIM5_Init();
  MX_TIM8_Init();
  MX_TIM12_Init();
  /* USER CODE BEGIN 2 */
  MX_USART2_Init();
  Motor_Init();
  Encoder_Init();

  printf("\r\n==================================================\r\n");
  printf("     STM32F446RE Rover Motor & Encoder Test       \r\n");
  printf("==================================================\r\n");
  printf("Baud rate: 115200 bps\r\n");
  printf("Motor Speed: +/-400 (PWM Duty ~50%%)\r\n\r\n");

  /* === 서보 모터 초기화 === */
  MX_USART3_Init();           /* USART3 (PB10/PB11, 1Mbps) 레지스터 초기화 */
  sts_init();                 /* STS3215 드라이버 초기화 */
  MX_USART2_RxInt_Init();     /* USART2 RX 인터럽트 활성화 */
  servo_boot_sequence();      /* PING → 토크 ON → 준비 완료 메시지 */

  HAL_Delay(1000);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  WheelState_t wheels[MOTOR_COUNT] = {0};
  uint32_t last_log_time = HAL_GetTick();
  /* 엔코더 속도 추정을 **로깅과 분리**한다.  예전엔 last_log_time 하나로 둘을
   * 같이 돌려서, 로그 주기를 바꾸면 속도 추정 주기까지 바뀌었다 (그리고 10Hz 는
   * 바퀴 속도 폐루프에 너무 느리다).
   * 8384 tick/rev 이므로 20ms 창에서도 분해능이 충분하다:
   *     0.20 m/s -> 119 tick / 20ms,   0.02 m/s -> 12 tick / 20ms */
  uint32_t last_enc_time = HAL_GetTick();
  const uint32_t ENC_PERIOD_MS = 20U;          /* 50 Hz.  PI 를 넣으면 100~200Hz 로 */
  uint32_t step_start_time = HAL_GetTick();
  uint8_t step = 0;

  /* Initialize last raw counts */
  for (uint8_t i = 0; i < MOTOR_COUNT; i++) {
    wheels[i].last_raw_count = Encoder_GetCount((MotorID)i);
  }

  /* Test parameters */
  const int16_t TEST_SPEED = 400; /* ~50% PWM Duty (PWM_MAX is 799) */

  while (1)
  {
    uint32_t now = HAL_GetTick();

    /* 1a. 엔코더 속도 추정 (50 Hz, 로깅과 독립) */
    if (now - last_enc_time >= ENC_PERIOD_MS)
    {
      float dt = (float)(now - last_enc_time) / 1000.0f;
      last_enc_time = now;

      for (uint8_t i = 0; i < MOTOR_COUNT; i++)
      {
        int32_t current_raw = Encoder_GetCount((MotorID)i);
        int32_t delta_tick = current_raw - wheels[i].last_raw_count;
        /* TIM1/4/8 are 16-bit: they roll over every 65536 ticks (~7.8 wheel
         * revolutions, ~6 s at full speed).  Without this fold the rollover
         * reads as a 65535-tick jump -> a ~4700 RPM spike on one wheel. */
        if (!Encoder_IsWide((MotorID)i)) delta_tick = (int16_t)delta_tick;
        wheels[i].last_raw_count = current_raw;
        wheels[i].total_count += delta_tick;

        /* Wheel RPM = (delta_tick / TICKS_PER_REV) * (60.0f / dt) */
        wheels[i].rpm = ((float)delta_tick / TICKS_PER_REV) * (60.0f / dt);

        /* Angular velocity w [rad/s] = RPM * (2*pi / 60) */
        wheels[i].rad_per_sec = wheels[i].rpm * (2.0f * PI / 60.0f);

        /* Linear velocity v [m/s] = w * R */
        wheels[i].linear_vel = wheels[i].rad_per_sec * WHEEL_RADIUS_M;
      }

    }

    /* 1b. 로깅 (10 Hz).  printf 는 제어 루프와 절대 같은 주기에 두지 않는다. */
    if (now - last_log_time >= 100)
    {
      float dt = (float)(now - last_log_time) / 1000.0f;
      last_log_time = now;
      if (s_log_enabled)
      {
      printf("[TIME:%6lums | dt:%3dms | R=%dmm]\r\n", now, (int)(dt * 1000.0f),
             (int)(WHEEL_RADIUS_M * 1000.0f + 0.5f));
      for (uint8_t i = 0; i < MOTOR_COUNT; i++)
      {
        int rpm_int = (int)wheels[i].rpm;
        int rpm_frac = (int)(fabsf(wheels[i].rpm) * 10.0f) % 10;
        int rad_int = (int)wheels[i].rad_per_sec;
        int rad_frac = (int)(fabsf(wheels[i].rad_per_sec) * 100.0f) % 100;
        int vel_int = (int)wheels[i].linear_vel;
        int vel_frac = (int)(fabsf(wheels[i].linear_vel) * 1000.0f) % 1000;

        printf(" M%d | Raw:%7ld | %4d.%1d RPM | %3d.%02d rad/s | %2d.%03d m/s\r\n",
               i + 1,
               wheels[i].last_raw_count,
               rpm_int, rpm_frac,
               rad_int, rad_frac,
               vel_int, vel_frac);
      }
      printf("-----------------------------------------------------------------\r\n");
      }  /* if (s_log_enabled) */
    }

    /* 2. Step-by-Step Motor Test Sequence
     *    텔레옵(V 명령)이 시작되면 중단한다. 안 그러면 이 시퀀스가
     *    Motor_SetSpeed 로 조이스틱 명령을 몇 초마다 덮어쓴다. */
    if (!s_teleop_active)
    {
    switch (step)
    {
      /* --- Motor 1 Test --- */
      case 0:
        if (s_log_enabled) printf("\r\n>>> [Step 0] Motor 1 Forward (+%d) for 2s <<<\r\n", TEST_SPEED);
        Motor_SetSpeed(MOTOR1, TEST_SPEED);
        step++;
        step_start_time = now;
        break;
      case 1:
        if (now - step_start_time >= 2000) {
          if (s_log_enabled) printf(">>> [Step 1] Motor 1 Reverse (-%d) for 2s <<<\r\n", TEST_SPEED);
          Motor_SetSpeed(MOTOR1, -TEST_SPEED);
          step++;
          step_start_time = now;
        }
        break;
      case 2:
        if (now - step_start_time >= 2000) {
          if (s_log_enabled) printf(">>> [Step 2] Motor 1 Stop (1s pause) <<<\r\n");
          Motor_Stop(MOTOR1);
          step++;
          step_start_time = now;
        }
        break;

      /* --- Motor 2 Test --- */
      case 3:
        if (now - step_start_time >= 1000) {
          if (s_log_enabled) printf("\r\n>>> [Step 3] Motor 2 Forward (+%d) for 2s <<<\r\n", TEST_SPEED);
          Motor_SetSpeed(MOTOR2, TEST_SPEED);
          step++;
          step_start_time = now;
        }
        break;
      case 4:
        if (now - step_start_time >= 2000) {
          if (s_log_enabled) printf(">>> [Step 4] Motor 2 Reverse (-%d) for 2s <<<\r\n", TEST_SPEED);
          Motor_SetSpeed(MOTOR2, -TEST_SPEED);
          step++;
          step_start_time = now;
        }
        break;
      case 5:
        if (now - step_start_time >= 2000) {
          if (s_log_enabled) printf(">>> [Step 5] Motor 2 Stop (1s pause) <<<\r\n");
          Motor_Stop(MOTOR2);
          step++;
          step_start_time = now;
        }
        break;

      /* --- Motor 3 Test --- */
      case 6:
        if (now - step_start_time >= 1000) {
          if (s_log_enabled) printf("\r\n>>> [Step 6] Motor 3 Forward (+%d) for 2s <<<\r\n", TEST_SPEED);
          Motor_SetSpeed(MOTOR3, TEST_SPEED);
          step++;
          step_start_time = now;
        }
        break;
      case 7:
        if (now - step_start_time >= 2000) {
          if (s_log_enabled) printf(">>> [Step 7] Motor 3 Reverse (-%d) for 2s <<<\r\n", TEST_SPEED);
          Motor_SetSpeed(MOTOR3, -TEST_SPEED);
          step++;
          step_start_time = now;
        }
        break;
      case 8:
        if (now - step_start_time >= 2000) {
          if (s_log_enabled) printf(">>> [Step 8] Motor 3 Stop (1s pause) <<<\r\n");
          Motor_Stop(MOTOR3);
          step++;
          step_start_time = now;
        }
        break;

      /* --- Motor 4 Test --- */
      case 9:
        if (now - step_start_time >= 1000) {
          if (s_log_enabled) printf("\r\n>>> [Step 9] Motor 4 Forward (+%d) for 2s <<<\r\n", TEST_SPEED);
          Motor_SetSpeed(MOTOR4, TEST_SPEED);
          step++;
          step_start_time = now;
        }
        break;
      case 10:
        if (now - step_start_time >= 2000) {
          if (s_log_enabled) printf(">>> [Step 10] Motor 4 Reverse (-%d) for 2s <<<\r\n", TEST_SPEED);
          Motor_SetSpeed(MOTOR4, -TEST_SPEED);
          step++;
          step_start_time = now;
        }
        break;
      case 11:
        if (now - step_start_time >= 2000) {
          if (s_log_enabled) printf(">>> [Step 11] Motor 4 Stop (1s pause) <<<\r\n");
          Motor_Stop(MOTOR4);
          step++;
          step_start_time = now;
        }
        break;

      /* --- All Motors Forward Test --- */
      case 12:
        if (now - step_start_time >= 1000) {
          if (s_log_enabled) printf("\r\n>>> [Step 12] All Motors Forward (+%d) for 3s <<<\r\n", TEST_SPEED);
          for (uint8_t i = 0; i < MOTOR_COUNT; i++) {
            Motor_SetSpeed((MotorID)i, TEST_SPEED);
          }
          step++;
          step_start_time = now;
        }
        break;
      case 13:
        if (now - step_start_time >= 3000) {
          if (s_log_enabled) printf(">>> [Step 13] All Motors Stop & Reset Encoders (2s pause) <<<\r\n");
          Motor_StopAll();
          for (uint8_t i = 0; i < MOTOR_COUNT; i++) {
            Encoder_Reset((MotorID)i);
            wheels[i].last_raw_count = 0;
            wheels[i].total_count = 0;
          }
          step = 0; /* Loop test sequence */
          step_start_time = now + 1000; /* 2s pause */
        }
        break;

      default:
        step = 0;
        break;
    }
    }  /* if (!s_teleop_active) */

    /* 2b. 텔레옵 워치독 — 명령이 끊기면 정지.
     *     마지막 속도 명령이 그대로 유지되면 조이스틱을 닫거나 USB 가
     *     빠져도 로버가 계속 굴러간다. */
    if (s_teleop_active && !s_teleop_timed_out &&
        (now - s_last_v_cmd_ms > TELEOP_TIMEOUT_MS))
    {
      Motor_StopAll();
      s_teleop_timed_out = 1;
      printf("[텔레옵] 명령 %dms 이상 끊김 — 모터 정지\r\n", TELEOP_TIMEOUT_MS);
    }
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

    /* 3. 서보 터미널 명령 처리 (비블로킹)
     *    파싱하는 동안 ISR 이 cmd_line 을 덮어쓸 수 있으므로 지역 버퍼로
     *    복사한 뒤 처리한다. */
    if (cmd_ready)
    {
      char line[CMD_BUF_SIZE];
      __disable_irq();
      memcpy(line, cmd_line, sizeof line);
      cmd_ready = 0;
      __enable_irq();
      line[sizeof line - 1] = '\0';

      process_command(line);
    }

  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 0;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 65535;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 0;
  sConfig.IC2Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 0;
  if (HAL_TIM_Encoder_Init(&htim1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */

}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 0;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 799;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */
  HAL_TIM_MspPostInit(&htim2);

}

/**
  * @brief TIM3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM3_Init(void)
{

  /* USER CODE BEGIN TIM3_Init 0 */

  /* USER CODE END TIM3_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */
  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 0;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 799;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_4) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */
  HAL_TIM_MspPostInit(&htim3);

}

/**
  * @brief TIM4 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM4_Init(void)
{

  /* USER CODE BEGIN TIM4_Init 0 */

  /* USER CODE END TIM4_Init 0 */

  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM4_Init 1 */

  /* USER CODE END TIM4_Init 1 */
  htim4.Instance = TIM4;
  htim4.Init.Prescaler = 0;
  htim4.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim4.Init.Period = 65535;
  htim4.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim4.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 0;
  sConfig.IC2Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 0;
  if (HAL_TIM_Encoder_Init(&htim4, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim4, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM4_Init 2 */

  /* USER CODE END TIM4_Init 2 */

}

/**
  * @brief TIM5 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM5_Init(void)
{

  /* USER CODE BEGIN TIM5_Init 0 */

  /* USER CODE END TIM5_Init 0 */

  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM5_Init 1 */

  /* USER CODE END TIM5_Init 1 */
  htim5.Instance = TIM5;
  htim5.Init.Prescaler = 0;
  htim5.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim5.Init.Period = 4294967295;
  htim5.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim5.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 0;
  sConfig.IC2Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 0;
  if (HAL_TIM_Encoder_Init(&htim5, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim5, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM5_Init 2 */

  /* USER CODE END TIM5_Init 2 */

}

/**
  * @brief TIM8 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM8_Init(void)
{

  /* USER CODE BEGIN TIM8_Init 0 */

  /* USER CODE END TIM8_Init 0 */

  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM8_Init 1 */

  /* USER CODE END TIM8_Init 1 */
  htim8.Instance = TIM8;
  htim8.Init.Prescaler = 0;
  htim8.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim8.Init.Period = 65535;
  htim8.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim8.Init.RepetitionCounter = 0;
  htim8.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 0;
  sConfig.IC2Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 0;
  if (HAL_TIM_Encoder_Init(&htim8, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim8, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM8_Init 2 */

  /* USER CODE END TIM8_Init 2 */

}

/**
  * @brief TIM12 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM12_Init(void)
{

  /* USER CODE BEGIN TIM12_Init 0 */

  /* USER CODE END TIM12_Init 0 */

  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM12_Init 1 */

  /* USER CODE END TIM12_Init 1 */
  htim12.Instance = TIM12;
  htim12.Init.Prescaler = 0;
  htim12.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim12.Init.Period = 799;
  htim12.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim12.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim12) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim12, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim12, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM12_Init 2 */

  /* USER CODE END TIM12_Init 2 */
  HAL_TIM_MspPostInit(&htim12);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
