/**
 * @file    imu_ism330.h
 * @brief   ISM330DHCX 6축 IMU (SPI3 마스터) 드라이버
 *
 * 핀 (rover_motor.ioc 와 동일):
 *   PC10 SPI3_SCK, PC11 SPI3_MISO, PC12 SPI3_MOSI, PC9 IMU_CS (GPIO, 평소 High)
 * 모드 0, 8비트, MSB first, fPCLK1/16 = 1MHz (센서 상한 10MHz).
 *
 * 설정: 가속도 208Hz ±4g (0.122 mg/LSB), 자이로 208Hz ±500dps (17.5 mdps/LSB),
 *       BDU=1 (상·하위 바이트가 다른 샘플에서 섞이지 않게), 주소 자동증가.
 * 50Hz 제어주기에서 가장 최근 샘플 하나를 읽는다. 값은 **센서 좌표계 원시값**이며
 * 차체 좌표계로의 회전(장착 방향)은 Pi 쪽에서 한다.
 *
 * SPI 마스터는 센서가 없어도 오류가 나지 않고 0x00/0xFF 를 읽는다. 그래서
 * WHO_AM_I(0x6B) 를 주기적으로 다시 확인해 유효 여부를 판단한다.
 *
 * CubeMX 가 만드는 MX_SPI3_Init() 은 호출하지 말 것 (레지스터로 직접 설정).
 */

#ifndef IMU_ISM330_H
#define IMU_ISM330_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/** SPI3·CS 핀 초기화 + 센서 리셋·설정.  @return 1 = WHO_AM_I 확인 성공 */
int Imu_Init(void);

/**
 * @brief  최신 샘플 읽기
 * @param  out  gyro x,y,z, acc x,y,z (원시 i16)
 * @return 1 = 유효, 0 = 센서 미확인 상태 (out 은 0)
 */
int Imu_Read(int16_t out[6]);

/**
 * @brief  WHO_AM_I 와 설정 레지스터 재확인. 어긋났으면(센서 단독 리셋 등) 바로 재설정한다.
 *         1초에 한 번 정도 부른다 (배선 흔들림·핫플러그 대비).
 * @return 1 = 유효
 */
int Imu_Check(void);

#ifdef __cplusplus
}
#endif

#endif /* IMU_ISM330_H */
