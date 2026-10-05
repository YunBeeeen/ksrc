/*
 * Pi <-> Nucleo SPI 프레임 호스트 테스트. STM32/HAL 불필요:
 *   gcc -std=c99 -Wall -Wextra -I.. -o /tmp/spi_test test_spi_link.c ../spi_link.c && /tmp/spi_test
 */
#include "spi_link.h"
#include <stdio.h>
#include <string.h>

static int g_fail = 0;

static void expect(const char *label, int cond)
{
    if (!cond) {
        printf("FAIL %s\n", label);
        g_fail = 1;
    } else {
        printf("ok   %s\n", label);
    }
}

int main(void)
{
    uint8_t f[SPI_LINK_FRAME_LEN];

    /* 1) CRC-16/CCITT-FALSE 표준 검증값 */
    expect("crc16.check_123456789",
           spi_link_crc16((const uint8_t *)"123456789", 9) == 0x29B1);

    /* 2) 명령 프레임 왕복 + 고정 위치 확인 */
    {
        spi_link_cmd_t c = { SPI_LINK_TYPE_VEL_CMD, 7, 0.1f, -0.05f, 0.5f };
        spi_link_cmd_t d;
        spi_link_pack_cmd(&c, f);
        expect("cmd.sync", f[0] == 0xA5 && f[1] == 0x5A);
        expect("cmd.vx_le", f[6] == 0xCD && f[7] == 0xCC && f[8] == 0xCC && f[9] == 0x3D);
        expect("cmd.padding_zero", f[20] == 0 && f[SPI_LINK_FRAME_LEN - 1] == 0);
        expect("cmd.parse_ok", spi_link_parse_cmd(f, &d) == SPI_LINK_OK);
        expect("cmd.values", d.type == 1 && d.seq == 7 && d.vx == 0.1f &&
                             d.vy == -0.05f && d.omega == 0.5f);

        f[10] ^= 0x01;
        expect("cmd.crc_detects_bitflip", spi_link_parse_cmd(f, &d) == SPI_LINK_ERR_CRC);
        spi_link_pack_cmd(&c, f);
        f[0] = 0x00;
        expect("cmd.bad_sync", spi_link_parse_cmd(f, &d) == SPI_LINK_ERR_SYNC);
        /* Pi 가 CS 만 내리고 0 만 보낸 경우 (배선 끊김 등) */
        memset(f, 0, sizeof f);
        expect("cmd.all_zero_rejected", spi_link_parse_cmd(f, &d) != SPI_LINK_OK);
        memset(f, 0xFF, sizeof f);
        expect("cmd.all_ff_rejected", spi_link_parse_cmd(f, &d) != SPI_LINK_OK);
    }

    /* 3) 텔레메트리 왕복 (음수 틱 / 음수 duty / 음수 imu 부호 보존) */
    {
        spi_link_telemetry_t t, u;
        memset(&t, 0, sizeof t);
        memset(&u, 0, sizeof u);   /* 구조체 끝 패딩까지 같게 (memcmp 비교) */
        t.seq_echo = 200; t.tick = 3;
        t.flags = SPI_TLM_FLAG_GATE_PENDING | SPI_TLM_FLAG_SRC_SPI;
        t.t_ms = 0xDEADBEEFu;
        t.cmd_vx = 0.2f; t.cmd_vy = -0.1f; t.cmd_omega = 1.5f;
        t.enc_ticks[0] = -123456; t.enc_ticks[3] = 2147483647;
        t.wheel_mps[1] = -0.25f;
        t.steer_rad[2] = -3.14159f;
        t.duty[0] = -799; t.duty[3] = 799;
        t.imu[0] = -32768; t.imu[5] = 32767;
        t.rx_err = 65535;
        t.steer_meas_rad[1] = 1.745f;
        t.steer_speed[0] = -3000; t.steer_load[2] = -1000;
        t.steer_volt[3] = 121; t.steer_temp[0] = 45; t.steer_status[1] = 0x20;
        t.servo_valid = 0x0B;
        spi_link_pack_telemetry(&t, f);
        expect("tlm.type", f[3] == SPI_LINK_TYPE_TELEMETRY);
        expect("tlm.parse_ok", spi_link_parse_telemetry(f, &u) == SPI_LINK_OK);
        expect("tlm.roundtrip", memcmp(&t, &u, sizeof t) == 0);
        f[120] ^= 0x80;
        expect("tlm.crc_detects_bitflip", spi_link_parse_telemetry(f, &u) == SPI_LINK_ERR_CRC);
    }

    if (g_fail) {
        printf("\nSOME TESTS FAILED\n");
        return 1;
    }
    printf("\nALL TESTS PASSED\n");
    return 0;
}
