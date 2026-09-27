#!/usr/bin/env bash
# Rebuilds libksrc_common.so from the real firmware/common/ sources so the
# simulator exercises the exact same kinematics/protocol C code that ships
# to the Nucleo -- not a Python reimplementation.
set -euo pipefail
cd "$(dirname "$0")"
gcc -shared -fPIC -O2 -o libksrc_common.so \
    ../firmware/common/swerve_kinematics.c \
    ../firmware/common/teleop_protocol.c \
    -lm
echo "built sim/libksrc_common.so"
