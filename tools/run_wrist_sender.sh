#!/usr/bin/env bash
# Host-side wrist camera ZMQ sender launcher.
# Cameras are pinned by USB PORT (by-path), so LEFT/RIGHT never swap on reboot/replug.
#   LEFT  wrist = USB port 1.4  (usb-c hub port)
#   RIGHT wrist = USB port 1.3  (usb-c hub port)
# NOTE: these are tied to the physical USB port. If you move the hub to a DIFFERENT
# PC port, update these paths. If left/right look swapped, swap the two values below.
# 같은 모델 폴백은 꺼 둔다 (CAMERA_USB_ID 가 비어 있음).
# 두 손목캠은 vid:pid 도 시리얼도 똑같다(0edc:2b10 / 200901010001). 그래서 폴백이 집어 든
# 카메라가 왼손인지 오른손인지 확인할 방법이 없다 — 좌우가 뒤바뀐 채 수집될 수 있고,
# 저장물만 봐서는 판별이 사실상 불가능하다. 다른 포트에 꽂혔다면 자동으로 넘어갈 게 아니라
# 아래 by-path 값을 고쳐야 할 상황이다. 굳이 되살리려면 CAMERA_USB_ID=0edc:2b10 으로 준다.
# Tip: after plugging in, find the current port with:
#   for v in /dev/video*; do echo "$v -> $(realpath /sys/class/video4linux/$(basename $v)/device)"; done
set -euo pipefail

LEFT_DEV=/dev/v4l/by-path/pci-0000:80:14.0-usb-0:1.4:1.0-video-index0
RIGHT_DEV=/dev/v4l/by-path/pci-0000:80:14.0-usb-0:1.3:1.0-video-index0

ROBOT_IP=${ROBOT_IP:-192.168.123.164}
LEFT_PORT=${LEFT_PORT:-55602}
RIGHT_PORT=${RIGHT_PORT:-55604}
WIDTH=${WIDTH:-640}
HEIGHT=${HEIGHT:-480}
# ── FPS 를 20 으로 "맞추지" 말 것 ──────────────────────────────────────────
# Loop 이 기록하는 값은 20fps 이고 dfvidcheck 의 기대값도 20 이다. 그래서 여기 30 이
# 불일치처럼 보이지만 **의도된 것이다.** 송신기는 생산자, Loop 은 소비자다.
#   · 30 으로 공급하면 Loop 이 20fps 로 표집할 때 항상 33ms 이내의 새 프레임이 있다.
#   · 20 으로 낮추면 생산·소비 주기가 같아져 맞물림(beat)이 생기고, 한 번만 밀려도
#     같은 프레임이 두 번 들어가거나 기록 fps 가 20 아래로 떨어질 수 있다.
# 대역폭·CPU 때문에 줄일 이유도 없다 — 실측(2026-08-22, 640x480 MJPG q85):
#   프레임당 43KB · 카메라 1대 1.31MB/s · 2대 2.62MB/s = USB2.0 실효의 약 7%.
#   스레드 CPU 도 4.4% 수준이다. 20 으로 낮춰 아끼는 것은 사실상 없다.
# USB 끊김은 대역폭이 아니라 커넥터 접촉 문제다(dfpc04/cable_analysis/ 참고).
FPS=${FPS:-30}
CAMERA_USB_ID=${CAMERA_USB_ID:-}
LOG_FILE=${LOG_FILE:-$HOME/dfws/logs/wrist_sender_%Y%m%d.log}
STATE_FILE=${STATE_FILE:-$HOME/dfws/logs/wrist_sender.state}

# Python interpreter with cv2 + pyzmq.
# Override with:  PYTHON=/path/to/python bash run_wrist_sender.sh
PYTHON=${PYTHON:-python3}
if ! "$PYTHON" -c "import cv2, zmq" 2>/dev/null; then
  echo "ERROR: '$PYTHON' is missing cv2/zmq. Set PYTHON to an interpreter that has them, e.g.:" >&2
  echo "  PYTHON=/path/to/venv/bin/python bash $0" >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

exec "$PYTHON" "$SCRIPT_DIR/host_wrist_zmq_sender.py" \
  --left-device "$LEFT_DEV" \
  --right-device "$RIGHT_DEV" \
  --robot-ip "$ROBOT_IP" \
  --left-port "$LEFT_PORT" \
  --right-port "$RIGHT_PORT" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --fps "$FPS" \
  --camera-usb-id "$CAMERA_USB_ID" \
  --log-file "$LOG_FILE" \
  --state-file "$STATE_FILE"
