#!/usr/bin/env bash
# Host-side wrist camera ZMQ sender launcher.
# Cameras are pinned by USB PORT (by-path), so LEFT/RIGHT never swap on reboot/replug.
#   LEFT  wrist = USB port 4.1.2
#   RIGHT wrist = USB port 4.1.3
# NOTE: these are tied to the physical USB port. If you move a camera to a DIFFERENT
# port, update the path to that port. If left/right look swapped, swap the two values below.
set -euo pipefail

LEFT_DEV=/dev/v4l/by-path/pci-0000:80:14.0-usb-0:4.1.2:1.0-video-index0
RIGHT_DEV=/dev/v4l/by-path/pci-0000:80:14.0-usb-0:4.1.3:1.0-video-index0

ROBOT_IP=${ROBOT_IP:-192.168.123.164}
LEFT_PORT=${LEFT_PORT:-55602}
RIGHT_PORT=${RIGHT_PORT:-55604}
WIDTH=${WIDTH:-640}
HEIGHT=${HEIGHT:-480}
FPS=${FPS:-30}

# Python that has cv2 + pyzmq. The system python does NOT, so default to the opencv_test venv.
# Override with:  PYTHON=/path/to/python bash run_wrist_sender.sh
PYTHON=${PYTHON:-/home/kimm/opencv_test/bin/python}
if ! "$PYTHON" -c "import cv2, zmq" 2>/dev/null; then
  echo "ERROR: '$PYTHON' is missing cv2/zmq. Set PYTHON to an interpreter that has them, e.g.:" >&2
  echo "  PYTHON=/home/kimm/opencv_test/bin/python bash $0" >&2
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
  --fps "$FPS"
