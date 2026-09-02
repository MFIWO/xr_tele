# AI Worker + HX5 native Vision Pro PoC

이 문서는 stock visionOS `Tracking Streamer`와 `avp_stream==2.51`을 이용해
AI Worker arm IK와 기존 HX5-D20 retarget 경로를 검증하는 실행 절차다. 첫 검증의
안전 경계는 `--sim`이 아니라 `--command-sink`다. `--sim`만으로는 이 브랜치의
AI Worker/HX5 motor DDS 생성이 막히지 않는다.

이 PoC는 native 앱의 mixed immersive space를 사용하므로 현실 passthrough 위에
하나의 mono `head + left wrist + right wrist` 합성 영상 plane을 표시한다. Swift,
protobuf, HX5 production controller는 변경하지 않는다.

## 원인, 데이터 흐름, 선택한 upstream API

기존 Vuer 경로는 Safari의 immersive WebXR session 안에서 pose와 영상을 받는다.
이 경로는 visionOS native `ImmersiveSpace`의 mixed composition을 직접 선택할 수
없기 때문에, Vuer를 robot/IK 쪽까지 교체하지 않고 XR 입출력만 stock native
`Tracking Streamer`로 대체했다.

로컬 `vuer==0.0.60` browser bundle도
`isSessionSupported("immersive-vr")`와 `requestSession("immersive-vr", ...)`를
호출한다. `televuer.py`의 Python `pass-through` 분기는 배경 영상 없이 숨긴
`Hands` 또는 `MotionControllers`만 올릴 뿐 `immersive-ar`/native mixed space를
요청하지 않는다. 따라서 이 이름은 Pico 계열 동작을 전제로 한 scene 선택이며,
Vision Pro system passthrough를 활성화하는 스위치가 아니다.

```text
Tracking Streamer (Vision Pro, TCP 12345 gRPC tracking)
  -> avp_stream.VisionProStreamer.get_latest()
  -> VisionProTeleopBackend -> 기존 TeleData ABI
  -> AIWorkerArmIK -> command sink 또는 기존 AIWorkerArmController
  -> 기존 HX5-D20 geometric retarget/controller

AI Worker camera DDS readers (head/left wrist/right wrist)
  -> 기존 head_and_wrist BGR composite
  -> VisionProTeleopBackend.render_to_xr()
  -> registered frame callback + update_frame()
  -> Linux TCP 9999 signaling/동적 WebRTC media -> native video plane

pedal U (UDP 8765) -> VisionProTeleopBackend HOLD -> arm/HX5 current-target hold
```

확인한 `avp_stream` 2.51 API는 다음과 같다.

- `VisionProStreamer(ip=..., record=False, ht_backend="grpc", origin="avp")`:
  `record=False`는 constructor에 주며 upstream session recording을 끈다. local
  constructor는 첫 tracking sample을 기다릴 수 있어 adapter의 daemon worker에서
  만든다.
- `get_latest()`: head/wrist, 25-joint `*_fingers`, 27-joint `*_arm`, pinch 값을
  포함한 최신 `TrackingData`를 반환한다.
- `configure_video(device=None, size="WIDTHxHEIGHT", fps=..., stereo=False)`와
  `register_frame_callback(...)`: program-generated mono BGR frame source를 만든다.
  `render_to_xr()`는 callback이 읽을 immutable latest frame을 갱신하고, stock
  `update_frame(frame)`의 `user_frame`도 갱신한다. 실제 copy/VideoFrame 생성과
  전송은 다음 WebRTC video-track pull에서 일어난다.
- `start_webrtc(port=9999, blocking=False)`는 outbound video를 시작하고
  `cleanup()`은 종료 경로에서 호출한다. `is_connected()`는 WebRTC media ICE
  상태일 뿐 tracking gRPC validity가 아니다.

stock normal tracking payload에는 일반 pose source timestamp/sequence와 ARKit
`isTracked`가 없다. 따라서 adapter의 `session_alive`는 새 raw object가 도착한
Linux monotonic 시각으로 계산한 freshness이며 native tracking-quality 표시는 아니다.

## 이 PoC에서 변경한 파일

- `README.md`: AI Worker에서 `--sim`만으로는 no-actuator 경계가 되지 않는다는
  경고와 pedal/Config Loop/replay 사용법을 기록한다.
- `teleop/teleop_hand_and_arm.py`: optional backend 선택, camera composite 연결,
  command-sink/실기 전 controller-construction gate, native tracking 상태와 U hold를
  기존 AI Worker IK/HX5 경로에 연결한다.
- `teleop/utils/visionproteleop_backend.py`: 좌표/25·27 joint adapter, mono video,
  timeout·유효성·jump·velocity·re-anchor 상태기계를 구현한다.
- `teleop/utils/ai_worker_command_sink.py`: 실제 AI Worker IK/HX5 target을 받되 motor
  DDS transport를 만들지 않는 memory-only controller를 제공한다.
- `teleop/utils/synthetic_image_client.py`: Vision Pro/DDS 없이 쓰는 deterministic
  세-camera 입력을 제공한다.
- `teleop/ai_worker_camera_diagnostic.py`: 실제 세 camera DDS reader의 topic, shape,
  FPS, freshness를 read-only로 검사한다.
- `teleop/robot_control/robotis_image_client.py`: camera DDS domain 선택 및 source/host
  receive timestamp 진단을 제공한다.
- `teleop/ai_worker_pedal_teleop.py`와
  `teleop/robot_control/robotis_ai_worker_lift.py`: 기존 W/A/S/D, hold O/P, latched U
  semantics와 fail-closed tracking heartbeat를 제공하고 dry-run 상태를 표시한다.
- `teleop/requirements-visionproteleop.txt`: optional `avp_stream==2.51` pin과 확인한
  artifact hash를 기록한다.
- `teleop/tests/test_xr_backend_selection.py`,
  `teleop/tests/test_visionproteleop_backend.py`,
  `teleop/tests/test_ai_worker_command_sink.py`,
  `teleop/tests/test_synthetic_image_client.py`,
  `teleop/tests/test_ai_worker_camera_diagnostic.py`,
  `teleop/tests/test_ai_worker_pedal_controls.py`,
  `teleop/tests/test_robotis_image_client_config.py`,
  `teleop/tests/test_robotis_ai_worker_virtual_leader.py`: backend/API,
  좌표·joint, fail-closed 상태, no-motor 경계, pedal/camera 진단과 sibling AI Worker
  URDF 기반 IK 회귀를 검사한다.
- `teleop/VISIONPROTELEOP_POC.md`: 검증 명령, PASS/FAIL과 실기 전 안전 경계를
  기록한다.

이 목록은 native PoC와 그 pedal/camera 안전 검증의 직접 변경 범위다. 같은 dirty
worktree의 episode/replay와 Config Loop 관련 사용자 변경은 별도 작업이며 이
PoC에서 되돌리거나 HX5 20-DoF ordering/scale/offset/optimizer/DDS layout을
변경하지 않았다.

## 현재 검증된 기준

- repository: `/home/kimm/Downloads/xr_tele`
- branch: `agent/ai-worker-hx5-teleop`
- 감사 시작 commit: `802adb4a7c86cd7c9f50969884a37da1d7d11bf1`
- Python: `/home/kimm/miniforge3/envs/tv/bin/python` (3.10.21)
- optional package: `avp_stream==2.51`
- 확인한 upstream source commit:
  `4c549905c2a8b214d79f7cd88e535101a1ce32af`
- 확인한 PyPI wheel SHA256:
  `03d19f56ca3d5a2013a0ae353ae68df8f898d1ae3c5567b49419b47421bc2412`

작업 시작 당시 tracked/untracked 사용자 변경이 이미 있었으므로 `git status`가
깨끗해야 한다고 가정하지 않는다. 아래 명령으로 branch, commit, 알려진 변경,
submodule pin을 먼저 기록한다.

```bash
cd /home/kimm/Downloads/xr_tele
git branch --show-current
git rev-parse HEAD
git status --short
git submodule status
```

PASS: branch가 `agent/ai-worker-hx5-teleop`이고 세 submodule 줄의 앞에 `-`, `+`,
`U`가 없다. FAIL: 다른 branch이거나 submodule이 초기화되지 않았거나 충돌 상태다.

## 데이터 계약과 좌표 변환

native adapter는 기존 wrapper와 같은 필드를 반환한다.

- head/left wrist/right wrist: 각각 `(4, 4)` rigid transform
- left/right hand position: 각각 `(25, 3)`, metre
- 선택적 hand rotation: 각각 `(25, 3, 3)` 무차원 rotation matrix. scalar angle과
  angle threshold만 radian을 사용한다.
- pinch bool과 `pinchValue` centimetre. upstream distance metre를 adapter가
  centimetre로 바꾼다. stock payload에 squeeze는 없어 squeeze bool/value는
  compatibility용 `False`/`0.0`이다.
- gesture 필드 이름은 좌우 각각 `*_hand_pinch`, `*_hand_pinchValue`,
  `*_hand_squeeze`, `*_hand_squeezeValue`다.
- 기존 controller ABI도 유지한다. 좌우 각각 `*_ctrl_trigger`,
  `*_ctrl_triggerValue`, `*_ctrl_squeeze`, `*_ctrl_squeezeValue`,
  `*_ctrl_aButton`, `*_ctrl_bButton`, `*_ctrl_thumbstick`,
  `*_ctrl_thumbstickValue` `(2,)`이며 native hand backend에서는 기본값이다.
- native 안전 annotation: `tracking_active`, `session_alive`,
  `head_pose_is_valid`, `left_arm_is_valid`, `right_arm_is_valid`,
  `tracking_state`, `tracking_reason`, `tracking_sample_age_s`,
  `native_tracking_status_available`.

stock skeleton 순서는 wrist `0`, thumb `1..4`, index `5..9`, middle `10..14`,
ring `15..19`, little `20..24`, optional forearm `25..26`이다. adapter는 stock
compatibility key인 25-joint `*_fingers`를 우선하고, 그것이 없으면 `*_arm`을
사용한다. 선택된 배열이 정확히 25개 또는 27개인지 검사하며, 27개이면 끝의
forearm 두 개만 제외해 기존 HX5가 받는 first-25 순서를 유지한다.

`A`를 stock `avp_stream`의 Y-up→Z-up transform, `P`를 기존
OpenXR→robot transform, `U_side`를 기존 left/right Unitree wrist transform이라
하면 adapter 식은 다음과 같다.

```text
H_robot = P A^-1 H_avp Rx(+pi/2) P^-1
W_robot = P A^-1 W_avp P^-1 U_side
W_robot.translation -= H_robot.translation
W_robot.translation += [0.15, 0.00, 0.45]
p_hand = T_hand P p_avp_wrist_local
```

축은 OpenXR/AVP right `+X` → robot right `-Y`, up `+Y` → up `+Z`,
back `+Z` → back `-X`다. 따라서 AVP forward `-Z`는 robot forward `+X`다.
단위는 translation metre, angle radian이다. identity, ±XYZ, 양 손목 방향,
roll/pitch/yaw, finger order/curl/pinch, orthonormality를 단위 테스트한다.

## 환경과 optional dependency

```bash
cd /home/kimm/Downloads/xr_tele
source /home/kimm/miniforge3/bin/activate tv
which python
python --version
python -m pip install -r teleop/requirements-visionproteleop.txt
python -c 'import importlib.metadata as m, avp_stream; from avp_stream import streamer; print("dist_version=", m.version("avp_stream")); print("protocol_version=", streamer.LIBRARY_VERSION); print("path=", avp_stream.__file__)'
```

PASS: interpreter가 `/home/kimm/miniforge3/envs/tv/bin/python`, Python이 3.10.x,
`dist_version=2.51`이고 import path가 해당 environment 안이다. 확인한 source의
Xcode marketing version은 2.51/build 16이지만 Python distribution 2.51 내부의
`LIBRARY_VERSION`/wire code는 `2.50.0`/25000이다. 따라서 앱의 Python version
card가 `2.50.0`으로 보여도 distribution mismatch가 아니며, compatibility warning이
없어야 한다. 실제 설치된 App Store build는 live 검증 때 함께 기록한다.

현재 검증 environment에는 `avp_stream`을 설치하지 않았으므로 위 install/live
import는 **NOT EXECUTED** 상태다. requirements의 SHA256 줄은 감사 기록이며 pip가
그 주석을 강제 검증하는 것은 아니다. Vuer는 이 package 없이 계속 import/실행
가능하다.

## 자동 테스트

```bash
cd /home/kimm/Downloads/xr_tele
/home/kimm/miniforge3/envs/tv/bin/python -m unittest discover -v -s teleop/tests
```

PASS: `OK`와 종료 코드 0. 특히 optional dependency, Vuer default/import,
25/27-joint, 좌표축, stale/NaN/jump/reconnect/re-anchor/U hold/cleanup, 실제 SH5
IK, 기존 HX5 retarget, O/P hold semantics, U latch/heartbeat, camera timestamp/config,
motor-module import guard, episode preflight, action-only replay, HX5 좌우 joint 순서,
Rerun state/target/sent/live 차이 계산과 camera label을 검사한다.

현재 host 결과: `Ran 92 tests ... OK`, skip 0. 기존 SG2 virtual-leader fixture도
실제 sibling `/home/kimm/Downloads/ai_worker`와 legacy `external_repos/ai_worker`
layout을 모두 찾도록 보완해 세 IK 회귀가 실제 URDF로 실행됐다. 현재 PoC의 SH5
IK도 command-sink 테스트와 전체 synthetic 실행에서 실제로 수행됐다.

위 결과는 loopback UDP socket 생성이 허용된 host 실행 기준이다. 격리 runner가
`AF_INET/SOCK_DGRAM` 생성을 차단하면 heartbeat/E-stop receiver 두 테스트가
`PermissionError: Operation not permitted` 환경 오류로 끝날 수 있다. release PASS는
그 격리 오류를 무시하는 것이 아니라 host에서 위 명령이 종료 코드 0인지로 판정한다.

## Vision Pro 없이 실행하는 완전한 no-hardware 경로

다음 명령은 합성 head/wrist/finger와 합성 카메라를 사용한다. 실제
`AIWorkerArmIK.solve_ik()`와 기존 HX5 geometric retargeter까지 실행하지만 arm,
hand, neck, lift, base publisher와 pedal UDP를 만들지 않고 5초 후 종료한다.

```bash
cd /home/kimm/Downloads/xr_tele
/home/kimm/miniforge3/envs/tv/bin/python teleop/teleop_hand_and_arm.py \
  --arm AI_WORKER \
  --ee hx5_d20 \
  --xr-backend visionproteleop \
  --visionpro-synthetic \
  --command-sink \
  --command-sink-duration 5 \
  --command-sink-log-rate 1 \
  --image-source none \
  --hx5-d20-retarget-mode geometric \
  --visionpro-status-rate 1 \
  --visionpro-settle-time 0.1 \
  --arm-found-confirm 0.1 \
  --no-record \
  --no-enable-neck
```

PASS 기준:

- `REANCHOR_REQUIRED` 다음 `TRACKING_OK`
- `tracking_hz` 약 60, `video_input_hz` 약 30. 이 synthetic 실행에는 실제 WebRTC
  consumer가 없으므로 `video_callback_hz=0`이어도 정상이며 live PASS로 해석하지 않는다.
- `invalid=0 rejected_jump=0`
- `hand_joints=(L25,R25)`이고 `hand_nonzero`가 양쪽 모두 0보다 큼
- `[command sink] ai_worker_arms ... target=[...]`와
  `[command sink] hx5_d20 ... target=[...]`가 시간에 따라 변함
- `no motor DDS transport was created`
- `duration reached`, `skip ctrl_dual_arm_go_home`, 종료 코드 0

FAIL: `TRACKING_STALE/HOLD`에 계속 머묾, IK/target 예외, nonzero 종료, 또는
production motor controller 생성 로그가 보임.

## 실제 AI Worker 카메라 publisher와 read-only 확인

SH5의 camera launch는 ZED head camera를 즉시, 두 RealSense wrist camera를
10초 뒤 시작한다. ROS 2와 `ffw_bringup`이 설치·source된 **AI Worker ROS host**의
터미널에서 실행한다.

```bash
export ROS_DOMAIN_ID=30
ros2 launch ffw_bringup camera.launch.py \
  head_camera_type:=zed \
  auto_assign_cameras:=false
```

이 개발 environment에는 `ros2` 실행 파일과 built
`ai_worker/install/setup.bash`가 없으므로 publisher launch는 **NOT EXECUTED**다.
소스 launch file은 `/home/kimm/Downloads/ai_worker/ffw_bringup/launch/camera.launch.py`
에서 확인했다.

카메라가 10초 이상 실행된 뒤 `xr_tele` host에서 다음 read-only 진단을 실행한다.
이 명령은 세 camera reader만 만들고 motor publisher는 만들지 않는다.

```bash
cd /home/kimm/Downloads/xr_tele
/home/kimm/miniforge3/envs/tv/bin/python -m teleop.ai_worker_camera_diagnostic \
  --domain 30 \
  --duration 5 \
  --freshness 0.5
```

확인하는 topic은 정확히 다음 세 개다.

```text
/zed/zed_node/left/image_rect_color/compressed
/camera_left/camera_left/color/image_rect_raw/compressed
/camera_right/camera_right/color/image_rect_raw/compressed
```

PASS: 세 줄 모두 `status=OK`, 양수 `fps`, `age<=0.500s`, 유효한
`frame_shape`이고 마지막 줄이 `PASS`. FAIL: `MISSING`, `INVALID`, `STALE`,
또는 `SETUP_ERROR`. 현재 host read-only 실행에서는 DDS reader가 정상 생성됐지만
세 topic이 모두 `status=MISSING`, 0 fps로 종료 코드 1이었다. 위 ROS camera launch를
이 PC에서 실행할 수 없고 실제 publisher가 없으므로 live composite 검증은 그
publisher가 제공될 때까지 막혀 있다.

## stock Tracking Streamer 설정

1. Vision Pro와 Linux host를 같은 Wi-Fi/LAN에 연결한다.
2. App Store의 **Tracking Streamer**를 열고 첫 화면의 `Local Network` 카드에서
   같은 LAN interface의 **IPv4**와 `gRPC Server Ready`를 확인한다. 현재 stock
   UI에는 별도 “local mode” toggle이 없다. Python에 dotted IPv4를 주는 것이
   local-network 선택이다. IPv6를 주면 upstream이 room code로 분류하므로 쓰지 않는다.
3. Linux terminal에서 IPv4, route/local address, AVP의 gRPC TCP 12345를 확인한다.

```bash
read -r -p 'Tracking Streamer Local Network 카드의 Vision Pro IPv4: ' VISIONPRO_IP
export VISIONPRO_IP
python -c 'import ipaddress, os; print("vision_pro_ipv4=", ipaddress.IPv4Address(os.environ["VISIONPRO_IP"]))'
python -c 'import os, socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect((os.environ["VISIONPRO_IP"], 1)); print("linux_ip=", s.getsockname()[0]); s.close()'
python -c 'import os, socket; s=socket.create_connection((os.environ["VISIONPRO_IP"], 12345), timeout=3); print("avp_grpc_tcp_12345=reachable"); s.close()'
```

PASS: 세 command가 모두 종료 코드 0이고 출력된 `linux_ip`가 AVP와 같은 LAN
route다. FAIL: IPv4 parse error, `No route`, timeout/refused. Linux→AVP TCP 12345,
AVP→Linux TCP 9999와 WebRTC ICE/RTP의 동적 UDP가 host firewall/AP
client-isolation에 막히지 않아야 한다. Python은 TCP 8888에도 backup-info
endpoint를 열지만 확인한 stock 앱 commit에서는 이를 호출하는 경로가 없어 자동
fallback/필수 방화벽 경로로 간주하지 않는다. 방화벽을 끄지 말고 같은 LAN
interface와 이 process에 필요한 traffic만 허용한다.

4. 앱에서 `START`를 누른다. 최초 cloud 안내가 나오면 이 PoC에서는 `Skip for
   now` 뒤 `Get Started` 또는 `Remind me later`를 한 번 더 누른다. `Don't ask
   again`을 선택하면 바로 다음 단계로 진행한다. 앱은 `combinedStreamSpace`를
   열며 현실 방이 보이는 default mixed passthrough 상태가 된다.
5. immersive status panel 상단 mode toggle에서 `Teleoperation`을 선택한다.
   persisted mode가 `EgoRecord`이면 `Network Stream`이 disabled된다.
6. gear → `Recording`에서 **Auto-Record를 OFF**로 바꾼다. Python의
   `record=False`와 xr_tele의 `--no-record`만으로는 Vision Pro 앱 자체의 자동
   녹화를 끄지 못한다.
7. gear → `Video Source` → `Network Stream
   (WebRTC from Python)`을 선택한다.
8. gear → `Video Plane`에서 Size, Distance, Height를 조절한다.
   `Lock To World`를 켜면 방에 고정되고 끄면 head-follow가 된다.
9. 아직 Linux teleop process를 시작하지 않았으므로 앱의 `Python`/`WebRTC`가
   `Waiting...`인 것이 정상이다. Linux host IP를 앱에 직접 입력하는 UI는 없다.
   다음 절의 Python process가 Vision Pro gRPC server에 연결하면 앱이 peer IP를
   표시하고 WebRTC negotiation을 시작한다.

이 단계의 PASS는 실제 방 passthrough가 계속 보이고 `Teleoperation`/`Network
Stream`이 선택 가능하고 Auto-Record가 OFF이며 TCP 12345 preflight가 성공하는
것이다. camera plane과 두 connection status의 최종 PASS는 Python process를
시작한 다음 판정한다.

## 실제 Vision Pro + camera + command sink 실행

카메라 진단이 PASS하고 `VISIONPRO_IP`를 export한 같은 Linux shell에서 실행한다.
아래는 foreground process를 유지하면서 이 process 하나만 가리키는 PID를 남긴다.

```bash
cd /home/kimm/Downloads/xr_tele
bash -c 'echo $$ > /tmp/xr_tele_visionproteleop.pid; exec /home/kimm/miniforge3/envs/tv/bin/python teleop/teleop_hand_and_arm.py \
  --arm AI_WORKER \
  --ee hx5_d20 \
  --xr-backend visionproteleop \
  --visionpro-ip "$VISIONPRO_IP" \
  --command-sink \
  --command-sink-log-rate 1 \
  --image-source robotis_dds \
  --ai-worker-ros-domain-id 30 \
  --camera head_and_wrist \
  --hx5-d20-retarget-mode geometric \
  --visionpro-status-rate 1 \
  --no-record \
  --no-enable-neck \
  --no-ai-worker-home-on-start \
  --skip-arm-go-home-on-exit'
```

시작 직후 앱의 `Python`/`WebRTC` card가 `Waiting...`에서 connected로 바뀌어야
한다. 별도 Linux terminal에서는 primary local signaling listener도 확인한다.

```bash
ss -ltn 'sport = :9999'
```

PASS: `:9999`가 `LISTEN`이고 앱에 Python peer IP가 나타난다. 앱의 local WebRTC
card는 server info 수신 여부에 기반하므로 그 card만으로 media 성공을 판정하지
않는다. 움직이는 plane과 아래의 양수 `video_callback_hz`가 실제 frame-pull PASS다.
TCP 8888 endpoint는 이 stock app commit에서 자동 fallback으로 호출되지 않으므로
PASS 조건이 아니다.

terminal의 `[VisionProTeleop prestart]`에서 `state=REANCHOR_REQUIRED`,
`session_alive=True`, `settling_complete=True`, 낮은 `fresh_ms`와 양수
`tracking_hz`를 확인한 뒤 `r`을 한 번 누른다. 기본 native safety 값은 tracking
timeout 0.25 s, continuous settle 0.5 s, wrist translation jump 0.15 m, rotation
jump 60 deg, velocity 3 m/s, angular velocity 8 rad/s다. 다음을 확인한다.

- `state=TRACKING_OK`, 낮은 `fresh_ms`, 양수 `tracking_hz`
- `head_xyz_m`, `head_yaw_rad`, `wrist_xyz_m=(L...,R...)`가 유한함
- `hand_joints=(L25,R25)`, 양수 `hand_nonzero`, 두 `pinch_cm` 값이 보임
- `video_input_hz`와 `video_callback_hz`가 모두 양수이고, 후자는 기본
  `--viewer-display-fps 15` 부근이며 Vision Pro plane의 세-camera composite가
  실제로 계속 움직임
- command-sink의 14-DoF arm target과 좌/우 20-DoF HX5 target이 변함
- arm/HX5 모두 `memory-only`, neck disabled
- 앱의 실제 방 passthrough가 유지되고 `Python`과 `WebRTC`가 connected. process
  시작 뒤에도 둘 중 하나가 계속 `Waiting...`이거나 callback이 0이면 FAIL

movement checklist:

1. 정면 기준 head yaw를 왼쪽으로 돌리면 `head_yaw_rad`가 `+`, 오른쪽이면 `-`로
   연속적으로 변해야 한다.
2. 왼쪽 손목과 오른쪽 손목을 하나씩 움직인다. 물리적 forward/back은 해당
   `wrist_xyz_m`의 X가 `+/-`, left/right는 Y가 `+/-`, up/down은 Z가 `+/-`로
   변하고 반대 손 값과 섞이지 않아야 한다.
3. 양손을 각각 open/close: 해당 HX5 20-DoF target만 먼저 변해야 한다.
4. 양쪽 thumb-index pinch: 해당 `pinch_cm`이 줄어야 한다.
5. camera plane: head가 위, left/right wrist가 아래 각 위치에 있고 움직임이
   멈추지 않아야 한다.
6. Wi-Fi를 잠시 끊는다: timeout 뒤 `TRACKING_STALE`, validity false, 마지막
   arm/hand target hold여야 한다.
7. 다시 연결한다: `REANCHOR_REQUIRED`이며 자동 재개하지 않아야 한다. 안정된
   자세를 취하고 `r`을 다시 눌러야만 `TRACKING_OK`가 된다.

`fresh_ms`는 Linux가 마지막 새 packet을 관찰한 monotonic age이지 end-to-end
transport latency가 아니다. stock protocol에는 그 latency를 계산할 일반 tracking
source timestamp가 없다.

## U hold를 actuator 없이 확인

위 live command-sink를 실행한 채 별도 terminal에서 pedal 장치를 먼저 확인한다.

```bash
cd /home/kimm/Downloads/xr_tele
/home/kimm/miniforge3/envs/tv/bin/python -m teleop.ai_worker_pedal_teleop --list-devices
/home/kimm/miniforge3/envs/tv/bin/python -m teleop.ai_worker_pedal_teleop \
  --domain-id 30 \
  --dry-run
```

첫 command의 PASS는 하나 이상의 `/dev/input/eventN`이 출력되는 것이다. 빈 출력은
pedal 미검출이고, 두 번째 command의 permission error는 해당 event device read 권한
문제다.

현재 host에서는 `/dev/input/event20`, `/dev/input/event23`이 검출됐고 두 번째
command가 `held_wasd=NONE motion=STOP lift=HOLD upper_body_estop=OFF
tracking_allow=False`를 출력한 뒤 Ctrl+C 종료 코드 0으로 정리됐다. 이 확인 중에는
실제 pedal key를 누르지 않았으므로 아래 O/P/U movement checklist는 아직 수동 확인
항목이다.

`--dry-run`은 base/lift DDS publisher를 만들지 않지만 U toggle UDP는 보낸다. 이
command-sink 조합은 base/lift safety heartbeat를 의도적으로 내보내지 않으므로
`tracking_allow=False`, `motion=STOP`이 정상이다. 대신 raw pedal 상태를 별도로
표시하므로 W/A/S/D를 누르고 있는 동안 `held_wasd=w/a/s/d`, 놓으면 `NONE`인지
확인한다. O를 누르고 있는 동안 `lift=UP`, P는 `lift=DOWN`, 키를 놓거나 O+P를
동시에 누르면 `lift=HOLD`여야 한다. 이 표시는 hold/release semantics만 검증하며
actuator publish를 검증하지 않는다.

U를 한 번 누르면 pedal terminal에 `upper_body_estop=ON`, teleop backend에 `HOLD`가
나오고 arm/HX5 target이 hold되어야 한다. 다시 U를 누르면 `OFF`가 나오지만 자동
재개하지 않으며 `REANCHOR_REQUIRED`와 `settling_complete=True`를 확인하고 새로
`r`을 눌러야 한다. U는 safety-rated hardware E-stop이 아니다.

## 종료와 이 process만 대상으로 한 비상 종료

정상 종료는 teleop terminal의 `q` 또는 `Ctrl+C`다. 종료 시 command sink는
arm home motion을 보내지 않고 native wrapper와 camera readers를 닫는다.

다른 terminal에서 이 실행 하나만 확인하고 interrupt하려면:

```bash
TELEOP_PID="$(cat /tmp/xr_tele_visionproteleop.pid)"
ps -p "$TELEOP_PID" -o pid=,args=
kill -INT "$TELEOP_PID"
```

`ps` 출력이 위 command의 `teleop_hand_and_arm.py`인지 반드시 확인한다. 종료하지
않을 때만 같은 PID에 `kill -TERM "$TELEOP_PID"`를 사용한다. `pkill python`,
`killall`, ROS 전체 종료 명령은 사용하지 않는다.

## 감독하 실기 진입 경계 — NOT EXECUTED

다음은 향후 감독하에 사용할 제안 명령일 뿐 이 작업에서는 실행하지 않았다.
`--command-sink`를 제거하고 `--allow-real-hardware`를 추가하면 곧바로 motor
controller를 만들지 않고 native **pre-arm gate**에 들어간다. 이 gate에서는 camera
reader, tracking/video와 U receiver만 활성화되고 로그에
`motor_controllers=absent`가 보여야 한다. `state=REANCHOR_REQUIRED`,
`session_alive=True`, `settling_complete=True`, 낮은 `fresh_ms`, `U_hold=False`를
모두 확인한 뒤 누른 새 `r`이 accepted되어야만 AI Worker arm/HX5 controller와
motor DDS publisher construction이 허용된다. 조건 전의 `r`은 거부·소비되므로
조건 회복 뒤 다시 눌러야 한다.

아래 sensitivity 값은 parser에 존재하는 wrist-input 축소/범위 제안값이지 검증된
hardware 속도 한계가 아니다. `--arm-smoothing-alpha` low-pass는 wrist translation에만
적용되고 orientation에는 `--arm-rot-gain`만 적용된다. AI Worker controller의 별도
3 rad/s cap과 기존 HX5 20-DoF retarget/controller semantics는 그대로다.

```bash
cd /home/kimm/Downloads/xr_tele
/home/kimm/miniforge3/envs/tv/bin/python teleop/teleop_hand_and_arm.py \
  --arm AI_WORKER \
  --ee hx5_d20 \
  --xr-backend visionproteleop \
  --visionpro-ip "$VISIONPRO_IP" \
  --allow-real-hardware \
  --image-source robotis_dds \
  --ai-worker-ros-domain-id 30 \
  --camera head_and_wrist \
  --hx5-d20-retarget-mode geometric \
  --arm-pos-gain-x 0.25 \
  --arm-pos-gain-y 0.25 \
  --arm-pos-gain-z 0.25 \
  --arm-rot-gain 0.25 \
  --arm-max-delta 0.05 \
  --arm-smoothing-alpha 0.15 \
  --ai-worker-command-duration 0.12 \
  --arm-startup-duration 4.0 \
  --arm-startup-max-step 0.02 \
  --no-record \
  --no-enable-neck \
  --no-ai-worker-home-on-start \
  --skip-arm-go-home-on-exit
```

그 전에 command-sink에서 전체 movement/network/U checklist를 PASS하고, U 송신용
pedal process를 별도 terminal에서 `--dry-run`으로 계속 실행한다. 이 첫 arm/hand
bringup에서는 base/lift publisher를 열지 않는다. 로봇을 지지대에 고정하고 작업
반경을 비우고 별도의 실제 hardware E-stop 담당자를 배치한다. hands는 움직이지
않은 채 한 wrist의 작은 변위부터 확인한다. native backend에서는 neck가 code에서도
강제로 disabled된다. software U는 hardware E-stop을 대체하지 않는다.

## 알려진 제한

- 이 environment에서는 Vision Pro/App Store 앱, 실제 passthrough, WebRTC,
  실제 camera DDS를 실행하지 못했다. 따라서 live 단계는 **NOT EXECUTED**다.
- 이 environment에는 `rerun`과 `loop_sdk`가 설치되어 있지 않다. AI Worker replay
  preflight/series/camera label과 `ai_worker.*` Loop schema는 전체 unit test에서
  검증했지만 Rerun viewer와 실제 Config Loop consumer 연결은 **NOT EXECUTED**다.
  consumer가 legacy `g1.*`만 선언한다면 `ai_worker.*` channel 추가가 필요하다.
- stock normal tracking payload에는 ARKit `isTracked`, source sequence, 일반 pose
  source timestamp가 없다. adapter는 새 raw object의 host arrival identity,
  monotonic timeout, finite/rigid/jump/velocity 검사로 fail closed하지만, 정지한
  손과 ARKit이 cached pose를 계속 보내는 tracking loss를 구별한다고 주장하지
  않는다.
- production HX5 구현은 요구대로 수정하지 않았다. native 실기 종료 시에는 main이
  먼저 publish worker를 pause한 뒤 종료를 확인하고 transport만 닫아 기존 `stop()`의
  zero/open 명령을 피한다. 다만 runtime U/stale가 들어온 바로 그 순간 worker가 이미
  한 publish iteration을 시작했다면, public synchronized barrier가 없는 기존 HX5
  구조상 그 한 iteration(좌/우 사이 포함)이 끝날 수 있다. 따라서 U는 여전히
  software hold이고 안전 등급 hardware E-stop이 아니다.
- upstream constructor는 첫 tracking sample까지 block할 수 있어 daemon startup
  worker에서 실행한다. upstream cleanup이 내부 gRPC/HTTP/WebRTC thread를 항상
  완전히 회수하지 못할 수 있으므로, live 재시작은 기존 Python process를 종료한
  뒤 새 process로 시작하는 것을 경계로 삼는다.
- 첫 PoC는 mono composite plane 하나뿐이며 stereo, 독립 panel 세 개, spatial
  registration, Swift/protobuf 변경은 포함하지 않는다.
- Vuer는 default(`--xr-backend vuer`)로 유지되고 `avp_stream`을 import하지 않는다.
  AI Worker/HX5 production classes와 Config Loop/recording 경로도 교체하지 않았다.
