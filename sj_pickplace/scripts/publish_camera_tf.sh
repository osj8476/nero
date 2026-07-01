#!/usr/bin/env bash
# publish_camera_tf.sh
#
# hand_eye_calibration.py 가 구한 tcp_link -> camera_color_optical_frame
# 고정 변환을 ~/.nero_calib/flange_to_camera.json 에서 읽어
# static_transform_publisher 로 쏴준다.
#
# display.launch.py 가 tcp_link 를 처리하는 방식(tcp_offset 인자 ->
# 런타임 static_transform_publisher)과 동일한 패턴이라, URDF/xacro를
# 따로 수정할 필요가 없다.
#
# 사용법:
#   ./publish_camera_tf.sh
#   (캘리브레이션 파일이 없으면 에러 메시지 내고 종료)

set -e

CALIB_FILE="${NERO_CALIB_DIR:-$HOME/.nero_calib}/flange_to_camera.json"

if [ ! -f "$CALIB_FILE" ]; then
  echo "❌ 캘리브레이션 파일 없음: $CALIB_FILE"
  echo "   먼저 hand_eye_calibration.py 를 실행해서 캘리브레이션부터 끝내세요."
  exit 1
fi

# JSON의 4x4 T_flange_camera(=tcp_link->camera) 행렬 -> xyz + rpy 추출
read -r X Y Z ROLL PITCH YAW <<PYOUT
$(python3 - "$CALIB_FILE" <<'PYEOF'
import sys, json
import numpy as np

path = sys.argv[1]
with open(path) as f:
    data = json.load(f)
T = np.array(data["T_flange_camera"])
xyz = T[:3, 3]
R = T[:3, :3]

sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
if sy > 1e-6:
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], sy)
    yaw = np.arctan2(R[1, 0], R[0, 0])
else:
    roll = np.arctan2(-R[1, 2], R[1, 1])
    pitch = np.arctan2(-R[2, 0], sy)
    yaw = 0.0

print(f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} {roll:.6f} {pitch:.6f} {yaw:.6f}")
PYEOF
)
PYOUT

echo "📐 tcp_link -> camera_color_optical_frame"
echo "   xyz = [$X, $Y, $Z]"
echo "   rpy = [$ROLL, $PITCH, $YAW]"
echo ""

exec ros2 run tf2_ros static_transform_publisher \
  --x "$X" --y "$Y" --z "$Z" \
  --roll "$ROLL" --pitch "$PITCH" --yaw "$YAW" \
  --frame-id tcp_link --child-frame-id camera_color_optical_frame
