#!/usr/bin/env bash
# bringup_with_camera.sh
#
# 실물 로봇 + RViz + 카메라 고정 TF + perception_node 를 터미널 한 번에
# 기동. 각각 다른 터미널에서 띄우던 걸 백그라운드 job + trap 으로 관리.
#
# 사전 조건:
#   - hand_eye_calibration.py 로 캘리브레이션 완료 (~/.nero_calib/flange_to_camera.json 존재)
#   - CAN 연결(can0) 및 RealSense USB 연결 확인
#
# 실제 패키지 이름/실행 인자는 환경에 맞게 아래 변수만 고치면 됨.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 환경에 맞게 조정 ────────────────────────────────────────────────
ARM_TYPE="${ARM_TYPE:-nero}"
EFFECTOR_TYPE="${EFFECTOR_TYPE:-agx_gripper}"
CAN_PORT="${CAN_PORT:-can0}"
# perception_node 가 ros2 run 으로 설치된 executable 이면 이 줄로 교체:
#   PERCEPTION_CMD=(ros2 run <package_name> perception_node)
PERCEPTION_CMD=(python3 "$SCRIPT_DIR/../sj_pickplace/perception_node.py")

PIDS=()
cleanup() {
  echo ""
  echo "🛑 종료 중... (${PIDS[*]})"
  kill "${PIDS[@]}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "1) 로봇 + RViz 기동 (arm_type=$ARM_TYPE effector_type=$EFFECTOR_TYPE can_port=$CAN_PORT)"
ros2 launch agx_arm_ctrl start_single_agx_arm_rviz.launch.py \
  arm_type:="$ARM_TYPE" effector_type:="$EFFECTOR_TYPE" can_port:="$CAN_PORT" &
PIDS+=($!)
sleep 5   # CAN 연결 / robot_state_publisher 안정화 대기

echo "2) 카메라 고정 TF 퍼블리시 (tcp_link -> camera_color_optical_frame)"
"$SCRIPT_DIR/publish_camera_tf.sh" &
PIDS+=($!)
sleep 1

echo "3) perception_node 기동"
"${PERCEPTION_CMD[@]}" &
PIDS+=($!)

echo ""
echo "✅ 모두 기동됨. 종료하려면 Ctrl+C 한 번."
echo "   확인용: ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame"
echo "          ros2 topic echo /detected_objects"

wait
