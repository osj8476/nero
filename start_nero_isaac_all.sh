#!/bin/bash
# ============================================================================
# start_nero_isaac_all.sh
#
# Isaac Sim + MoveIt2/ROS2 스택 전체를 tmux 없이 한 번에 띄우는 스크립트.
# 백그라운드 job + trap 방식 (tmux 세션 관리 없이, 이 스크립트를 실행한
# 터미널 하나에서 전부 관리됨).
#
# 이 스크립트가 켜는 것 (순서대로):
#   0. Isaac Sim (USD 로드, --open 만 함 — ▶ 재생은 GUI에서 직접 눌러야 함)
#   1. robot_state_publisher (rsp)
#   2. static_virtual_joint_tfs (world -> base_link)
#   3. ros2_control_node (topic_based_ros2_control, Isaac Sim과 브릿지)
#   4. spawn_controllers (1회 실행, arm/gripper/joint_state_broadcaster)
#   5. move_group
#   6. moveit_rviz (RViz)
#   7. 카메라 정적 TF (gripper_base -> camera_color_optical_frame)
#
# 이 스크립트가 켜지 않는 것 (직접 별도 터미널에서 실행):
#   - Claude Code CLI (claude mcp add로 등록된 mcp_robot_server는 CLI가 알아서 실행)
#   - perception_node_sim / planning_node / 박스 서버 / mcp_robot_server
#     (원하면 아래 "선택 사항" 섹션 주석 해제해서 포함 가능)
#
# 종료: 이 스크립트가 실행 중인 터미널에서 Ctrl+C 한 번
#       -> trap이 모든 백그라운드 프로세스(Isaac Sim 포함)를 정리함
#
# 사용 전 꼭 확인/수정할 것 (환경마다 다를 수 있음):
#   - ISAACSIM_PATH, USD_PATH
#   - ROS_DISTRO_SETUP, WS_SETUP 경로
#   - MOVEIT_CFG_DIR 안의 xacro/yaml 경로
#   - 카메라 정적 TF 값 (hand-eye calibration 결과, ~/.nero_calib/flange_to_camera.json 참고)
#
# 사용법:
#   chmod +x start_nero_isaac_all.sh
#   ./start_nero_isaac_all.sh
# ============================================================================

set -e

# ── 환경 설정 (필요시 이 값들만 수정) ──────────────────────────────────────
ISAACSIM_PATH="$HOME/isaacsim/isaac-sim.sh"
USD_PATH="/home/bpdl/nero/nero_simul.usd"

ROS_DISTRO_SETUP="/opt/ros/humble/setup.bash"
WS_SETUP="/home/bpdl/ros2_ws/install/setup.bash"

MOVEIT_PKG="nero_gripper_moveit_config"
MOVEIT_CFG_DIR="/home/bpdl/ros2_ws/src/agx_arm_sim/Moveit2/${MOVEIT_PKG}/config"
XACRO_FILE="${MOVEIT_CFG_DIR}/nero.urdf.xacro"
CONTROLLERS_YAML="${MOVEIT_CFG_DIR}/ros2_controllers.yaml"

# 카메라 정적 TF (hand-eye calibration 결과값 — 최신 값으로 갱신 필요시 여기 수정)
CAM_X=0.073089; CAM_Y=0.026438; CAM_Z=0.038050
CAM_ROLL=-0.350796; CAM_PITCH=-0.000000; CAM_YAW=1.569204
CAM_PARENT="gripper_base"; CAM_CHILD="camera_color_optical_frame"

# ── 박스 위치 랜덤화 설정 (2026-07 추가) ──────────────────────────────────
# 실행할 때마다 TestBox/TestBox1/TestBox2를 안전 영역 내 새 랜덤 위치로
# 재배치. auto_stack_demo.py와 짝을 이뤄서 "매번 다른 배치에서도 되는지"
# 자동 반복 검증할 때 씀. 순수 usd 파일 수정이라 Isaac Sim을 (재)여는
# 시점에만 반영됨 -- 그래서 safety map 갱신과 마찬가지로 Isaac Sim 열기
# "직전"에 실행해야 함.
RANDOMIZE_BOXES=true   # 매번 랜덤 배치로 검증하고 싶으면 true로
RANDOMIZE_SCRIPT="/home/bpdl/nero/randomize_box_positions.py"
RANDOMIZE_SEED=""       # 비워두면 매번 다른 랜덤, 재현하려면 숫자 지정 (예: 42)

# ── 안전지도(도달가능/불일치 맵) 자동 갱신 설정 ──────────────────────────
# 메인 usd에는 embed_reference_once.py로 SAFETY_MAP_USD를 참조(Reference)로
# 한 번만 심어두면 됨. 이후로는 아래 CSV/스크립트로 SAFETY_MAP_USD 파일
# 내용만 매번 새로 덮어써도 Isaac Sim이 열 때마다 자동으로 최신 내용을 읽음.
# build_safety_map_layer.py는 순수 OpenUSD(pip usd-core)만 쓰므로 Isaac Sim의
# python.sh가 아니라 일반 python3을 씀 (pip install --user usd-core 로 설치됨).
# venv를 따로 만드셨다면 그 venv의 python 경로로 바꿔도 됨.
SAFETY_MAP_PYTHON="python3"
SAFETY_MAP_BUILD_SCRIPT="/home/bpdl/nero/build_safety_map_layer.py"
SAFETY_MAP_USD="/home/bpdl/nero/safety_map_layer.usd"

# ★ 지도 종류를 바꾸고 싶으면 이 한 줄만 수정 ★
#   "mismatch"      : sim/실물 컨벤션 불일치 지도 (summary csv 필요)
#   "sim_reachable" : sim 기준 IK 해 존재 지도   (raw csv 필요)
SAFETY_MAP_SOURCE="mismatch"

# CSV가 들어있는 폴더 (파일명이 실행마다 바뀌므로, 폴더 안에서 SAFETY_MAP_SOURCE에
# 맞는 가장 최근 CSV를 자동으로 찾아 씀). 폴더 구조가 다르면 이 경로만 수정.
SAFETY_MAP_CSV_DIR="/home/bpdl/nero"

# SAFETY_MAP_SOURCE에 맞춰 찾을 CSV 패턴이 자동으로 정해짐 (따로 손댈 필요 없음)
if [ "$SAFETY_MAP_SOURCE" = "sim_reachable" ]; then
    SAFETY_MAP_CSV_PATTERN="*mismatch_raw_*.csv"
else
    SAFETY_MAP_CSV_PATTERN="*mismatch_summary_*.csv"
fi

ENABLE_SAFETY_MAP=true   # 지도 자동 갱신을 끄고 싶으면 false로

# 공통 source 커맨드
SRC_CMD="source ${ROS_DISTRO_SETUP} && source ${WS_SETUP}"

# 로그를 남길 디렉토리 (각 노드 stdout/stderr 확인용)
LOG_DIR="/tmp/nero_isaac_logs"
mkdir -p "$LOG_DIR"

PIDS=()
cleanup() {
    echo ""
    echo "🛑 전체 종료 중... (${PIDS[*]})"
    kill "${PIDS[@]}" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "✅ 정리 완료. 로그는 $LOG_DIR 에 남아있습니다."
}
trap cleanup EXIT INT TERM

run_bg() {
    # run_bg <로그파일이름> <bash -c로 실행할 커맨드 전체>
    local logname="$1"; shift
    bash -c "$*" > "${LOG_DIR}/${logname}.log" 2>&1 &
    PIDS+=($!)
    echo "  -> PID $! ($logname), 로그: ${LOG_DIR}/${logname}.log"
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " NERO Isaac Sim + MoveIt2 전체 스택 기동"
echo " 종료하려면 이 터미널에서 Ctrl+C 한 번"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$RANDOMIZE_BOXES" = true ]; then
    echo "사전 단계) 박스 위치 랜덤화 — Isaac Sim 열기 전에 usd 안 박스 좌표 새로 씀"
    SEED_ARG=""
    if [ -n "$RANDOMIZE_SEED" ]; then
        SEED_ARG="--seed $RANDOMIZE_SEED"
    fi
    python3 "$RANDOMIZE_SCRIPT" --main-usd "$USD_PATH" $SEED_ARG \
        > "${LOG_DIR}/0_randomize_boxes.log" 2>&1 \
        && echo "   -> 박스 재배치 완료 (로그: ${LOG_DIR}/0_randomize_boxes.log)" \
        || echo "   ⚠️ 박스 재배치 실패 (로그: ${LOG_DIR}/0_randomize_boxes.log) — 이전 위치 그대로 진행합니다."
fi

if [ "$ENABLE_SAFETY_MAP" = true ]; then
    echo "사전 단계) 안전지도(safety map) 갱신 — Isaac Sim 열기 전에 usd 레이어 새로 씀"
    # 폴더(하위 폴더 포함) 안에서 패턴에 맞는 가장 최근(mtime 기준) CSV 자동 탐색
    SAFETY_MAP_CSV="$(find "$SAFETY_MAP_CSV_DIR" -type f -name "$SAFETY_MAP_CSV_PATTERN" -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -n1 | cut -d' ' -f2-)"
    if [ -n "$SAFETY_MAP_CSV" ] && [ -f "$SAFETY_MAP_CSV" ]; then
        echo "   -> 사용할 CSV: $SAFETY_MAP_CSV"
        "$SAFETY_MAP_PYTHON" "$SAFETY_MAP_BUILD_SCRIPT" \
            --source "$SAFETY_MAP_SOURCE" \
            --csv "$SAFETY_MAP_CSV" \
            --out "$SAFETY_MAP_USD" \
            > "${LOG_DIR}/0_safety_map_build.log" 2>&1 \
            && echo "   -> 지도 갱신 완료: $SAFETY_MAP_USD" \
            || echo "   ⚠️ 지도 갱신 실패 (로그: ${LOG_DIR}/0_safety_map_build.log) — 지도 없이 계속 진행합니다."
    else
        echo "   ⚠️ '$SAFETY_MAP_CSV_DIR' 안에서 '$SAFETY_MAP_CSV_PATTERN' 패턴의 CSV를 못 찾았습니다 (건너뜀)"
    fi
fi

echo "0) Isaac Sim 실행 + USD 로드"
run_bg "0_isaacsim" "'$ISAACSIM_PATH' --open '$USD_PATH' --ext-folder $HOME/Documents/isaac-sim-mcp/ --enable isaac.sim.mcp_extension"
echo "   Isaac Sim 창이 뜨면 반드시 ▶ 재생 버튼을 눌러주세요 (ActionGraph는 재생 중에만 동작)."
sleep 25   # Isaac Sim 자체 부팅이 오래 걸림 — 환경에 따라 늘려야 할 수 있음

echo "1) robot_state_publisher"
run_bg "1_rsp" "${SRC_CMD} && ros2 launch ${MOVEIT_PKG} rsp.launch.py"
sleep 3

echo "2) static_virtual_joint_tfs"
run_bg "2_static_tf" "${SRC_CMD} && ros2 launch ${MOVEIT_PKG} static_virtual_joint_tfs.launch.py"
sleep 2

echo "3) ros2_control_node"
run_bg "3_ros2_control" "${SRC_CMD} && ros2 run controller_manager ros2_control_node --ros-args \
    -p robot_description:=\"\$(xacro ${XACRO_FILE})\" \
    --params-file ${CONTROLLERS_YAML}"
sleep 5

echo "4) spawn_controllers"
run_bg "4_spawn_ctrl" "${SRC_CMD} && ros2 launch ${MOVEIT_PKG} spawn_controllers.launch.py"
sleep 5

echo "5) move_group"
run_bg "5_move_group" "${SRC_CMD} && ros2 launch ${MOVEIT_PKG} move_group.launch.py"
sleep 10

#echo "6) RViz"
#run_bg "6_rviz" "${SRC_CMD} && ros2 launch ${MOVEIT_PKG} moveit_rviz.launch.py"
#sleep 3


echo "6) 카메라 정적 TF"
run_bg "7_camera_tf" "${SRC_CMD} && ros2 run tf2_ros static_transform_publisher \
    --x ${CAM_X} --y ${CAM_Y} --z ${CAM_Z} \
    --roll ${CAM_ROLL} --pitch ${CAM_PITCH} --yaw ${CAM_YAW} \
    --frame-id ${CAM_PARENT} --child-frame-id ${CAM_CHILD}"
    
echo "7) 시각화 (visualize_3d_bpdl)"
run_bg "8_visualize" "${SRC_CMD} && python3 ~/nero/visualize_3d_bpdl.py"
sleep 2

# ── 선택 사항: perception/planning/박스서버까지 여기서 같이 켜고 싶으면 주석 해제 ──
# [2026-08-19] vlm_boxyolo.py가 듀얼 모델 서버(box 커스텀 + COCO 25종)로
# 바뀌면서 인자명이 --conf → --conf-box/--conf-coco로 변경됨. 옛 --conf 인자
# 그대로 두면 argparse가 즉시 에러내며 죽는다(실측 확인, 스크래치 포트 8011
# 검증 결과 box 검출 정확도는 기존과 100% 동일 — conf_box=0.75 유지로 회귀 없음).
echo "8) 박스 서버 (듀얼: box 커스텀 + COCO 25종)"
run_bg "8_box_server" "python3 ~/nero/yolo/vlm_boxyolo.py --port 8002 --model ~/nero/yolo/best.pt --model-coco yolov8n.pt --conf-box 0.75 --conf-coco 0.25"
sleep 3

echo "9) perception_node_sim"
run_bg "9_perception" "${SRC_CMD} && ros2 run sj_pickplace perception_node_sim"
sleep 3

#echo "10) planning_node"
#run_bg "10_planning" "${SRC_CMD} && ros2 run sj_pickplace planning_node --ros-args -p use_moveit2:=true"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 전체 기동 완료. 확인용 명령어:"
echo "   ros2 node list"
echo "   ros2 topic list | grep -E 'joint_states|controller|camera'"
echo "   tail -f ${LOG_DIR}/5_move_group.log   # move_group 로그 실시간 확인"
echo ""
echo "Claude Code와 planning node 는 별도 터미널에서 직접 실행하세요."
echo "종료: 이 터미널에서 Ctrl+C"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

wait
