#!/bin/bash
SESSION="nero"
SJ_DIR="/home/bpdl/sj"
ROS_SETUP="/opt/ros/jazzy/setup.bash"
WS_SETUP="$SJ_DIR/ros2_ws/install/setup.bash"
AGX_SETUP="$SJ_DIR/install/setup.bash"
BEST_PT="$SJ_DIR/ros2_ws/src/nero/yolo/best.pt"
URDF_LAUNCH="$SJ_DIR/nero_cam.launch.py"
CONF=0.85

activate_can() {
    echo "[can] gs_usb 모듈 로드..."
    sudo insmod "$SJ_DIR/can_setup/gs_usb_build/gs_usb.ko" 2>/dev/null || true
    sleep 1
    echo "[can] can4 UP 시도..."
    for i in 1 2 3; do
        sudo ip link set can4 type can bitrate 1000000 2>/dev/null
        sudo ip link set can4 up 2>/dev/null
        STATE=$(ip link show can4 2>/dev/null | grep -o "UP" || echo "DOWN")
        if [ "$STATE" = "UP" ]; then
            sudo ip link set can4 txqueuelen 1000
            echo "[can] can4 UP 성공!"
            return 0
        fi
        echo "[can] 재시도 $i/3..."
        sleep 1
    done
    echo "[can] can4 UP 실패."
    return 1
}

if [ "$1" = "stop" ]; then
    tmux kill-session -t "$SESSION" 2>/dev/null
    pkill -f vlm_boxyolo 2>/dev/null
    echo "[stop] 완료"
    exit 0
fi

if [ "$1" = "can" ]; then
    activate_can; exit $?
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux attach -t "$SESSION"; exit 0
fi

activate_can
echo ""

tmux new-session -d -s "$SESSION" -x 220 -y 50

# 0: agx_arm_ctrl
tmux rename-window -t "$SESSION:0" "robot"
tmux send-keys -t "$SESSION:0" "source $ROS_SETUP && source $AGX_SETUP && sleep 2 && ros2 launch agx_arm_ctrl start_single_agx_arm.launch.py can_port:=can4 arm_type:=nero effector_type:=agx_gripper" Enter

# 1: TF (relay + robot_state_publisher)
tmux new-window -t "$SESSION:1" -n "tf"
tmux send-keys -t "$SESSION:1" "source $ROS_SETUP && source $WS_SETUP && sleep 5 && ros2 run topic_tools relay /feedback/joint_states /joint_states & ros2 launch $URDF_LAUNCH" Enter

# 2: vlm_boxyolo
tmux new-window -t "$SESSION:2" -n "yolo"
tmux send-keys -t "$SESSION:2" "source $ROS_SETUP && /home/bpdl/miniconda3/bin/python3 $SJ_DIR/ros2_ws/src/nero/yolo/vlm_boxyolo.py --port 8002 --model $BEST_PT --conf $CONF" Enter

# PYTHONPATH 수정 (구버전 sj/install 제거)
export PYTHONPATH=/home/bpdl/sj/ros2_ws/install/sj_pickplace/lib/python3.12/site-packages:/home/bpdl/sj/install/agx_arm_ctrl/lib/python3.12/site-packages:/home/bpdl/sj/install/agx_arm_msgs/lib/python3.12/site-packages:/opt/ros/jazzy/lib/python3.12/site-packages
# 3: perception_node
tmux new-window -t "$SESSION:3" -n "perception"
tmux send-keys -t "$SESSION:3" "source $ROS_SETUP && source $SJ_DIR/ros2_ws/install/setup.bash && export PYTHONPATH=/home/bpdl/sj/ros2_ws/install/sj_pickplace/lib/python3.12/site-packages:/home/bpdl/sj/install/agx_arm_ctrl/lib/python3.12/site-packages:/home/bpdl/sj/install/agx_arm_msgs/lib/python3.12/site-packages:/opt/ros/jazzy/lib/python3.12/site-packages && sleep 8 && /home/bpdl/sj/ros2_ws/install/sj_pickplace/lib/sj_pickplace/perception_node" Enter

# 4: planning_node
tmux new-window -t "$SESSION:4" -n "planning"
tmux send-keys -t "$SESSION:4" "source $ROS_SETUP && source $WS_SETUP && export PYTHONPATH=/home/bpdl/sj/ros2_ws/install/sj_pickplace/lib/python3.12/site-packages:/home/bpdl/sj/ros2_ws/install/pymoveit2/lib/python3.13/site-packages:/home/bpdl/sj/install/agx_arm_ctrl/lib/python3.12/site-packages:/home/bpdl/sj/install/agx_arm_msgs/lib/python3.12/site-packages:/opt/ros/jazzy/lib/python3.12/site-packages && sleep 10 && /home/bpdl/sj/ros2_ws/install/sj_pickplace/lib/sj_pickplace/planning_node --ros-args -p use_moveit2:=false" Enter

# 5: mcp_robot_server
tmux new-window -t "$SESSION:5" -n "mcp"
tmux send-keys -t "$SESSION:5" "source $ROS_SETUP && source $WS_SETUP && sleep 12 && /usr/bin/python3 $SJ_DIR/ros2_ws/src/nero/sj_pickplace/mcp_robot_server.py" Enter

# 6: monitor
tmux new-window -t "$SESSION:6" -n "monitor"
tmux send-keys -t "$SESSION:6" "source $ROS_SETUP && sleep 15 && watch -n1 'ros2 node list 2>/dev/null; echo; ros2 topic list 2>/dev/null | grep -E \"detected|joint|camera|feedback\"'" Enter

tmux select-window -t "$SESSION:0"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " NERO 스택 시작 | Ctrl+b+숫자로 창 전환"
echo " 0:robot 1:tf 2:yolo 3:perception 4:planning 5:mcp 6:monitor"
echo " stop: ~/sj/start_nero.sh stop"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

tmux attach -t "$SESSION"
