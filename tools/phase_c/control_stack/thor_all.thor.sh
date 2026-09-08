#!/usr/bin/env bash
# THOR 로컬. 이 프로젝트에 필요한 Thor 서비스 3개를 한 번에.
#   thor_stack (move_group+pick_ik+STOMP)  ·  yolo_server(:8002 YOLO+SAM)  ·  cgn_server(:8010)
# 사용:  ~/grasp/thor_all.sh {up|down|status}
set -uo pipefail
D="$HOME/grasp"
case "${1:-status}" in
  up)   "$D/thor_stack.sh" up; "$D/yolo_server.sh" up; "$D/cgn_server.sh" up ;;
  down) "$D/cgn_server.sh" down; "$D/yolo_server.sh" down; "$D/thor_stack.sh" down ;;
  status)
    "$D/thor_stack.sh" status
    "$D/yolo_server.sh" status
    "$D/cgn_server.sh" status
    ;;
  *) echo "사용: $0 {up|down|status}"; exit 1 ;;
esac
