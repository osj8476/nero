#!/usr/bin/env bash
# ============================================================================
# thor_planning.sh — 데스크탑에서 Thor plan-only 스택 제어 (ssh 얇은 래퍼)
#
# 실제 로직은 Thor 의 ~/grasp/thor_stack.sh 에 있다 (tmux 안 씀, setsid+PGID).
# 이 파일은 데스크탑 셸에서 ssh 한 번으로 부르는 용도.
#
#   ./thor_planning.sh up | down | restart | status | log
#
# Thor 셸에 이미 들어가 있으면 그냥 ~/grasp/thor_stack.sh 를 직접 쓰면 됨.
#
# 환경변수: THOR_SSH  ssh 대상 (기본 "thor")
# ============================================================================
set -eo pipefail
THOR="${THOR_SSH:-thor}"
CMD="${1:-status}"

case "$CMD" in
  up|down|restart|status)
    exec ssh "$THOR" "~/grasp/thor_stack.sh $CMD"
    ;;
  log)
    exec ssh -t "$THOR" "~/grasp/thor_stack.sh log"
    ;;
  *)
    echo "사용: $0 {up|down|restart|status|log}"; exit 1
    ;;
esac
