#!/usr/bin/env bash
# ============================================================================
# yolo_server.sh  —  THOR 로컬. vlm_boxyolo YOLO 추론 서버 (:8002).
#   box → best.pt (conf 0.75) / COCO 25종 → yolov8n.pt (conf 0.25)
#   PC 의 perception_node_sim 이 BOX_SERVER_URL=http://<thor>:8002/detect 로 호출.
#
#   tmux 안 씀 — setsid + PGID (thor_stack.sh 와 같은 방식).
#
# 사용: ~/grasp/yolo_server.sh {up|down|status|log}
# ============================================================================
set -eo pipefail
DIR="$HOME/grasp/yolo"
PY="$HOME/grasp/cgn_venv/bin/python3"
LOG="$HOME/grasp/logs/yolo_server.log"
PGIDFILE="$HOME/grasp/.yolo_server.pgid"
PORT=8002
mkdir -p "$HOME/grasp/logs"

_up() {
  if curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT/health"; then
    echo "[yolo] 이미 :$PORT 응답 중"; return 0
  fi
  setsid bash -lc "cd '$DIR' && exec '$PY' vlm_boxyolo.py --host 0.0.0.0 --port $PORT --model best.pt --conf-box 0.75 --model-coco yolov8n.pt --conf-coco 0.25 --sam '$HOME/grasp/mobile_sam.pt'" >"$LOG" 2>&1 &
  child=$!
  sleep 0.5
  ps -o pgid= -p "$child" 2>/dev/null | tr -d ' ' >"$PGIDFILE" || echo "$child" >"$PGIDFILE"
  echo "[yolo] 기동 (PGID $(cat $PGIDFILE)), 로그 $LOG"
  printf '[yolo] /health 대기'
  for i in $(seq 1 60); do
    if curl -s -o /dev/null -m 2 "http://127.0.0.1:$PORT/health"; then echo ' — OK'; break; fi
    printf '.'; sleep 1
    [ "$i" -eq 60 ] && { echo ' — 타임아웃'; tail -n 15 "$LOG" | sed 's/^/  /'; return 1; }
  done
  curl -s "http://127.0.0.1:$PORT/health"; echo
}

_down() {
  [ -f "$PGIDFILE" ] && { pgid=$(cat "$PGIDFILE"); kill -TERM -- "-$pgid" 2>/dev/null || true; sleep 1; kill -KILL -- "-$pgid" 2>/dev/null || true; rm -f "$PGIDFILE"; }
  pkill -9 -f "vlm_boxyolo.py" 2>/dev/null || true
  echo "[yolo] 내림"
}

case "${1:-status}" in
  up) _up ;;
  down) _down ;;
  restart) _down; sleep 1; _up ;;
  status)
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null; then
      echo "[yolo] UP :$PORT  ->  $(curl -s http://127.0.0.1:$PORT/health)"
    else echo "[yolo] DOWN"; fi
    ;;
  log) exec tail -n 40 -f "$LOG" ;;
  *) echo "사용: $0 {up|down|restart|status|log}"; exit 1 ;;
esac
