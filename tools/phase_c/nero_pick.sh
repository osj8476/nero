#!/usr/bin/env bash
# ============================================================================
# nero_pick.sh  —  한 방 pick 사이클. 명령 하나로 capture → Thor 파이프라인 → exec.
#
#   ./nero_pick.sh                          # label=cup, 안정 pick (시맨틱 ON)
#   ./nero_pick.sh --label bottle
#   ./nero_pick.sh --task "손잡이 잡아서 들어줘"   # 자연어 지시 → 그 파트 pick
#   ./nero_pick.sh --sem-off                # 시맨틱 끄고 기하 전용 (A/B 대조군)
#   ./nero_pick.sh --no-cache               # 시맨틱 캐시 무시하고 VLM 재호출
#   ./nero_pick.sh --dry                    # 계획까지만, exec 안 함
#   ./nero_pick.sh --obj box                # npz target id (cup/bottle→bottle, box→box)
#
# 선행: T1 thor_all.sh up  ·  T2 Isaac ▶  ·  T3 nero_pc_control.sh up
#       (시맨틱 쓰려면 ssh thor '~/grasp/vlm_server.sh up')
# ============================================================================
set -eo pipefail
LABEL=cup; OBJ=""; SEMOFF=""; DRY=""; TASK=""; NOCACHE=""
while [ $# -gt 0 ]; do case "$1" in
  --label)    LABEL="$2"; shift 2 ;;
  --obj)      OBJ="$2"; shift 2 ;;
  --task)     TASK="$2"; shift 2 ;;
  --sem-off)  SEMOFF=1; shift ;;
  --no-cache) NOCACHE=1; shift ;;
  --dry)      DRY=1; shift ;;
  *) echo "무시: $1"; shift ;;
esac; done
# cup/bottle/cyl 류(pot/kettle/mug 포함)는 npz 가 bottle_id 로 → OBJ=bottle.
# box·pan(얕은 원반) 은 box_id → OBJ=box.  (capture 의 BOTTLE_LABELS 와 일치시킬 것)
[ -z "$OBJ" ] && { case "$LABEL" in box|pan|book|laptop|tray) OBJ=box ;; *) OBJ=bottle ;; esac; }

source /opt/ros/humble/setup.bash
source "$HOME/ros2_ws/install/setup.bash"
cd "$(dirname "$0")"
NPZ="$HOME/grasp/pick.npz"
t0=$EPOCHREALTIME
_lap() { awk "BEGIN{printf \"  [%5.1fs] %s\n\", $EPOCHREALTIME-$t0, \"$1\"}"; }

echo "══ 1. capture (label=$LABEL) ══"
rm -f "$NPZ" "${NPZ%.npz}.rgb.png"
if ! python3 capture_flange_npz.py --label "$LABEL" --sam --out "$NPZ" > /tmp/nero_cap.out 2>&1 || [ ! -f "$NPZ" ]; then
    echo "!! capture 실패 — stale 데이터로 진행하지 않고 중단:"; tail -4 /tmp/nero_cap.out | sed 's/^/   /'
    exit 1
fi
grep -E 'seg_px|obj_world|arm_q|못 받음' /tmp/nero_cap.out || true
_lap "capture 완료"

echo "══ 2. → Thor  파이프라인 (obj=$OBJ${TASK:+, task=\"$TASK\"}) ══"
rsync -q "$NPZ" "${NPZ%.npz}.rgb.png" thor:grasp/
ssh thor "${SEMOFF:+NERO_SEM_OFF=1 }${NOCACHE:+NERO_SEM_NOCACHE=1 }~/grasp/run_pipeline.sh ~/grasp/pick.npz $OBJ $LABEL $(printf '%q' "$TASK")" 2>&1 \
  | grep -E 'seed\]|sem\]|regions\]|오탐|unified rank|reachable:|plan [0-9]|no reachable|중단|approach \[|Cartesian fraction|캐시 HIT' || true
_lap "Thor 파이프라인 완료"

if ! ssh thor 'test -f ~/grasp/step6_pick.json' || \
   [ "$(ssh thor 'stat -c %Y ~/grasp/step6_pick.json')" -lt "$(date -d '2 min ago' +%s)" ]; then
  echo "!! step6_pick.json 이 새로 안 생김 — 파이프라인 실패 (reachable grasp 없음 등). 중단."
  exit 2
fi
rsync -q thor:grasp/step6_pick.json "$HOME/grasp/step6_pick.json"
rsync -q thor:grasp/viz_overlay.json "$HOME/grasp/viz_overlay.json" 2>/dev/null || true  # 라이브 뷰 오버레이

if [ -n "$DRY" ]; then _lap "dry — exec 생략"; exit 0; fi

echo "══ 3. exec ══"
python3 exec_pick.py | grep -E '\[p[0-9]|grasp_cfg|grip-close|verdict|NO_REACH' || true
_lap "완료"
