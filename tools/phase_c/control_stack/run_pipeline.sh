#!/usr/bin/env bash
# ============================================================================
# run_pipeline.sh  —  THOR 로컬. CGN grasp 파이프라인 3단계를 한 번에.
#   A. two_cgn.py            (cgn_venv — torch 필요, ROS 불필요)
#   B. two_pick_plan_fast.py (system python3 + ROS — 랭킹 + fast IK)
#   C. step6_base.py         (system python3 + ROS — STOMP + Cartesian)
#
# 선행: ~/grasp/plan_stack.sh up  (move_group 떠 있어야 B,C 가 됨)
# 사용:  ~/grasp/run_pipeline.sh <scene.npz> [object]
#   object 기본값 bottle
# 출력:  ~/grasp/step6_pick.json
# ============================================================================
set -eo pipefail          # NOTE: -u 금지 — ROS setup.bash 가 unbound var 를 씀
NPZ="${1:?사용: $0 <scene.npz> [object]}"
OBJ="${2:-bottle}"
VENV="$HOME/grasp/cgn_venv/bin/python3"
cd "$HOME/grasp"

echo "════ A. CGN (상주 서버 :8010) ════"
if curl -s -o /dev/null -m 2 http://127.0.0.1:8010/health; then
    curl -s -X POST http://127.0.0.1:8010/infer -H 'Content-Type: application/json' \
        -d "{\"npz_path\": \"$NPZ\", \"out_dir\": \"$HOME/grasp\"}" | python3 -c "import sys,json;r=json.load(sys.stdin);print('  %dms' % r['ms']); [print('  ',x) for x in r['results']]"
else
    echo "  (서버 down — two_cgn.py 로 폴백, 모델 재로드 ~10s)"
    "$VENV" two_cgn.py "$NPZ"
fi

echo "════ B+C. rank + STOMP (ROS) ════"
source /opt/ros/jazzy/setup.bash
source "$HOME/ros2_ws/install/setup.bash"
rm -f "${OBJ}_pick.json"
python3 two_pick_plan_fast.py "$OBJ"
if [ ! -f "${OBJ}_pick.json" ]; then
    echo "!!! two_pick_plan_fast 가 ${OBJ}_pick.json 을 못 만듦 (reachable grasp 없음)."
    echo "!!! stale 데이터로 step6 안 돌림. 중단."
    exit 2
fi
cp "${OBJ}_pick.json" side_grasp.json
python3 step6_base.py

echo "════ 완료 → ~/grasp/step6_pick.json ════"
python3 - <<'PY'
import json
d = json.load(open("/home/bpdl/grasp/step6_pick.json"))
print("  approach", [round(x, 2) for x in d["approach"]],
      " contact", [round(x, 3) for x in d["contact"]],
      " Cartesian fraction", d.get("cartesian_fraction"))
print("  transit %d + advance %d + retreat %d wp" %
      (len(d["transit"]), len(d["advance"]), len(d["retreat"])))
PY
