# Phase C — grasp → pick_ik → STOMP → 궤적 (프로토타입)

재설계 5-Phase 중 Phase 3 검증. **Thor(ROS 2 Jazzy)에서 구동** — STOMP MoveIt 플러그인이
Jazzy부터 정식(`ros-jazzy-moveit-planners-stomp`), Humble엔 없음. PC↔Thor ROS 2 DDS 는
동작 안 함(Humble/Jazzy + 캠퍼스 스위치) → 라이브 실행 브리지는 Phase 4.

기존 `sj_pickplace/` 는 안 건드림. 검증 완료 후 Phase 4 에서 이관.

## 흐름

```
CGN grasp (camera_optical, 4x4)                     ← run_cgn_scene.py / cgn_frame.py
  → base_link TF   world_T_optical = W_T_cam @ diag(1,-1,-1,1)   (Isaac Sim 은 정확한 카메라 pose,
                                                                   실물은 tf2 _cam_to_base)
  → rank3.py : 기하 prefilter → cost = score − wθ·θ⁴ − wc·d_com → fast IK (상위 K)
  → step4b.py : STOMP(seed→pre-grasp) + /compute_cartesian_path(pre→grasp 직선)
  → JointTrajectory  (Phase 4 에서 FollowJointTrajectory 로 실행)
```

## 스크립트

| 파일 | 역할 |
|---|---|
| `cgn_frame.py` | Isaac Sim 렌더 npz(`depth_m`+`K`+`seg`+`W_T_cam`) → CGN → **base_link** 프레임 grasp 리스트 JSON |
| `ik_probe.py` | pick_ik 로 특정 위치의 도달 가능 자세 envelope 스캔 (top-down 자세는 넓게 도달, tilted 는 실패) |
| `rank_grasps.py` | v1 — 전체 grasp + 180° twin → `/compute_ik` → `score − w·θ⁴` 랭킹 (느림, 참고용) |
| `rank2.py` | v2 — v1 + pre-grasp reachable 체크 + box-axis 정렬 항 (여전히 느림, 400 IK) |
| **`rank3.py`** | **현재 표준.** 기하 prefilter(IK 0, ~5ms) → `score − 0.5·θ⁴ − 1.2·d_com`(CoM 항) →
fast IK 상위 K=24 (`timeout` 120ms). ~2s. **d_com = contact ↔ 관측 centroid 거리** (3c) |
| `step4b.py` | `rank3_best.json` → STOMP pre-grasp + Cartesian descent → `c4_trajectory.json` |
| `c2c3_test.py` | 초기 C2/C3 스모크 (pick_ik + STOMP joint-goal plan) |
| `bottle_all.py` | 원통 물병 예제 — CGN 전체 grasp 를 base_link 로 |

## `moveit_jazzy_config/` — 참고 사본

`agx_arm_sim/Moveit2/nero_gripper_moveit_config/` (upstream `agilexrobotics/agx_arm_sim`,
push 불가) 에 적용한 Jazzy 포팅. **이 레포로는 push 안 됨 — Thor 에 직접 반영.**

- `stomp_planning.yaml`, `pilz_..._planning.yaml` : Humble→Jazzy 포맷
  (`planning_plugin`(str) → `planning_plugins`(list), `request_adapters` folded-str → list).
  Jazzy 기본값(`/opt/ros/jazzy/share/moveit_configs_utils/default_configs/`) 이 정답.
- `kinematics.yaml` : pick_ik. **timeout 0.08 / attempts 4** (rank3 필터용 — 정밀 solve 필요 시 상향).
- `move_group.launch.py` : `pipelines=["ompl","stomp","pilz_..."]`, `default_planning_pipeline="stomp"`.
- `headless_plan_test.launch.py` : rsp + jsp + static tf + move_group, ros2_control 없이 (계획만).

## 실행 (Thor)

```bash
sudo apt install ros-jazzy-pick-ik           # 1회
cd ~/ros2_ws && colcon build --packages-select agx_arm_description nero_gripper_moveit_config
source install/setup.bash
ros2 launch nero_gripper_moveit_config headless_plan_test.launch.py   # tmux 권장
# 다른 창:
python3 rank3.py <grasps_base.json>          # → rank3_best.json
python3 step4b.py                             # → c4_trajectory.json
```

## 검증 상태 (2026-09-07)

- C1 ✅ MoveIt+pick_ik+STOMP Jazzy 빌드
- C2 ✅ pick_ik reachability 필터
- C3 ✅ STOMP 궤적 (error_code=1)
- C4 ⚠️ Isaac Sim 실행 — 팔은 도달, **그리퍼 물리 파지 실패** (caging 실패: 손끝 10cm/물체
  ≤7cm, 센터링 오차로 한 손가락이 물체를 밀어냄). **해법 = 접촉 시 fixed-joint attach, Phase 4.**
- Step 1+2(twin+θ⁴): reachability 0/8 → 118~209/후보. Step 3c(CoM): grasp 을 물체 중심으로.
- 속도: rank2 ~10분 → rank3 ~2s.
