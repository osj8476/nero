---
name: nero-startup-runbook
description: "NERO CGN-pick 스택 처음부터 기동 순서 (Thor·Isaac 재부팅 후, PC ssh 기준)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6ba50bd9-f824-411e-816c-7dcb14f08a0e
  modified: 2026-09-08T12:57:37.644Z
---

Thor·Isaac Sim 재부팅 후 아무것도 안 떠있을 때. 데스크탑에서, 터미널 3개.
상세: repo `tools/phase_c/control_stack/SETUP_FROM_SCRATCH.md`. 관련 규칙 [[feedback-cgn-pick-rules]].

## 기동

```
T1  ssh thor
    ~/grasp/thor_all.sh up          # thor_stack(move_group+pick_ik+STOMP) + yolo_server(:8002 YOLO+SAM) + cgn_server(:8010)
    ~/grasp/thor_all.sh status      # 3개 UP 확인 → exit 가능 (setsid, 서비스 유지)

T2  ~/nero/start_nero_isaac_all.sh  # Isaac + rsp + ros2_control + move_group(Humble) + camera TF + perception + visualize
    → Isaac 창 뜨면 ▶ 재생 필수 (ActionGraph 는 재생 중에만 → 안 누르면 팔 안 움직임)
    → 종료: Ctrl+C 한 번 (Isaac 포함 전부)

T3  ~/ros2_ws/mcp/nero_pc_control.sh up   # planning_node + mcp_robot_server(:9000 http, tmux nero_pc)

    Claude Code:  /mcp                    # nero-robot connected 확인
```

기동 순서 이유: Thor YOLO(:8002) 먼저 → perception_node_sim 이 붙음. Isaac ▶재생 →
ros2_control 이 팔 구동. move_group(T2) → nero_pc_control(T3) 가 붙음.

## pick 한 사이클 (~26s, 전부 ROS, Isaac 스크립트 0)

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
cd ~/ros2_ws/src/nero_sj_pickplace/tools/phase_c
python3 capture_flange_npz.py --label <cup|bottle|box> --sam --out ~/grasp/X.npz
rsync -q ~/grasp/X.npz thor:~/grasp/ && ssh thor '~/grasp/run_pipeline.sh ~/grasp/X.npz bottle'
rsync -q thor:~/grasp/step6_pick.json ~/grasp/ && python3 exec_pick.py   # verdict: grip_after_close>0.020 = HELD
```

## 종료

```
T3  ~/ros2_ws/mcp/nero_pc_control.sh down
T2  Ctrl+C
T1  ssh thor '~/grasp/thor_all.sh down'
```

## 트러블

- move_joints success 인데 팔 안 움직임 → Isaac 창 ■정지→▶재생 (AG 재init)
- 팔 STATUS_ABORTED/TIMED_OUT → `pkill -f rqt_joint_trajectory` (전원 켜지면 goal 선점)
- perception 빈값 → `curl 163.239.19.132:8002/health`
- run_pipeline "CGN 서버 down" → `ssh thor '~/grasp/cgn_server.sh up'`
