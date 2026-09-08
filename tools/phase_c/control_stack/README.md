# 제어 스택 — 두 개의 독립 스택

## 머신 (헷갈리지 말 것)

| 이름 | hostname | IP | 무엇이 도는가 |
|---|---|---|---|
| **데스크탑** | `bpdl-desktop` | 163.239.19.67 | Claude Code 세션, 이 repo, **Isaac Sim**, 베이스 ROS(Humble), YOLO/VLM 서버(8002/8003) |
| **Thor** (젯슨) | `bpdl` | 163.239.19.132 | `ssh thor`, MoveIt2(Jazzy) move_group + pick_ik + STOMP, CGN |

- 데스크탑에서 Thor 접근: `ssh thor`
- **Thor 셸에서 직접** plan 스택 제어: `~/grasp/thor_stack.sh` (아래 1-A)
- **데스크탑에서** Thor plan 스택 제어: `tools/phase_c/control_stack/thor_planning.sh` (아래 1-B)

NERO 재설계 이후 로봇 제어는 **서로 독립인 두 스택**으로 나뉜다.
DDS가 데스크탑↔Thor 사이에서 안 통하므로 둘은 서로 안 보이고, 충돌하지 않는다.
필요한 것만 띄우거나, 둘 다 띄워도 된다.

| | Thor plan-only 스택 | PC 자연어 제어 스택 |
|---|---|---|
| 머신 | Thor (젯슨, ROS 2 **Jazzy**) | PC (ROS 2 **Humble**) |
| 목적 | CGN grasp 파이프라인의 **궤적 계산** | 자연어 조인트/픽 명령 |
| 구성 | move_group + pick_ik + STOMP (로봇 없음) | planning_node + move_group + ros2_control + Isaac Sim + nero-robot MCP |
| 실행 주체 | 사람 (아래 스크립트) | 사람 (아래 스크립트) |
| 로봇 동작 | 없음 (파일로 궤적만 출력) | Isaac Sim 로봇이 실제로 움직임 |
| 시작 | Thor: `~/grasp/thor_stack.sh up`  ·  데스크탑: `thor_planning.sh up` | `start_nero_isaac_all.sh` → `~/ros2_ws/mcp/nero_pc_control.sh up` |

---

## 1. Thor plan-only 스택

CGN → grasp → STOMP 궤적을 계산할 때만 필요. `headless_plan_test.launch.py`
(rsp + jsp + `static_transform_publisher 0 0 0 0 0 0 world base_link` + move_group,
**ros2_control 없음**)를 백그라운드로 띄운다 (tmux 없음, `setsid` + PGID 파일).

### 1-A. Thor 셸에서 직접 (이게 기본) — `~/grasp/thor_stack.sh`

tmux 안 씀. `setsid` 로 자체 프로세스 그룹 만들어 백그라운드로 띄우고 PGID 를
파일(`~/grasp/.thor_stack.pgid`)에 남긴다 → `down` 이 그 그룹 전체를 한 방에 정리.

```bash
~/grasp/thor_stack.sh up        # 띄우고 /move_group 뜰 때까지 대기 후 종료 (터미널 안 잡음)
~/grasp/thor_stack.sh status
~/grasp/thor_stack.sh log        # move_group 로그 tail -f (Ctrl-C 로 빠짐, 스택 유지)
~/grasp/thor_stack.sh restart
~/grasp/thor_stack.sh down
```

로그: `~/grasp/logs/move_group.log`

### 1-B. 데스크탑 셸에서 (ssh 경유)

```bash
cd ~/ros2_ws/src/nero_sj_pickplace/tools/phase_c/control_stack
./thor_planning.sh up          # = ssh thor "~/grasp/thor_stack.sh up"
./thor_planning.sh {status|log|down|restart}
```

`up` 이 끝나면 파이프라인 실행 (Thor `~/grasp/`). **한 줄:**

```bash
~/grasp/run_pipeline.sh ~/grasp/newpick.npz          # A,B,C 전부, 인터프리터 자동
```

수동으로 나눠 실행할 때 **인터프리터가 다르다** (이거 때문에 torch 에러 남):

```bash
cd ~/grasp
# A. two_cgn — cgn_venv (torch 필요, ROS 불필요)
~/grasp/cgn_venv/bin/python3 two_cgn.py ~/grasp/newpick.npz
# B,C — system python3 + ROS
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
python3 two_pick_plan_fast.py bottle                 # 랭킹 + fast IK  (~2s)
cp bottle_pick.json side_grasp.json
python3 step6_base.py                                # STOMP + Cartesian 검증  (~12s)
# -> step6_pick.json  (transit/advance/retreat joint 궤적)
```

launch 파일: `agx_arm_sim/Moveit2/nero_gripper_moveit_config/launch/headless_plan_test.launch.py`
(rsp + jsp + `static_transform_publisher 0 0 0 0 0 0 world base_link` + move_group, **ros2_control 없음**).

---

## 2. PC 자연어 제어 스택

Claude 에게 자연어로 조인트/픽 명령을 주고 Isaac Sim 로봇이 움직이게 한다.

**체인**

```
Claude Code ──http──▶ nero-robot MCP (:9000)
            ──/arm_command──▶ planning_node
            ──▶ move_group (Humble) ──▶ ros2_control ──▶ Isaac Sim
```

**시작 순서**

```bash
# (1) 베이스 스택 — Isaac Sim + ROS. 이 창은 켜둔 채로.
./start_nero_isaac_all.sh
#     Isaac Sim 창이 뜨면 ▶ 재생 버튼 누르기

# (2) planning_node + nero-robot MCP(:9000)
~/ros2_ws/mcp/nero_pc_control.sh up

# (3) Claude Code 에서
/mcp          # nero-robot 이 connected 인지 확인
```

**왜 MCP 를 사람이 띄우나:** 이 프로젝트의 `~/.claude.json` 에서 nero-robot 이
`{"type":"http","url":"http://127.0.0.1:9000/mcp"}` 로 등록돼 있다. Claude Code 는
서버를 자식으로 실행하지 않고 **이미 떠 있는 HTTP 서버에 붙기만** 한다.
(예전 stdio 방식은 `run_mcp_robot_server.sh` — Claude Code 가 직접 실행.)

**자연어 명령 예시**

| 사용자 발화 | Claude 가 호출하는 tool |
|---|---|
| "옵저베이션 자세로 이동해" | `go_home` (저장된 `observation` = `[0, -0.9991, -0.0008, 1.3986, 0.0056, 0.0012, 1.5491]`) |
| "joint1 을 0.5 로 돌려" | `move_joints(j1=0.5)` |
| "관절 다 0 으로" | `move_joints(j1=0,...,j7=0)` |
| "지금 관절값 알려줘" | `get_joint_positions` |
| "이 자세 'pre_pick' 으로 저장" | `save_pose("pre_pick")` |
| "빨간 컵 집어" | `infer_grasp` → `pick_object` |
| "(x,y,z) 에 놓아" | `place_object` |

저장 포즈 파일: `~/.local/share/nero_robot/saved_poses.json`
(`NERO_POSES_FILE` 로 오버라이드 가능).

**제어**

```bash
~/ros2_ws/mcp/nero_pc_control.sh status
~/ros2_ws/mcp/nero_pc_control.sh log      # tmux 'nero_pc' (planning / mcp 두 창)
~/ros2_ws/mcp/nero_pc_control.sh down
```

---

## 파일

| 파일 | 위치 | 역할 |
|---|---|---|
| `thor_stack.sh` | Thor `~/grasp/` (repo: `thor_stack.thor.sh`) | **Thor 로컬** plan 스택 up/down/restart/status/log (tmux 없음, setsid+PGID) |
| `thor_planning.sh` | 데스크탑 `tools/phase_c/control_stack/` | `ssh thor "~/grasp/thor_stack.sh …"` 얇은 래퍼 |
| `nero_pc_control.sh` | `~/ros2_ws/mcp/` | planning_node + MCP HTTP 서버 up/down/status/log |
| `run_mcp_robot_server_http.sh` | `~/ros2_ws/mcp/` | mcp_robot_server 를 streamable-http :9000 로 실행 |
| `run_mcp_robot_server.sh` | `~/ros2_ws/mcp/` | (기존) stdio 모드 — Claude Code 가 직접 실행하던 예전 방식 |
| `start_nero_isaac_all.sh` | repo 루트 | (기존) Isaac Sim + 베이스 ROS 스택 |

> `mcp_robot_server.py` / `planning_node.py` / `start_nero_isaac_all.sh` 자체는
> 건드리지 않았다 (사용자 작업분). 위 스크립트들은 그 위에 얹는 얇은 런처.
