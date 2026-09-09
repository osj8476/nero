# 처음부터 세팅 (Thor·Isaac Sim 재부팅 후, 아무것도 안 떠있음)

모든 명령은 **데스크탑(`bpdl-desktop`)** 에서. Thor 는 `ssh thor` 로.
터미널 3개 쓴다: **T1 = Thor(ssh)**, **T2 = Isaac 스택**, **T3 = MCP 스택**.

---

## 0. 사전 (한 번만, 재부팅과 무관)

```bash
# Isaac Sim 창을 GUI 로 띄우려면 데스크탑에 물리 로그인/디스플레이(:1)가 살아있어야 함
echo $DISPLAY                      # :1 나와야 함
ssh thor 'echo ok'                 # 키인증 통과 확인
```

---

## 1. Thor 서비스  —  터미널 T1

```bash
ssh thor
~/grasp/thor_all.sh up
```

`thor_all.sh up` 이 순서대로 띄운다 (전부 setsid 백그라운드, 셸 안 잡음):

| 서비스 | 포트/노드 | 용도 |
|---|---|---|
| `thor_stack.sh`  | `/move_group` + pick_ik + STOMP | grasp IK / 모션 계획 |
| `yolo_server.sh` | `:8002` (YOLO best.pt + COCO + SAM mobile_sam) | 물체 bbox + 인스턴스 마스크 |
| `cgn_server.sh`  | `:8010` (Contact-GraspNet, 모델 1회 로드) | 6-DoF grasp 생성 |

확인:
```bash
~/grasp/thor_all.sh status
#  [thor] UP  + /move_group ...
#  [yolo] UP :8002 -> {"sam":true}
#  {"status":"ok","model":"contact_graspnet","device":"cuda:0"}
```
다 뜨면 T1 은 그냥 둬도 되고 `exit` 해도 서비스는 유지됨.

---

## 2. Isaac Sim + 베이스 ROS  —  터미널 T2 (계속 점유)

```bash
~/nero/start_nero_isaac_all.sh
```
- Isaac Sim 창이 뜬다 → **반드시 ▶ 재생 버튼 누르기.**
  (ActionGraph 는 재생 중에만 동작. 안 누르면 ros2_control 이 팔을 못 움직임.)
- 스크립트가 이어서 rsp / static_tf / ros2_control / spawn_controllers /
  move_group(Humble) / 카메라 정적 TF / perception_node_sim(→ Thor :8002) /
  visualize_3d_bpdl 를 순서대로 띄운다 (~1분).
- 종료: 이 터미널에서 **Ctrl+C 한 번** (Isaac 포함 전부 정리).

토글은 스크립트 상단에 이미 세팅돼 있음:
`RANDOMIZE_BOXES=false`, `ENABLE_SAFETY_MAP=false`, `ENABLE_RQT_JTC=false`(중요 — 켜면 MCP 선점),
`ENABLE_CAMERA_TF=true`, `ENABLE_PERCEPTION=true`, `ENABLE_VISUALIZE=true`.

확인 (다른 터미널):
```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 node list | grep -E 'move_group|perception|robot_state'
ros2 control list_controllers            # arm_controller / gripper_controller / jsb 전부 active
ros2 topic hz /camera/depth/image_raw    # ~12Hz
```

---

## 3. nero-robot MCP 스택  —  터미널 T3

`start_nero_isaac_all.sh`(T2)가 move_group 을 다 띄운 뒤:

```bash
~/ros2_ws/mcp/nero_pc_control.sh up
```
→ `planning_node` + `mcp_robot_server`(HTTP `:9000`) 를 tmux `nero_pc` 로.
선행 노드(`/move_group`, `/robot_state_publisher`) 없으면 중단하니 T2 먼저.

확인:
```bash
~/ros2_ws/mcp/nero_pc_control.sh status
#  MCP :9000/mcp -> HTTP 406  (= listening, 정상)
```

---

## 4. Claude Code 연결

Claude Code 에서:
```
/mcp
```
→ `nero-robot` 이 **connected** 로 나와야 함. (isaac-sim MCP 는 자동)

이제 "옵저베이션 자세로 이동해", "컵 집어" 등 사용 가능.

---

## 전체 기동 순서 요약

```
T1  ssh thor  →  ~/grasp/thor_all.sh up            (move_group·YOLO·CGN 서버)
T2  ~/nero/start_nero_isaac_all.sh  →  ▶ 재생       (Isaac + 베이스 ROS)
T3  ~/ros2_ws/mcp/nero_pc_control.sh up            (planning_node + MCP :9000)
    Claude Code:  /mcp                              (nero-robot connected 확인)
```

---

## CGN pick 한 사이클 (기동 완료 후, ~26s)

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
cd ~/ros2_ws/src/nero_sj_pickplace/tools/phase_c

# 1) 현재 flange 뷰에서 물체 캡처 (SAM 마스크 + RGB — RGB 는 VLM 시맨틱 패스용)
python3 capture_flange_npz.py --label cup --sam --out ~/grasp/cuppick.npz

# 2) Thor 파이프라인 (A' VLM 시맨틱 → A CGN → B rank → C STOMP).  npz + rgb.png 둘 다 rsync.
rsync -q ~/grasp/cuppick.npz ~/grasp/cuppick.rgb.png thor:~/grasp/
ssh thor '~/grasp/run_pipeline.sh ~/grasp/cuppick.npz bottle'
rsync -q thor:~/grasp/step6_pick.json ~/grasp/step6_pick.json

# 3) 실행 (verdict: grip_after_close > 0.020 = HELD). 끝에 <ts>.exec.json 을 Thor runs/ 로 올림.
python3 exec_pick.py
```
`--label` : `cup`/`bottle`/`box` 등. bottle-류는 원통(side), box 는 top.

### VLM 시맨틱 패스 (Task 1~3, 선택)

`~/grasp/vlm_server.sh up` (Thor, Qwen3-VL vLLM :8005, 기동 ~90s, GPU ~4GB+) 를 띄우면
run_pipeline 이 매 사이클 RGB 를 VLM 에 보내 `{obj}_sem.json`(grasp_region / exclude /
keep_upright / shape) 을 만들고, two_pick_plan_fast 가 **aspect_w 분기 대신 CGN raw grasp
단일 랭킹** + 손잡이 제외 필터로 동작. vLLM down 이면 조용히 기하 전용으로 폴백.

```bash
ssh thor '~/grasp/vlm_server.sh up'      # 켤 때만
ssh thor '~/grasp/vlm_server.sh down'    # GPU 회수
```

### 계측 로그 (Task 0)

매 사이클 `~/grasp/runs/<ts>.json`(plan) + `<ts>.exec.json`(exec verdict, PC 가 rsync).

```bash
ssh thor 'python3 ~/grasp/runs/show.py 20'   # 최근 20 사이클 표 (cgn/reach/frac/j1/verdict/타이밍)
```

---

## 종료

```
T3  ~/ros2_ws/mcp/nero_pc_control.sh down
T2  Ctrl+C
T1  ssh thor '~/grasp/thor_all.sh down'
```

---

## 트러블슈팅

| 증상 | 원인/조치 |
|---|---|
| `move_joints` success 인데 팔 안 움직임 | Isaac ▶ 재생 안 눌렀거나 timeline 이상 → **Isaac 창에서 ■정지 → ▶재생** (ActionGraph 재init) |
| 팔 명령이 STATUS_ABORTED / TIMED_OUT | `rqt_joint_trajectory_controller` 가 떠서 선점 → `pkill -f rqt_joint_trajectory` |
| perception `/detected_objects` 빈값 | YOLO 서버(Thor :8002) 확인 `curl 163.239.19.132:8002/health` |
| run_pipeline `CGN 서버 down` | `ssh thor '~/grasp/cgn_server.sh up'` |
| capture `SAM 요청 실패` | `SAM_SERVER_URL` 확인 (기본 163.239.19.132:8002). `BOX_SERVER_URL` env 는 낡은 주소라 안 씀 |
| Thor plan st 못 뜸 | `~/grasp/thor_stack.sh log` — headless_plan_test.launch.py 는 cwd /tmp 에서 정상 |
