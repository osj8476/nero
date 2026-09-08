---
name: feedback-cgn-pick-rules
description: CGN pick 실행 시 지켜야 할 규칙 + 파이프라인 현황 (사용자 지시 2026-09-08)
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 6ba50bd9-f824-411e-816c-7dcb14f08a0e
  modified: 2026-09-08T12:18:39.345Z
---

"pick using CGN" 작업에서 반드시 따를 것 (사용자 지시, 2026-09-08):

1. **임의로 Isaac Sim 스크립트를 작성/실행하지 않는다.** `mcp__isaac-sim__execute_script`
   로 물체 스폰·재배치·팔 이동·씬 조작·**스크린샷/렌더**를 즉석에서 하지 마라.
   → 검증도 Isaac 씬 캡처 없이. flange 카메라 ROS 이미지도 찍지 마라 (사용자 2026-09-08 재확인).
2. **명령이 오면 로봇이 "지금 보고 있는 그 화면"에 대해서만 pick한다.** observation
   자세로 옮기거나 카메라를 재조준하지 마라 — 현재 flange 카메라 뷰 그대로.
3. **로봇을 움직이는 것은 Isaac Sim MCP가 아니라 `nero-robot` MCP `move_joints` /
   ros2_control(`/execute_trajectory`) 경로로.** execute_script `apply_action` 금지.
4. **파지 성공 판정 = 그리퍼가 완전히 안 닫힘.** `grip_after_close > 0.020` (prismatic)
   이면 손끝 사이에 물체 있음 = HELD. 완전히 닫히면(≈0.014) 허공 = MISSED.
   씬 캡처로 육안 확인하지 마라. (`exec_pick.py` 의 `verdict` 필드.)

**Why:** 즉석 Isaac 스크립트/스크린샷 루프가 세션당 수십 분 딜레이. 재설계 3층
아키텍처(MCP는 target만)와 일관. 실행 경로는 nero-robot MCP → planning_node →
move_group → ros2_control 로 단일화.

---

## 파이프라인 현황 (2026-09-08, 컵 pick 성공 + 최적화 완료)

**흐름 (전부 ROS, Isaac 스크립트 0회, ~26s):**
```
1. capture_flange_npz.py --label <l> --sam   (PC, ~3s)
     RGB -> Thor /detect?want_mask -> SAM 인스턴스 마스크로 seg
     depth + K + TF(base_link<-camera_color_optical_frame) -> two_cgn 규약 npz
2. run_pipeline.sh <npz> bottle              (Thor)
     A. CGN 상주 서버 :8010 (curl)           ~1.7s  ← 모델 1회 로드
     B. two_pick_plan_fast.py                ~3s    랭킹 + fast IK + orig/flip twin
     C. step6_base.py                        ~4s    STOMP-to-pose + Cartesian 검증
3. rsync step6_pick.json -> PC
4. exec_pick.py                              (PC, ~15s)
     /execute_trajectory 로 transit/advance/close/retreat/lift
     verdict: grip_after_close > 0.020 -> HELD
```

**Thor 서비스 (setsid+PGID, ~/grasp/):**
- `thor_stack.sh {up|down|status|log}` — move_group + pick_ik + STOMP (headless_plan_test)
- `yolo_server.sh` — vlm_boxyolo :8002 (YOLO + SAM, `--sam ~/grasp/mobile_sam.pt`, health `"sam":true`)
- `cgn_server.sh` — cgn_server.py :8010 (CGN 상주, model 1회 로드)
- `run_pipeline.sh <npz> [obj]` — A(curl→8010) + B + C 한 방

**PC:**
- `start_nero_isaac_all.sh` — Isaac + rsp + ros2_control + move_group(Humble). 토글:
  RANDOMIZE_BOXES=false, ENABLE_SAFETY_MAP=false, **ENABLE_RQT_JTC=false**(전원 켜지면
  `/arm_controller/joint_trajectory` 10Hz 발행 → MCP goal 선점, 로봇 안 움직임),
  ENABLE_CAMERA_TF=true, ENABLE_PERCEPTION=true(THOR_YOLO_HOST=163.239.19.132), ENABLE_VISUALIZE=true.
- `~/ros2_ws/mcp/nero_pc_control.sh up` — planning_node + nero-robot MCP(:9000 http)
- `tools/phase_c/capture_flange_npz.py` — `SAM_SERVER_URL` env (BOX_SERVER_URL 은 낡은 윈도우 랩탑 주소라 안 씀)

**머신:** 데스크탑 `bpdl-desktop` 163.239.19.67 (Claude, repo, Isaac, YOLO/VLM 서버는 Thor).
Thor(젯슨) `ssh thor` = hostname `bpdl` 163.239.19.132. DDS 데스크탑↔Thor 안 통함 → 두 move_group 충돌 없음.

## 검증됨 / 개선

- **Isaac ActionGraph articulation_controller**: 런타임 `set_disabled(False)` 만으론 재연결 안 됨.
  **timeline stop→play 필수** (AG 재init). 그 후 ros2_control→팔 정상.
- **원통 side-pick HELD** (rose +13cm). **컵/머그 pick HELD** (grip 0.038 = 반경).
- **180° flip twin**: cand tuple 에 `orig`/`flip` 태그. two_pick_plan_fast IK 체가 둘 다 검증,
  cost 순 채택. step6/pick json 에 `twin` 필드. (실측: "4 orig + 4 flip reachable" 확인)
- **CGN 상주 서버**: 추론 1.7s (재로드 10s 제거).
- **exec_pick 타이밍**: settle 5→1.5s, ramp 0.3→0.13s/wp. 70s → 15s.
- **two_cgn robust fit**: aspect = 5~95 percentile spread (손잡이 outlier 제외). circle_fit
  반복 outlier 제거. cyl axis = 카메라→물체 방향 depth-bias 보정 (radius × 0.35).
- **run_pipeline**: reachable grasp 없으면 stale side_grasp.json 로 step6 안 돌리고 exit 2.
- **compact 물체**: 실제 CGN grasp 랭킹 (bbox top-down 합성은 폴백).

## 남은 것

- **머그 손잡이**: SAM 마스크가 손잡이 포함 → circle_fit/aspect 흔들림. robust 처리로 됐지만
  depth-bias factor(0.35) 는 임시 튜닝값. 손잡이 세그먼트 분리가 정석.
- **box top-pick**: 원거리 저높이 IK 어려움. 미완.
- **step6 궤적 품질**: j1 크게 배회하는 경우 있음 (fraction 1.0 이지만 특이점 근처).
- Phase 4: SAM→CGN→rank→STOMP 를 상주 grasp 서비스로, planning_node 얇게.
