# 저장소 구조 — nero vs ros2_ws/src/nero_sj_pickplace

## 요약
`/home/bpdl/nero`와 `/home/bpdl/ros2_ws/src/nero_sj_pickplace`는 **같은
GitHub 저장소**(`github.com/osj8476/nero.git`)를 각각 다른 시점/브랜치
상태로 체크아웃해둔 두 개의 로컬 워킹카피다. 실수로 "최신 코드가 어디
있지?"를 헷갈리기 쉬운 구조라 문서화한다.

## 현재 상태 (2026-08-11 기준)
- 공통 조상 커밋: `f8dd728` (Delete sj_pickplace/command_parser.py) —
  이 시점까지는 두 저장소 히스토리가 동일했다.
- 이후 **두 방향으로 갈라짐**:
  - `/home/bpdl/nero` → `6f77138`: "Isaac Sim 안전맵/워크스페이스 도구
    추가, sj_pickplace 소스는 nero_sj_pickplace로 이전" — 커밋 메시지
    자체가 "실제 패키지 소스는 이제 저기 없다"고 선언하고 있음.
  - `/home/bpdl/ros2_ws/src/nero_sj_pickplace` → `02b04bd`: "그립 각도/
    폐루프 place 검증 모듈 분리, sim perception 노드 추가" — 이게 실제
    로봇 동작 코드의 최신 라인.
- `nero_sj_pickplace`는 현재 `origin/master`보다 1커밋 앞서 있고, 커밋
  안 된 로컬 변경(`mcp_robot_server.py`, `planning_node.py`)도 있다 —
  즉 이 저장소 자체도 원격과 완전히 동기화된 상태가 아니다.

## 지금 뭐가 어디 있는지
- **로봇 동작 코드(패키지 본체)** — `grasp_kinematics.py`,
  `mcp_robot_server.py`, `planning_node.py`, `placement_verification.py`,
  `perception_node.py`/`perception_node_sim.py`, `camera_calibration.py` →
  전부 `ros2_ws/src/nero_sj_pickplace/sj_pickplace/`에만 있다. `nero/`
  쪽에는 동명 파일이 없다.
- **Isaac Sim 관련 자산** (`safety_map_layer.usd`, `nero_simul.usd`,
  `build_safety_map_layer.py`, `start_nero_isaac_all.sh`) → `nero/`에만
  있음.
- **IK 도달 범위 스캔 도구 + 결과** (`tools/ik_*.py`,
  `tools/ik_side_reachable_map.txt`) → `nero/`에만 있음. → 자세한 내용은
  [[grasp_kinematics_ik]] 참고.
- **데이터셋/라벨링/모델 서빙 도구** (`tools/box_dataset_capture.py`,
  `tools/sam_labeler.py`, `yolo/vlm_boxyolo.py`, `yolo/best.pt` 등) →
  `nero/`에만 있음. → 자세한 내용은 [[perception_dev_tools]] 참고.
  단, `yolo/best.pt`가 실제로 서빙되어 `perception_node.py`가 쓰는
  운영 가중치 파일이므로, **패키지 코드는 `ros2_ws`에 있어도 그 코드가
  의존하는 모델 파일은 `nero/`에 있다** — 두 저장소가 실질적으로 서로
  의존한다.

## 주의할 점
- `nero/`에서 `grasp_kinematics.py`나 `mcp_robot_server.py` 같은 파일을
  찾으면 "없다"가 정상이다. 없다고 `ros2_ws` 쪽이 잘못됐다고 판단하지
  말 것.
- 반대로 IK 스캔 결과나 YOLO 학습/라벨링 도구가 필요하면 `ros2_ws`가
  아니라 `nero/`를 봐야 한다.
- 두 저장소 다 origin이 같은 `osj8476/nero.git`이라, 한쪽에서 push하면
  다른 쪽 fetch 시 갈라진 히스토리(diverged branch)로 보일 수 있다.

## 관련 문서
- [[mcp_pickplace_architecture]]
- [[perception_calibration]]
- [[grasp_kinematics_ik]]
- [[perception_dev_tools]]
