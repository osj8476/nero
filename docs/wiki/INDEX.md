# 프로젝트 위키 인덱스

이 파일은 `docs/wiki/` 아래 주제별 문서의 목차다. 각 줄은
`- [제목](파일명.md) — 한 줄 요약` 형식으로 유지한다 (본문은 여기 쓰지 않는다).

새 세션에서 특정 주제(그립, 스캔, place 검증 등)를 다시 건드리게 되면
먼저 이 인덱스에서 관련 항목이 있는지 확인하고 해당 문서를 읽어라.

## 목차

- [저장소 구조 (nero vs ros2_ws/src/nero_sj_pickplace)](repo_layout.md) — 같은 원격을 공유하는 두 워킹카피, 갈라진 시점과 각각 뭐가 있는지
- [MCP 로봇 서버 & Pick/Place 오케스트레이션](mcp_pickplace_architecture.md) — mcp_robot_server/planning_node/placement_verification 아키텍처, 상수, 재발방지 이력, 안전정지 임시비활성화/속도0.15/from_scan재조회/_abs_angle_to_quat(2026-08-14), joint1 pre-rotate 완전 제거(2026-08-17, 그립각도 근본수정 이후 불필요해짐), stack_boxes 제거(2026-08-28)
- [인지 파이프라인 & 카메라 캘리브레이션](perception_calibration.md) — perception_node(실물/시뮬), hand-eye 캘리브레이션, YOLO 각도보정, 알려진 사각지대(쌓인 박스가 raw 모델에서 병합되는 이슈 포함), 실물 3D dedup 포팅 이력, d435_camera_joint 수정 위치/calib_visual.py 버그/재캘리브 자세 가이드(2026-08-14)
- [그립 각도 선택 & IK 도달범위 스캔 도구](grasp_kinematics_ik.md) — grasp_kinematics.py 구조, TCP offset 상수 위치 정정, ik_side_reachable_map.txt는 미연결 오프라인 산출물, sim x>0 팔 꼬임 근본원인(pitch=180 axis-flip, +180도 보정) + YawCandidateSelector 4후보 확장 + align/descend/place position_yaw 오염 버그(sim_box_aligned_quat 신설, 2026-08-15)
- [인지 모델 개발 도구](perception_dev_tools.md) — 데이터셋 캡처/SAM 라벨링/YOLO 서빙(vlm_boxyolo.py, best.pt) 워크플로우와 운영 배포 컨벤션, box_yolo_v6 배포 완료(2026-08-12, 부분개선/v5 백업 있음), 모델 서버 포트 전환 시 주의사항
- [Grasp Geometry 파이프라인(RANSAC+PCA) 신뢰 불가 판정](grasp_geometry_pipeline.md) — estimate_object_geometry의 major_axis_yaw_deg가 실제 회전과 무관하게 편향 수렴하는 현상 실측 확인(2026-08-31), NoOp segmentation 배경누출 의심, 코드는 유지하되 사용 보류
- [VLM 능력 확장 — open-vocab/segmentation/grasp정책/배치](vlm_capability_tiers.md) — 2026-09-02: YOLO-World /detect_open_vocab 티어, SamSegmentation/DepthPlane backend + SEG_BACKEND, resolve_grasp_dir(변환표 코드화), analyze_scene container_type/placement_targets. 전부 Thor 컴퓨트, 배선/synthetic 검증 완료·실기 미검증
