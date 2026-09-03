# Grasp Geometry 파이프라인 (RANSAC+PCA) — 신뢰 불가 판정

## 요약
`020a2bc`(2026-08-28)에서 도입된 `geometry_3d.py`(RANSAC 평면적합 + PCA
주축) 기반 물체 형상 추정 파이프라인. `estimate_object_geometry` MCP
툴로 수동 조회 가능하며, `pick_object`의 실제 grasp 판단에는 애초부터
자동 연결되지 않은 프로토타입이었다. 2026-08-31 실측 검증 결과,
**핵심 출력값(`major_axis_yaw_deg`)이 물체의 실제 회전과 무관하게
편향된 값에 수렴하는 현상이 확인되어 신뢰 불가로 판정, 사용 보류**.

## 현재 상태 / 결론
- **`major_axis_yaw_deg`(PCA 장축 방위각)는 현재 구현 상태로 신뢰하지
  말 것.** box 물체를 여러 각도로 재배치하며 5회 반복 측정한 결과 실제
  회전각과 무관하게 매번 82~89° 근처로 수렴(분산 거의 없음) — 무작위
  노이즈가 아니라 구조적 편향으로 판단됨.
- 반면 기존 `angle_base_deg`(Hough 2D, `perception_node.py`)는 같은
  측정에서 실제 각도(30°→31.7°, 오차 1.7°)를 정확히 맞춘 경우도 있어
  상대적으로 더 신뢰할 만함 — 단, 이것도 매번 정확했던 건 아니라서
  "완전히 믿을 수 있다"는 뜻은 아니다. Hough 쪽도 근본적으로 2D 원근왜곡
  취약성이 있음([[grasp_kinematics_ik]] 참고).
- `normal_yaw_deg`(RANSAC 평면 normal)는 박스 윗면을 위에서 보는
  구도에서는 정상적으로 verticality 게이트에 걸려 `null`을 반환함 —
  이 부분은 설계대로 동작하는 것으로 보임(문제는 `major_axis_yaw_deg`
  쪽).
- **코드 자체(`geometry_3d.py`의 PCA 수학, `mcp_robot_server.py`의
  `_geometry_to_base_link`/`_rotate_vec_to_base` TF 회전 변환)는 검토
  결과 로직 버그는 발견되지 않았다.** 문제는 알고리즘이 아니라 입력
  데이터 쪽으로 추정됨(아래 "원인 추정" 참고).
- **결정(2026-08-31, 사용자 판단)**: 이 파이프라인을 `pick_object`
  기본 경로에 연결하는 걸 보류하고, 접근각도는 당분간 사용자가 직접
  지정하는 방식으로 진행한다. **코드/파일은 삭제하지 않고 남겨둔다**
  (SAM 등 실제 segmentation이 붙으면 재검토 가능한 상태로 보존).

## 원인 추정 (미확정, 정황 증거)
아래 3가지 정황이 겹쳐서 "NoOp segmentation의 배경 누출"을 유력한
원인으로 본다 — 단, 직접 bbox 크기나 point cloud를 시각화해서 확정한
건 아니라 **가설 단계**:
1. `extents[0]`(장축 길이)가 매 측정마다 0.27~0.44m — 실제 작은 탁상용
   박스(15cm 안팎 추정) 대비 과도하게 큼.
2. `point_count`가 매번 `max_points` 상한(2000)에 꽉 참 — bbox 마스크
   영역의 원본 유효 depth 점 개수가 훨씬 많았다는 뜻, 즉 마스크가
   박스 하나보다 훨씬 넓은 영역(테이블 등 배경 포함)을 덮었을 가능성.
3. `plane_inlier_ratio`가 계속 중간값(0.46~0.6)대 — 단일 평면(박스
   윗면)만이 아니라 다른 표면이 섞여 있음을 시사.

`segmentation_backend.NoOpSegmentationBackend`는 YOLO bbox를 그대로
사각형 마스크로 쓴다(진짜 segmentation 없음, 모듈 자체 docstring에도
명시된 한계). 카메라가 옵저베이션 자세로 고정된 채 반복 측정했기 때문에,
누출된 배경(테이블) 형상은 박스가 실제로 얼마나 회전했든 카메라 기준
거의 일정하게 유지된다 — 이게 `major_axis_yaw_deg`가 회전과 무관하게
82~89°에 몰리는 현상과 정합적이다.

## 이력
- 2026-08-28: `020a2bc`로 `geometry_3d.py`/`estimate_object_geometry`
  등 파이프라인 전체 도입(GPU/ROS 없는 원격 세션, synthetic 데이터
  단위테스트만 완료 상태로 push, 실기 미검증 명시).
- 2026-08-31: Isaac Sim + Jetson Thor(YOLO/VLM 서버 이관 후) 환경에서
  최초 실기 검증. bottle(원통형)과 box(사각형) 양쪽에서 `estimate_object_geometry`
  반복 호출, 사용자가 Isaac Sim에서 직접 확인한 실제 회전각(45°, 30°
  등)과 대조. box를 5회 재측정한 결과 `major_axis_yaw_deg`가 82~89°에
  거의 고정 수렴하는 편향 확인 → 사용자 판단으로 이 신호 사용 보류
  결정. 코드는 유지, 문서만 "신뢰 불가"로 표시.
- 2026-09-03: point cloud sanity spike(`tools/PC_SPIKE_RESULT.md`) 로
  실물 RealSense D435i depth 를 grasp net 입력 관점에서 2 라운드 검증.
  848×480 + High Accuracy preset + spatial/temporal 필터로 hole_ratio 는
  중앙ROI median ~20% → ~9% 로 개선됐으나, ROI depth bimodality 단봉
  프레임 다수 = **물체/작업면이 depth 만으로 분리 안 됨** → SAM mask crop
  전제. 여기서 관측된 배경 누출 + 얕은 관측각이 위 "extents 0.27~0.44m /
  마스크가 배경 포함" 증상과 같은 원인임을 실물 depth 쪽에서 재확인.
  결정: `SAM crop → Contact-GraspNet` 1순위, SAM segmentation 실기 검증이
  grasp net 구현보다 선행. 검은 무광 작업면·투명/고반사 물체는 depth
  경로 밖.

## 폐기된 접근 / 하지 말 것
- **`major_axis_yaw_deg`를 side/pinch 접근각 결정에 그대로 사용하지
  말 것** — 위 편향 문제가 해결(NoOp segmentation을 실제 segmentation으로
  교체 등)되기 전까지는 부정확한 값을 자신 있게 내놓는 게 더 위험함
  (Hough 폴백이나 사용자 지정값보다 나쁠 수 있음).
- `estimate_object_geometry`를 "정답을 알려주는 도구"로 소비하지 말 것
  — 현재는 참고/실험용으로만 취급.

## 관련 문서
- [[grasp_kinematics_ik]] — 기존 `angle_base_deg`(Hough) 계열의 알려진
  한계(2D 원근왜곡)
- [[perception_dev_tools]] — segmentation/YOLO 서빙 관련
- 관련 코드: `sj_pickplace/geometry_3d.py`, `sj_pickplace/segmentation_backend.py`,
  `sj_pickplace/point_cloud.py`, `sj_pickplace/mcp_robot_server.py`의
  `estimate_object_geometry`/`_geometry_to_base_link`
