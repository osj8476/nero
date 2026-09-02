# VLM 능력 확장 — open-vocab 검출 / segmentation backend / grasp 정책 / 배치 리셉터클

## 요약

YOLO가 못 주는 것(물체 축 방향, open-vocab 물체, 큰 물체 안의 파트, 놓을
곳 의미론)을 VLM/경량모델 티어로 채우기 위한 2026-09-02 라운드. 핵심
설계 원칙: **8B VLM(8005)은 "의미 질의 전용"으로 격리하고, 기하·검출은
Thor 의 경량 로컬 모델(YOLO-World, SAM/depth)로 내려 속도를 확보한다.**

구현 대상 5개 중 value map(연구용 설명맵)은 이번 라운드 제외.

## 현재 상태 / 결론

### (1) segmentation backend — 물체 축 방향
- `segmentation_backend.py` 에 실제 backend 2종 추가:
  - `DepthPlaneSegmentationBackend` — 모델 0개. bbox 안 median depth 기준
    band 밖(배경 테이블면) 픽셀 제외. 추가 비용 ~수 ms.
  - `SamSegmentationBackend` — ultralytics MobileSAM/SAM2. bbox=box prompt,
    `point_hint_px`=point prompt. lazy load, 실패 시 NoOp 사각마스크 폴백.
- `get_default_backend()` 가 `SEG_BACKEND` 환경변수로 분기
  (`noop` 기본 / `depth_plane` / `sam`). 인스턴스 캐시.
- `mcp_robot_server.estimate_object_geometry` 가 `NoOpSegmentationBackend()`
  대신 `get_default_backend().segment(..., depth_image=, point_hint_px=)`
  사용. 반환 JSON 에 `seg_source` 추가.
- **왜**: CLAUDE.md "major_axis_yaw_deg 신뢰 불가"의 유력 원인이
  bbox-as-mask 배경 오염 → 실제 마스크를 주면 PCA 장축이 배경에 안 끌림.
  참고: [Grasp Geometry 파이프라인](grasp_geometry_pipeline.md).
- **검증 수준**: 배선/폴백/synthetic 통과
  (`tools/test_vlm_feature_additions.py`). depth_plane 이 synthetic 에서
  배경을 실제로 제외함(2000px vs NoOp 3000px). **실기·실측 yaw 정확도
  개선은 미검증** — band_m(0.06) 은 시작값이지 튜닝값 아님.

### (2) open-vocab 검출 — YOLO가 못 잡는 물체
- `vlm_grasp_server.py` 에 `/detect_open_vocab` 엔드포인트(YOLO-World,
  `yolov8l-worldv2.pt`, 레포 루트에 있음). lazy load + 로드시 GPU 강제이동
  + 워밍업. `YOLOWORLD_MODEL` 로 가중치 지정. ultralytics 없으면 503.
- `mcp_robot_server._openvocab_bbox()` 헬퍼 → `_vlm_ground_bbox_for_grasp`
  (infer_grasp 경로)와 `ground_object()` MCP 툴 둘 다 **8B VLM
  /ground_object 전에** open-vocab 을 먼저 시도. 실패/미설치면 조용히 VLM
  폴백. 성공 시 응답 `source: "open_vocab"`, `grounding: "precise_bbox"`.
- **latency 실측(demo env, RTX 3080Ti급)**: 콜드 ~6s(로드+워밍업), warm
  같은 클래스 ~18ms, 클래스 스위칭 ~50~130ms. 8B VLM(2~4s) 대비 20~100x.
- **confidence 주의**: YOLO-World 는 threshold 낮추면 오검출 뿌림
  (`vlm_grounding_dino.py` 의 GroundingDINO 메모와 동일 성격). 클라이언트는
  top-1 만 쓰고 `MIN_CONF=0.3` 로 거른다 — 애매하면 VLM 으로 폴백됨.
  plain "box"(무지 골판지)는 conf 0.08 수준으로 잘 안 잡힘 → VLM 폴백.
- **검증 수준**: 엔드포인트 HTTP 왕복 + 실이미지 검출 확인
  (test_scene.jpg 에서 cup 0.93). **실기 카메라·pick 성공 미검증.**

### (3) 큰 물체 안 파트 추출 (crop)
- `estimate_object_geometry(target_label, parent_label=)` — parent_label 을
  주면 YOLO 검출 없이 `ground_object(part, parent)` 로 파트 bbox 확보 후
  그 영역에 geometry. `SEG_BACKEND=sam` 이면 파트 중심을 SAM point prompt
  로 같이 넘김. 계층적 그라운딩 인프라(`_ground_hierarchical`)는 기존 것
  재사용.
- **왜**: "서랍 손잡이 축 방향" 같은 질의를 한 툴에서. VLM 은 파트
  bbox/point 만, 축 계산은 로컬 geometry.
- **검증 수준**: 배선만. 실기 미검증.

### (4a) grasp 정책 — 맥락 → grasp_dir
- `grasp_types.resolve_grasp_dir(vlm_response, object_z=None) -> ResolvedGrasp`
  — CLAUDE.md "그립 형태가 애매할 때" 변환표를 **코드로**. 지금까진 Claude
  가 매 호출 손으로 적용하던 것.
  - PINCH+VERTICAL → side, PINCH+HORIZONTAL → pinch, TOP/SIDE 그대로
  - TOP 인데 object_z 가 허리 대역(`WAIST_Z_MIN_M`~`MAX_M` = 0.30~0.50m)
    이면 → side (`top_downgraded_to_side=True`)
  - `side_approach_deg` 는 `suggested_side_approach_deg` → approach_direction
    매핑 순으로 채움
- **아직 planning_node 에 호출 연결 안 함** — planning_node 는 연구실 PC
  쪽 미push 코드가 있어 이번 세션은 nero 기준. 함수만 추가+단위테스트.
  PC pull 후 `pick_object`/`slide_object` 진입부에서 호출 연결 예정.
- z 대역 경계(0.30~0.50)는 CLAUDE.md 에 "확정값 아님, 경험적 범위"로
  명시됨 — 실측 후 조정.
- **검증 수준**: 순수 함수 단위테스트 8케이스 통과.

### (4b) 놓을 곳 의미론 — 바구니/선반/서랍
- `vlm_grasp_server._SCENE_SYSTEM` + `SceneResponse` + `_validate_scene` 에
  객체별 `container_type` (basket/bin/tray/box/drawer/shelf/bowl/none) +
  `is_open` 추가. 허용집합 밖/누락이면 안전값(none/false).
- `mcp_robot_server._analyze_scene_core` 반환에 `placement_targets` 추가 —
  리셉터클만 필터, `is_open=True` 우선 정렬. 좌표는 계산 안 함(호출부가
  `ground_object` 로, CLAUDE.md "놓을 곳은 pick 전에 좌표 확보" 규칙).
- **검증 수준**: 스키마/검증 함수 + `_validate_scene` synthetic. **실제
  VLM(8005) 이 이 필드를 잘 채우는지 미검증** — 서버가 안 떠 있었음.

## 이력

- 2026-09-02: 위 5개 구현. 착수 순서 = 위험도順
  (segmentation backend → resolve_grasp_dir → open-vocab → 파트 → scene).
  전부 additive(백엔드 abstraction, 우선순위 티어 추가, 기본값 있는 스키마
  필드). 기존 `tools/test_grasp_geometry_pipeline.py` 통과 유지.
- 컴퓨트 분담 결정: **전부 Thor**. PC 는 Isaac Sim + planning_node +
  시각화 프론트만. 근거: 속도 중요 + PC 는 3080Ti 를 Isaac Sim 이 점유 +
  구형 CPU + 16GB RAM. 경량 perception 모델은 Thor 통합메모리(115GB free)
  에 co-locate. 리스크: Thor iGPU CUDA 컨텍스트 추가 → nvmap 누수 가중
  (memory: vllm-thor-setup). 완화: 모델을 vlm_grasp_server 프로세스 안에
  lazy load(별도 프로세스 안 늘림).

## 폐기된 접근 / 하지 말 것

- **GroundingDINO** (`vlm_grounding_dino.py`): confidence 가 실제 박스와
  그리퍼 오검출에서 겹쳐 threshold 로 분리 불가(레포 기존 메모). YOLO-World
  로 감. YOLO-World 도 conf 보정은 조심 — top-1 + 폴백으로 회피.
- **YOLO-World 를 CPU 로**: 클래스당 ~3.7s. GPU 필수. 로드 직후 GPU 강제
  이동 안 하면 클래스 스위칭 시 CLIP txt_feats/predictor device mismatch 로
  죽음(ultralytics 8.4 이슈) — 로더 워밍업 + set_classes 후 txt_feats.to(dev)
  + `model.predictor=None` 조합으로 해결.
- **8B VLM 에게 bbox 좌표를 직접 시키기**: 레포 기존 메모대로 수십 cm
  오차 + few-shot 앵커링. open-vocab 검출기가 진짜 bbox 회귀를 함.
- **CLAUDE.md 변환표를 프롬프트 안에서만 관리**: 규칙(정책)이므로 코드
  (`resolve_grasp_dir`)로. 프롬프트에는 순수 시각 질의만 남기는 방향.

## 관련 문서

- [Grasp Geometry 파이프라인](grasp_geometry_pipeline.md) — segmentation
  배경누출이 major_axis_yaw 를 망가뜨리는 현상(이번 backend 추가의 동기)
- [인지 모델 개발 도구](perception_dev_tools.md) — vlm_boxyolo/YOLO 서빙,
  모델 서버 포트 전환 주의
- [MCP 로봇 서버 아키텍처](mcp_pickplace_architecture.md)
- 관련 코드: `sj_pickplace/segmentation_backend.py`,
  `sj_pickplace/grasp_types.py` (`resolve_grasp_dir`),
  `yolo/vlm_grasp_server.py` (`/detect_open_vocab`, scene container_type),
  `sj_pickplace/mcp_robot_server.py` (`_openvocab_bbox`,
  `estimate_object_geometry` parent_label, `_analyze_scene_core`
  placement_targets), `tools/test_vlm_feature_additions.py`
- CLAUDE.md: "그립 형태가 애매할 때 — infer_grasp 필수",
  "estimate_object_geometry의 major_axis_yaw_deg — 신뢰 불가",
  "놓을 곳은 반드시 pick 전에 미리 스캔/좌표 확보"
