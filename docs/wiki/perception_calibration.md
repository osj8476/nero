# 인지(Perception) 파이프라인 & 카메라 캘리브레이션

대상 파일: `sj_pickplace/perception_node.py`,
`sj_pickplace/perception_node_sim.py`, `sj_pickplace/camera_calibration.py`,
`sj_pickplace/scripts/hand_eye_calibration.py`

## 요약
그리퍼에 붙은 RealSense 카메라(eye-in-hand)로 박스를 YOLO 검출하고,
카메라 프레임 3D 좌표를 tf2 + hand-eye 캘리브레이션으로 `base_link`
좌표로 변환해 `/detected_objects`에 publish하는 파이프라인. 실물/시뮬
두 버전이 동일한 출력 스키마를 공유해서 `planning_node.py`는 어느 쪽을
쓰든 코드 변경이 필요 없다.

## 파이프라인 개요 (perception_node.py, 실물)
1. 백그라운드 스레드가 RealSense 컬러+뎁스 프레임(정렬됨, 640x480@30fps,
   수동 노출 `CAM_EXPOSURE=500`)을 계속 받아 `/camera/color/image_raw`,
   `/camera/camera_info`로 publish.
2. 10Hz 타이머가 컬러 프레임을 JPEG→base64 인코딩해 외부 YOLO 추론
   서버(`BOX_SERVER_URL`, 기본 `http://127.0.0.1:8002/detect`)에 POST.
   **인플라이트 가드**(`_inflight` 플래그)로 응답 오기 전 중복 요청을
   막음 — GPU 경합 시 겹친 요청이 `/detected_objects`를 0.5~0.6Hz로
   떨어뜨렸던 문제 때문에 추가됨.
3. bbox 뎁스는 5x5 그리드 포인트를 역투영해 중심 평균 — 단일 픽셀
   샘플링 대비 사선 각도에서 발생하던 ~25~35mm Y축 계통오차를 잡기
   위한 조치.
4. 카메라 프레임 3D → tf2 lookup으로 `base_link` 변환 (eye-in-hand라
   카메라가 팔과 같이 움직이므로 매 감지마다 실시간 조회 필요).
5. `filter_detections`(크기 필터, `MIN_BBOX_SIZE`만 봄)로 1차 거르고
   depth 조회로 3D화한 뒤 `_dedup_3d`(x,y,z 3D 위치 기준 dedup,
   `DEDUP_XY_THRESH_M=0.03`/`DEDUP_Z_THRESH_M=0.025`)로 진짜 중복만
   병합. 2026-08까지는 여기서 2D bbox 중심좌표만 보는 구
   `DEDUP_THRESH=0.08` dedup을 썼었다 — 실물/시뮬 차이 이력은 아래
   "이력" 참고.
6. 결과를 `/detected_objects`(JSON, `std_msgs/String`)로 publish.
   QoS는 전 노드 공통 `BEST_EFFORT`/`KEEP_LAST(1)`(과거 QoS 불일치
   버그 수정 이력 있음).

## 시뮬 버전 차이 (perception_node_sim.py)
- RealSense 직접 구동 대신 Isaac Sim이 퍼블리시하는
  `/camera/color|depth/image_raw`, `/camera/camera_info`를
  `message_filters.ApproximateTimeSynchronizer`(slop 0.15s)로 동기화
  — 독립 콜백 방식은 시뮬 렉 때 컬러/뎁스 페어링이 거의 항상 실패해서
  교체됨.
- 그리퍼 자체 오검출 제거: `GRIPPER_MIN_DEPTH_M=0.15`(그리퍼는 항상
  프레임 하단 ~0.1m 부근에 잡힘).
- dedup(`_dedup_3d`)은 이 파일에 별도로 정의돼 있지 않다 — 2026-08부터
  `perception_node.py`에서 `_compute_box_angle_base`/
  `_transform_with_fallback`와 같은 패턴으로 import해서 재사용한다
  (로직 두 곳에 중복 유지 안 함). 아래 "이력" 참고.

## 카메라 캘리브레이션
- `camera_calibration.py`: RealSense 컬러 스트림 프로파일에서 딱 한 번
  얻은 `CameraIntrinsics`(fx/fy/cx/cy) — **뎁스/IR 내부파라미터가
  아님을 명시적으로 구분**(섞으면 15~25mm 베이스라인 오차 발생).
  `pixel_to_camera_xyz()`는 표준 핀홀 역투영, 유효하지 않으면 더미값
  대신 `None` 반환. 레거시 고정-외부파라미터 경로(`flange_to_camera.json`,
  `~/.nero_calib/`)는 오프라인 검증용으로만 남아있고, **운영 경로는
  perception_node.py의 실시간 tf2 lookup**. `pixel_to_robot_xyz()`는
  구 homography/고정평면 방식이 eye-in-hand 전환 때 제거된 흔적으로
  hard-fail 스텁만 남아있음.
- `hand_eye_calibration.py`: `/feedback/tcp_pose` 구독 + RealSense
  캡처 + 체스보드(`9x6`, `25mm`) PnP → 포즈별 샘플 수집(최소 6장,
  8~10장+ 권장, 회전 다양성 필요) → `cv2.calibrateHandEye`(TSAI)로
  `tcp_link → camera_color_optical_frame` 외부파라미터 산출. base→target
  스프레드가 10mm 넘으면 경고. 결과를 URDF `<origin>` xyz/rpy + JSON으로
  출력하는 대화형 스크립트.

## YOLO 모델
- 실제 서빙은 `nero/yolo/vlm_boxyolo.py`(포트 8002)가 담당 — 자세한
  서빙/데이터셋 도구는 [[perception_dev_tools]] 참고.
- git 이력상 "v5 모델(mAP50=0.993)" + "박스각도 joint7 yaw보정" 커밋
  존재. 각도 보정 로직 자체는 `perception_node.py`의
  `_compute_box_angle_base`(Hough 직선 히스토그램 투표 + 원형평균,
  변 길이 가중치, 테두리 아티팩트/수직엣지 제외)에 있고, 결과
  `angle_base_deg`가 top-down 접근 자세의 `roll=-angle` 계산에 쓰인다
  (실제 적용은 `planning_node.py` 쪽).

## 알려진 이슈 / 사각지대
- **쌓인 박스 2개가 raw YOLO 모델 출력 단계에서 이미 1개 bbox로
  병합됨 (2026-08-11 확진, 미해결)** — `_dedup_3d` 포팅(아래 이력)으로도
  재현됐다. `perception_node`를 우회해 카메라 프레임을 YOLO 서버
  (`vlm_boxyolo.py`, `/detect`)에 직접 던져본 결과 raw 모델이 쌓인
  박스 쌍을 conf=0.915의 높은 확신으로 1개 bbox로 냄(분리 배치된
  박스는 conf=0.880으로 정상 검출) — 즉 post-processing(dedup) 문제가
  아니라 **모델/학습 데이터 문제**다. 원인 분석과 재학습(v6) 진행
  상황은 [[perception_dev_tools]] 참고. dedup 로직 자체는 정상 동작 확인됨.
- 근접 top-down 각도에서 카메라가 아무것도 못 잡는 인식 사각지대 —
  CLAUDE.md의 `placement_verified: null` 규칙과 같은 원인.
- RGB→BGR 변환 누락 버그(PIL은 RGB, YOLO는 BGR 기대) — 수정 완료.
- FP16(`half=True`) 사용 시 검출 누락 — `half=False` 강제.
- sim-time vs wall-clock tf 도메인 불일치로 "extrapolation into the
  past" 발생 가능 — `_transform_with_fallback`이 "now" 스탬프로 재시도.
- 근접 촬영 시 bbox가 타이트해서 실제 엣지를 테두리 아티팩트로 오인해
  각도 검출이 깨졌던 문제 → 비례 마진 추가로 수정. 뎁스 마스킹이
  회색 픽셀을 0으로 만들어 13~15도 유사-안정 오검출을 만들던 문제도
  별도 수정.

## 취약 지점 (향후 유지보수 시 확인 포인트)
- 하드코딩된 엔드포인트(`127.0.0.1:8002`), 함수별로 다른 뎁스 유효
  범위(0.05~3.0m vs 0.05~2.0m) 혼용, 여러 매직 threshold
  (`MIN_BBOX_SIZE`, `DEDUP_THRESH`, `GRIPPER_MIN_DEPTH_M`,
  `VERTICAL_EXCLUDE_DEG` 등)가 회귀테스트 없이 실측으로만 튜닝됨.
- 캘리브레이션 유효성은 hand-eye 캡처 중 체스보드가 절대 움직이지
  않는다는 가정에 강하게 의존.

## 이력
- 2026-08-11: `perception_node.py`(실물)에 3D dedup(`_dedup_3d`,
  `DEDUP_XY_THRESH_M=0.03`/`DEDUP_Z_THRESH_M=0.025`)을 새로 포팅.
  기존 실물 경로는 `filter_detections`가 크기 필터 + 2D bbox 중심좌표
  dedup(`DEDUP_THRESH=0.08`, depth 미고려)을 같이 했는데, 쌓인 두
  박스는 top-down 근처 각도에서 x,y footprint가 거의 겹쳐서 이
  dedup이 "같은 박스"로 오인해 하나를 통째로 지워버리는 문제가
  실측 확인됨(placement_verification이 새로 놓인 박스를 못 찾아
  verified=false/null로 오판하는 원인 중 하나였음).
  `perception_node_sim.py`는 이 문제를 먼저 겪고 `_dedup_3d`로 이미
  고쳐뒀었는데 실물 쪽엔 포팅이 안 돼 있었던 상태. 조치: `_dedup_3d`를
  `perception_node.py`에 정의하고 `filter_detections`는 크기 필터만
  하도록 축소, `_dispatch_inference`에서 3D 변환 후 `_dedup_3d` 호출.
  `perception_node_sim.py`는 이제 자기 로컬 정의 대신 이걸 import해서
  재사용(기존 `_compute_box_angle_base`/`_transform_with_fallback`
  공유 패턴과 통일). 콜드 컴파일 + 유닛 스모크테스트(쌓인 박스 유지,
  진짜 중복은 병합, 크기필터 정상), `colcon build` 성공으로 검증.
  **이 포팅 이후에도 쌓인 박스 미검출이 재현됨 — 원인은 dedup이
  아니라 raw 모델이었음이 같은 날 추가로 확진됨** (위 "알려진 이슈"
  참고, 상세 원인은 [[perception_dev_tools]]).

- 2026-08-14: hand-eye 재캘리브레이션 시도 3회 + 수동 오프셋 보정.
  **이번 세션에서 밝혀진 핵심 사항 2가지:**

  **① calib_visual.py가 잘못된 joint를 패치하고 있었음 (버그)**
  `calib_visual.py`의 `_apply_to_urdf()`가 수정하던 joint가
  xacro 주석(`<!-- -->`) 안의 `usb_plug` 템플릿 joint여서 실제 tf2 체인에
  전혀 반영되지 않았다. 실제 카메라 포즈를 결정하는 joint는
  `d435_camera_joint` (parent=`camera_stand_link`, child=`d435_camera_link`)이며
  `nero_with_camera.urdf` 411번째 줄에 있다:
  ```
  gripper_base → camera_stand_joint → camera_stand_link → d435_camera_joint → d435_camera_link → ... → camera_color_optical_frame
  ```
  **캘리브레이션 결과를 URDF에 반영하려면 반드시 `d435_camera_joint` origin을 수정해야 한다.**
  적용 후 `robot_state_publisher`를 반드시 재시작해야 tf2가 갱신된다.

  **② hand-eye 캘리브레이션 값이 튀는 근본 원인: orientation 다양성 부족**
  3회 시도 결과(1차 x=21.9mm, 2차/3차 ~149mm) 비교 분석:
  - 실패 공통점: j7 변화 범위 30~45° 수준으로 너무 좁음. j5=j6=0 고정 자체는
    문제없지만, j7만으로는 카메라 orientation 다양성이 거의 안 생긴다.
  - 성공 요건: j1을 -60~+60° 범위에서 5단계 이상 변화시키면 j7 범위가
    좁아도 전체 orientation 다양성 확보 가능. j1 60° 변화 = 카메라가
    3D에서 크게 다른 방향을 봄 = 캘리브레이션에 유효한 제약 추가.
  - 참고: `/home/bpdl/sy/cali/handeye/output/cam2gripper_transform_final.yaml`
    (30샘플, TSAI/PARK/HORAUD 서로 15mm 이내 일치)이 현재까지 가장 신뢰도
    높은 결과. 이 값을 `d435_camera_joint`에 역산 적용하는 공식:
    `T_d435 = T_stand^{-1} * T_cam2gripper * T_after_d435^{-1}`

  **수동 보정 방법 (재캘리브레이션 없이 오프셋 미세조정)**
  - 편향 방향이 j1 회전과 함께 돌면 → 캘리브레이션 상수 오차(systematic).
    j1=0° 편향 방향과 j1=90° 편향 방향이 90° 회전 관계이면 확진.
  - `d435_camera_joint` xyz의 y 값을 조정해서 보정 (camera_stand_joint가
    rpy=(0,0,3.14)=180° z회전이라 y가 base 기준으로 반전됨에 주의).
    y를 +δ 하면 base 기준 pick 좌표가 -y 방향으로 δ만큼 이동.
  - 이번 세션 최종 값: `xyz="-0.031 0.012 0.0018"` (원래 -0.028에서 +0.04 보정,
    실측 검증). **재캘리브레이션 전까지 임시값 — 미검증.**

  **탑다운 픽앤플레이스 최적 캘리브레이션 자세 구성**
  - j1: -60°, -30°, 0°, +30°, +60° (5단계)
  - j7: 각 j1마다 0.7, 1.0, 1.4 rad (3단계) → 총 15샘플
  - j5=j6=0 고정 유지
  - 체스보드는 박스가 놓이는 테이블 높이 기준으로 고정 배치
  - 4가지 알고리즘 결과가 10mm 이내 일치하면 신뢰 가능

## 관련 문서
- [[mcp_pickplace_architecture]] — `/detected_objects` 소비 측
- [[perception_dev_tools]] — YOLO 모델 학습/데이터셋/서빙 도구
- [[repo_layout]]
