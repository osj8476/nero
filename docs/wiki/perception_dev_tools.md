# 인지 모델 개발 도구 (데이터셋 캡처 / 라벨링 / 서빙)

대상 파일: `nero/tools/box_dataset_capture.py`, `nero/tools/sam_labeler.py`,
`nero/tools/visualize_detection.py`, `nero/tools/grab_frame.py`,
`nero/yolo/vlm_boxyolo.py`, `nero/yolo/vlm_yoloworld.py`,
`nero/yolo/realsense_demo.py`, `nero/yolo/env_check.py`, `nero/yolo/best.pt`,
`augment_dataset.py`(홈 디렉토리, 저장소 밖), `nero/visualize_3d_bpdl.py`
(홈 디렉토리 nero/ 바로 아래, `nero/tools/`나 `nero/scripts/`가 아님 —
`/detected_objects`+카메라 이미지를 구독해 실시간 bbox+cam/base 좌표
오버레이를 그리는 GUI 시각화 도구, 2026-08-12에 이 세션에서 실제로
찾아서 재시작해봄)

([[repo_layout]] 참고 — 전부 `nero/` 저장소에만 존재, `ros2_ws`에는 없음)

## 요약
운영 중인 `perception_node.py`가 쏘는 HTTP 검출 요청을 받아주는 YOLO
서버(`vlm_boxyolo.py`)와, 그 모델을 학습시키기 위한 데이터셋
캡처/라벨링 도구 모음. 실험적 대안(open-vocab YOLO-World, VLM 클러스터
데모)도 같이 들어있다.

## 워크플로우 (추정)
1. **캡처** — `box_dataset_capture.py`: RealSense 컬러 프레임
   1280x720@30fps 캡처. `s`=1장 저장, `a`=자동캡처 토글(기본 1초
   간격), 파일명 자동 인덱싱(재실행해도 이어서 번호 매김). 각도/거리/
   배경/조명/부분크롭을 다양하게 200~300장 이상 권장한다고 docstring에
   명시.
   - `grab_frame.py`(2026-08 추가, `nero/tools/`): `box_dataset_capture.py`의
     축소판 — `/camera/color/image_raw` 구독해서 프레임 1장만 저장.
     eye-in-hand 로봇팔을 `move_joints`로 조금씩(joint1 ±0.02~0.03rad,
     joint2 ±0.05~0.1rad 수준 — 그 이상은 시야가 좁아 박스가 프레임
     밖으로 바로 나가버림, joint1 0.08rad 차이만으로 완전히 빈 화면이
     되는 것도 실측 확인됨) 스윕하면서 특정 장면(쌓인 박스 등)을
     스크립트로 여러 각도 캡처할 때 씀. v6 데이터셋 캡처에 이 방식
     사용 (아래 "모델 버전 계보" 참고).
2. **라벨링** — `sam_labeler.py`: Meta SAM(ViT-B,
   `~/sj/sam_vit_b_01ec64.pth`) 기반 반자동 라벨러. Ctrl+클릭하면
   포인트 프롬프트로 세그멘테이션 후 bbox 추출, 일반 클릭-드래그는
   수동 bbox. YOLO 포맷(정규화 cx,cy,w,h) 텍스트 라벨로 저장/로드.
   `--negative-only`로 배경 전용(빈 라벨) 이미지 대량 생성 가능. SAM
   미설치 시 수동 모드로 자동 폴백.
3. **학습** — (이 저장소 밖에서) ultralytics로 YOLO 학습, 산출물은
   보통 `runs/detect/box_yolo_v2/weights/best.pt` 경로에 생김
   (`vlm_boxyolo.py` 주석 기준).
4. **QA** — `visualize_detection.py`: 라이브 RealSense 프레임을 실행
   중인 검출 서버(`/detect`, 기본 포트 8002)에 던져서 bbox/신뢰도/
   추론시간을 실시간 오버레이 — 학습된 모델을 정적 이미지가 아니라
   라이브 피드로 눈으로 검증.
5. **배포** — 학습된 `best.pt`를 `nero/yolo/best.pt`로 복사/배치.

## 운영 배포 컨벤션
- `nero/yolo/best.pt` (약 22.5MB, YOLOv8 계열) = **실제 서빙 중인
  가중치 파일**. `start_nero_isaac_all.sh`에서
  `python3 ~/nero/yolo/vlm_boxyolo.py --port 8002 --model ~/nero/yolo/best.pt --conf 0.75`
  로 기동됨 — 이게 [[perception_calibration]] 문서의
  `BOX_SERVER_URL=127.0.0.1:8002`가 실제로 가리키는 프로세스.
- `vlm_boxyolo.py`는 클래스 목록에 `"box"`가 없으면 경고를 낸다 —
  `--model`을 실수로 다른 체크포인트로 바꿨을 때 감지하는 안전장치.

## 모델 버전 계보 (box_yolo_v5 vs v6)
- **`box_yolo_v5` — 이전 배포판 (2026-08-12부로 v6로 교체됨).**
  `/home/bpdl/dataset/v3/box_dataset`(train 3186 / val 562 = 3748장)로
  yolov8s.pt 기반 150 epoch, batch16, imgsz640 학습,
  mAP50=0.9927(150 epoch 시점). 가중치는
  `nero/yolo/best_v5_backup_20260811.pt`에 백업돼 있음 — 롤백 필요시
  이 파일을 `nero/yolo/best.pt`로 복사.
- **`box_yolo_v6` — 배포됨 (2026-08-12).** 목적은 "학습 데이터
  근본원인" 문제(쌓인/맞닿은 박스가 학습 데이터에 전혀 없어 raw
  모델이 이런 경우 bbox를 병합)를 고치는 것. 150 epoch 완료
  (2.155시간 소요), 최종 검증 mAP50=0.994/mAP50-95=0.939/
  Precision=0.991/Recall=0.986 — v5(mAP50=0.9927) 대비 회귀 없음.
  2026-08-12에 `nero/yolo/best.pt`로 배포 완료 — **지금부터
  `ros2 topic`/MCP 서버가 실제로 쓰는 가중치는 v6다.** 배포 전 v5는
  `nero/yolo/best_v5_backup_20260811.pt`로 백업해둠(롤백 시 이 파일을
  다시 `best.pt`로 복사하면 됨).
  **다만 이건 완전한 해결이 아니라 부분적 개선이다** — 추가한 학습
  데이터가 배치 2가지(2단/3단 쌓기)를 좁은 각도 범위(로봇 팔을 크게
  움직이면 박스가 프레임 밖으로 나가버려서 소폭 스윕만 가능했음)에서
  41장만 캡처한 것이라, 그 각도/배치와 비슷한 경우는 잘 잡지만 학습
  때 못 본 각도·배치 조합에서는 여전히 병합될 수 있음(배포 전
  사용자가 시각화로 여러 각도 돌려보며 직접 확인 — "되는 부분도 있고
  안 되는 부분도 있다"). 더 견고하게 만들려면 더 다양한 각도/배치로
  데이터를 보강해 재학습(v7) 필요 — 아직 안 함.

### 학습 데이터 근본원인 (쌓인 박스가 raw 모델에서 1개로 병합되는 이유)
[[perception_calibration]]의 "쌓인 박스 2개가 raw YOLO 출력 단계에서
이미 1개로 병합됨" 이슈를 조사한 결과(2026-08-11), v5 학습 데이터셋
(`/home/bpdl/dataset/v3/box_dataset`, 7396장 원본 중 실제 사용된
이미지 파일 3748장)에 **쌓이거나 맞닿은 박스 사례가 단 한 장도 없음**을
확인:
- 증강 스크립트 `/home/bpdl/augment_dataset.py`의 `paste_on_background()`가
  크롭을 항상 정확히 1개만 배경에 합성하는 구조(`new_boxes`가 항상
  길이 1) — 합성 증강 경로로는 애초에 겹치는/쌓인 다중박스 이미지가
  생성될 수 없다.
- 데이터셋 내 멀티박스 라벨(902장 2개/500장 3개/8장 4개)은 전부
  "기존 이미지 증강"(원본 캡처 사진에 우연히 여러 박스가 찍힌 경우)
  분기에서 온 것이고, 좌표를 확인해보면 전부 서로 안 겹치는 분리
  배치였음.
- 결론: dedup 같은 후처리 문제가 아니라 **모델이 쌓인 박스라는
  케이스 자체를 학습한 적이 없어서** 생기는 문제. → v6 재학습으로
  이 케이스를 데이터셋에 추가하는 방향으로 대응 (아래).

### Isaac Sim ground-truth bbox 자동 라벨링 조사 (결론: 안 함)
쌓인 박스 라벨링을 수작업 대신 Isaac Sim에서 자동 생성할 수 있는지
조사(2026-08-11) — **결론: 이번 작업 범위에서 제외, 사람이 직접
라벨링(`sam_labeler.py`)하는 쪽을 택함.** 근거:
- 이 프로젝트에 Isaac Sim Replicator/synthetic-data 파이프라인이
  아예 없음. USD에 `bbox_2d_t`/`SemanticLabel` 토큰은 있으나 미연결
  기본값이고, 활성 publisher가 없음(`ros2 topic list`/
  `ros2 topic echo /tf`에도 박스별 ground truth 없음).
- `/home/bpdl/nero/randomize_box_positions.py`가 pxr로 박스 pose를
  읽고 GT CSV까지 쓰는 패턴이 있어 얼핏 재사용 가능해 보이지만,
  **USD 파일을 Isaac Sim이 로드하기 전에만 동작**하는 오프라인
  스크립트라 실행 중인 라이브 스테이지에는 쓸 수 없음(그런 코드가
  없음).
- 이걸 새로 만들려면 `omni.replicator` 스크립팅이 필요한데, 그 정도
  투자 대비 41장 수동 라벨링(아래) 쪽이 훨씬 저렴하다고 판단.

### v6 재학습 진행상황
1. `grab_frame.py` + `move_joints` 소폭 스윕으로 두 배치 캡처: (1)
   "쌓인 박스 쌍 + 분리된 박스 1개" 21장, (2) "3단 쌓기" 20장 = 총
   41장. 저장 위치 `/home/bpdl/dataset/v6_stack_raw/images/`.
2. `sam_labeler.py`로 41장 전부 수동 라벨링(쌓인/맞닿은 박스도 각각
   개별 bbox로) — 이 환경엔 SAM 체크포인트
   (`~/sj/sam_vit_b_01ec64.pth`)가 없어 실제로는 수동 클릭-드래그
   모드로 동작.
3. **주의(재발 방지, `augment_dataset.py` 사용 시 필독)**: 이 스크립트는
   라벨 파일을 `--src-images`로 지정한 폴더 **안에서** 이미지와 같은
   stem으로 찾는다 — `--labels`처럼 라벨 경로를 별도 지정하는 옵션이
   없음. 처음에 라벨을 별도 폴더에 둔 채 실행했다가 전부 빈 라벨
   (0바이트)로 증강되는 걸 뒤늦게 발견했음 — 라벨을 이미지 폴더로
   복사한 뒤 재실행해서 해결. **다음에 이 스크립트 쓸 때 반드시 라벨을
   이미지와 같은 폴더에 둘 것.**
4. `augment_dataset.py --src-images dataset/v6_stack_raw/images --out
   dataset/v6/augmented_stack --per-image 10` → 451장(원본41+증강410),
   전부 3-box 라벨 정상 확인.
5. v3 데이터셋(3748장) + 신규 451장(85/15 train/val 분리, train
   383/val 68)을 합쳐 `/home/bpdl/dataset/v6/box_dataset/` 구성
   (train 3569 / val 630 = 4199장, `nc:1 names:['box']`). 병합에 쓴
   스크립트는 임시본이라 저장소에 없음(필요시 재작성 가능한 정도로만
   기록).
6. 학습(2026-08-11 15:23 시작): `yolo detect train model=yolov8s.pt
   data=/home/bpdl/dataset/v6/box_dataset/data.yaml epochs=150
   imgsz=640 batch=16 device=0 project=/home/bpdl/runs/detect
   name=box_yolo_v6` — v5와 동일 하이퍼파라미터로 비교 가능하게 함.
   150 epoch 완료(2.155시간), 최종 mAP50=0.994/mAP50-95=0.939 — v5
   대비 회귀 없음.
7. 평가: 임시 포트(8003)로 v6 띄워서 v5(8002, 운영중)와 raw 검출
   동시 비교. 처음 확인 땐 v5=1개 병합/v6=2개 분리로 명확히 차이
   났으나, 이후 재확인 시도에선 **물리 시뮬레이션이 안정화되며 쌓여
   있던 박스가 실제로 넘어져서 옆으로 분리돼버림**(Isaac Sim 박스가
   완벽히 안정적으로 쌓여있지 않을 수 있다는 걸 실측 확인 — 재현
   테스트 계획 시 참고) — 그 뒤로는 v5도 잘 맞혀서 비교가 무의미해짐.
   사용자가 직접 화면으로 여러 각도 돌려보며 최종 판단: 부분적
   개선(위 "모델 버전 계보" 참고).
8. **배포 (2026-08-12)**: `nero/yolo/best.pt`를
   `best_v5_backup_20260811.pt`로 백업 후, v6 가중치로 교체.
   `vlm_boxyolo.py --port 8002 --model nero/yolo/best.pt`로 재시작,
   `perception_node_sim`/`/detected_objects` 정상 동작 확인.

### 운영 중 모델 서버 포트 전환 시 주의사항 (2026-08-12 실측)
v5/v6를 동시에 띄워 비교하려고 `perception_node_sim`을 커스텀 포트로
돌리려다 두 번이나 사고가 났음 — 다음 세션에서 같은 실험 할 때 필독:

- **`BOX_SERVER_URL`과 `BOX_HEALTH_URL`은 서로 다른 환경변수다**
  (둘 다 기본값이 포트 8002). `BOX_SERVER_URL`만 새 포트로 바꾸고
  `BOX_HEALTH_URL`을 안 바꾸면, `_wait_for_box_server`가 계속 죽은
  옛 포트의 헬스체크만 보고 "박스 서버 대기 중..."을 반복하다 60초
  후 `RuntimeError`로 죽는다. 둘 다 같이 바꿔야 한다.
- **`perception_node_sim`/`vlm_boxyolo`가 특정 터미널(pts/1 등)에서
  죽으면 기본 설정(v5, 포트 8002)으로 자동 재기동되는 것으로 보이는
  동작이 관찰됨** (정확한 메커니즘/스크립트는 미확인 — tmux나 별도
  감시 스크립트로 추정). 이 상태에서 커스텀 포트로 별도
  `perception_node_sim`을 띄우면 **두 인스턴스가 동시에 같은
  `/camera/*` 토픽을 구독하고 `/detected_objects`에 동시 publish하며
  충돌** → 둘 다 아무것도 발행 안 하는 상태로 멈춰버림(실측 확인,
  원인 불명). 이 충돌을 겪은 인스턴스는 재시작해도 한동안 이상
  동작할 수 있어 완전히 새로 띄우는 게 안전함.
- **권장 방법**: 여러 버전을 동시에 비교하고 싶으면, 시각화/실제
  파이프라인에 연결하는 커스텀 포트 조합을 쓰지 말고 **표준 포트
  8002 하나에 원하는 모델을 올렸다 내렸다 하면서 순서대로 비교**하라
  (이번에 실제로 그렇게 전환해서 성공함). 순수 raw 검출 개수만
  비교하려면 `perception_node_sim`을 거치지 않고 카메라 프레임을
  직접 캡처해서 두 포트에 동시 POST하는 스크립트(1회성, 저장소 밖)로
  충분하다 — 실제 인식 파이프라인을 건드릴 필요 없음.

## 모델 실험 비교
- **`vlm_boxyolo.py`** — 텍스트 인코더 없는 폐쇄 어휘(box 전용) YOLO
  서버. `perception_node.py`와 스키마 호환되는 드롭인 서버로 설계됨.
  RGB→BGR 변환 명시적 처리, `half=False` 강제(FP16이 검출 누락 유발
  — [[perception_calibration]]의 알려진 이슈와 동일 계열 버그).
  box 전용 가중치가 없어도 테스트 가능하도록 stock `yolov8n.pt`(COCO)
  플레이스홀더 모드 지원.
- **`vlm_yoloworld.py`** — `ultralytics.YOLOWorld` 기반 개방 어휘 대안.
  Moondream2 VLM 서버와 동일 API 스키마를 대체하기 위한 것. 라벨별
  순차 추론이 아니라 한 번의 forward pass로 전체 라벨 검출(RTX 4090
  기준 100+FPS). CLIP 텍스트 인코더가 다중 단어 라벨을 잘못
  토큰화하는 문제를 라벨 리맵 테이블로 우회, 클래스 임베딩
  캐싱(`set_classes()` 반복 호출 방지).
- **`realsense_demo.py`** — Windows용 라이브 데모 클라이언트. 1~3대의
  검출 서버 클러스터에 부하분산 디스패치, 컬러/뎁스 정렬, 클러스터링을
  통한 처리량 확장을 시연하는 목적이 강함(창 제목이 "MoonDream2
  Cluster" 하드코딩) — 박스 전용이 아니라 COCO류 범용 물체 대상.
  발표/벤치마킹용 데모 성격이 짙고 지속 유지되는 운영 도구는 아닌
  것으로 보임.

## `env_check.py`
학습/추론 시작 전 사전점검 스크립트: torch/CUDA/VRAM,
`ultralytics` import 가능 여부, `pyrealsense2` + 실제 1프레임
캡처 테스트, 선택적 `rclpy` import 체크, 선택적
`--check-server URL` 헬스체크. pass/fail/warn 표만 출력하고 학습/추론은
안 함.

## 일회성 vs 반복 사용 도구
- **반복 사용 도구**(범용 CLI): `box_dataset_capture.py`,
  `sam_labeler.py`, `env_check.py`.
- **운영 서버/클라이언트**(계속 유지보수됨): `visualize_detection.py`,
  `vlm_boxyolo.py`, `vlm_yoloworld.py` — `vlm_boxyolo.py`에는 "2026-07
  두 사본 통합" 같은 병합 이력 주석이 있어 실제로 계속 손보고 있는
  파일임을 알 수 있음.
- **데모/벤치마크 성격이 강한 것**: `realsense_demo.py`.

## 관련 문서
- [[perception_calibration]] — 이 도구들이 만든 모델을 실제로 소비하는
  운영 파이프라인
- [[repo_layout]]
