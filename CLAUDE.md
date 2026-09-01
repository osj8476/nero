# NERO Pick & Place — 프로젝트 규칙

이 파일은 세션 시작 시 항상 로딩된다. mcp tool 응답 JSON 필드보다
우선순위가 높으므로, 반복적으로 문제가 됐던 규칙은 여기 적는다.

## 박스 높이 (2026-07-23 확정)
박스 높이는 항상 0.05m 고정값이다 (`box_height_m` 필드로 반환됨,
`mcp_robot_server.py`의 `DEFAULT_BOX_HEIGHT_M` 상수와 동일).
**스캔 결과에 물체가 안 보이거나 애매해도, 높이를 재확인하기 위해
재스캔하거나 go_home으로 복귀하지 마라.** 물체 자체가 안 보이는 문제와
높이 불확실성은 별개다. 후자를 이유로 전자의 조치(재스캔/홈복귀)를
반복 호출하지 마라.

## 그립 각도/TCP 오프셋 (2026-06-30 사고, 재발 방지)
top 그립과 side 그립은 서로 다른 TCP offset 상수를 쓴다
(`TOP_TCP_OFFSET`은 `planning_node.py`, `SIDE_TCP_OFFSET`은
`grasp_kinematics.py`에 정의 — 파일은 다르지만 값은 우연히 같다).
하나로 통일하면 다른 하나가 깨진다 — 실제로 이 상수를 하나로 합쳤다가
top_down place가 전부 ABORTED난 사고가 있었다. 이 두 상수는 **절대
같은 값으로 합치지 마라.**

**SIDE_TCP_OFFSET은 아직 실측 미검증 상태다** (2026-08 기준, top값을
임시로 재사용 중). side 그립 관련 수치 조정은 이 값이 정식 검증되기
전까지 보류한다.

## 그립 형태(top/side/pinch)가 애매할 때 — infer_grasp 필수 (2026-08-26 추가)
`pick_object`를 `grasp_dir='auto'`(기본값)로 부르거나, Claude가 임의로
top/side/pinch 중 하나를 추측해서 지정하지 마라. approach 실패 후
에러 메시지나 정황만으로 "이게 top이었나 side였나"를 역추론하는 것도
신뢰할 수 없다 — 실제로 이 방식으로 오판한 사고가 있었다(cup pick 실패를
side 문제로 잘못 짚었다가, 실제로는 top-down 시도였던 것으로 드러남,
2026-08-26). **그립 형태가 명확하지 않은 물체는 pick 전에 반드시
`infer_grasp(label)`을 먼저 호출해서 VLM이 추론한 그립 형태를 확인하라.**
LABEL_GRASP_HINT 같은 서버 내부 휴리스틱값을 믿고 넘어가지 말 것 — 그
휴리스틱이 실제로 어떤 grasp_dir을 골랐는지조차 로그 메시지 버그로
신뢰 못 하는 경우가 있었다.

**주의 — VLM의 `grasp_type`을 `grasp_dir`에 그대로 문자열 매칭하지 마라
(2026-08-26 추가, 개념 불일치 발견).** VLM의 PINCH는 "손끝으로 얇은
단면을 잡는다"는 일반적 분류학 용어이고, 이 프로젝트의 `grasp_dir='pinch'`
는 특정 실측 각도(`roll=90°,pitch=180°` — 손잡이 같은 **가로로 놓인**
얇은 물체용으로 검증된 값)를 가리키는, 이름만 같은 별개의 것이다. 병처럼
**세로로 긴** 물체를 VLM이 PINCH로 분류해도 그 각도 그대로 쓰면 기하학적으로
안 맞는다. `infer_grasp` 응답의 `grasp_type`과 `orientation`을 같이 보고
아래 표로 변환한 값을 `pick_object`/`slide_object`의 `grasp_dir`에 넣어라:

| infer_grasp 응답 | 실제로 넘길 grasp_dir |
|---|---|
| grasp_type=TOP | `top` |
| grasp_type=SIDE | `side` |
| grasp_type=PINCH, orientation=HORIZONTAL (손잡이/바 등 가로로 놓인 물체) | `pinch` |
| grasp_type=PINCH, orientation=VERTICAL (병 등 세로로 긴 물체) | `side` (세로 원통을 감싸쥐는 덴 side 공식이 기하학적으로 더 맞음) |

**VLM의 grasp_type=TOP을 z높이 확인 없이 그대로 믿지 마라 (2026-08-26 추가,
사용자 실측 지적).** `infer_grasp`는 크롭 이미지의 2D 형태만 보고 판단한다
— 그 물체의 실제 z좌표(로봇 팔 기준 "허리 높이" 대역인지)는 크롭 이미지에
안 담기는 정보라 VLM이 원천적으로 고려할 수 없다. 이 로봇은 z가 대략
0.3~0.5m대(로봇 허리 높이)인 물체는 top-down 접근이 IK 실패하기 쉽고
side가 되는 경우가 실측으로 여러 번 확인됐다(2026-08-26, 책 z=0.388에서
재현). **`grasp_type=TOP`이 나와도, 물체의 z좌표(YOLO center_3d.z 또는
ground_object의 base_link_point.z)가 이 대역이면 곧이곧대로 top으로
pick하지 말고 side로 시도하거나, 최소한 top 실패 시 곧바로 side로
전환하라.** 이 z 대역의 정확한 경계는 아직 체계적으로 측정된 게 아니라
이번 세션에서 반복 관찰된 경험적 범위다 — 확정값 아님.

**접근각(`side_approach_deg`)도 infer_grasp 응답에서 가져와라 (2026-08-26 추가).**
grasp_dir이 `side`/`pinch`면, `infer_grasp` 응답의 `suggested_side_approach_deg`
필드를 그대로 `pick_object`/`slide_object`의 `side_approach_deg`에 넣어라.
이건 VLM이 이미지에서 본 대략적인 접근 방향(FRONT/LEFT/RIGHT/BACK)을 각도로
바꾼 추정값일 뿐이지 정밀한 계산값이 아니다 — 틀려도 서버의 접근각 후보
자동 탐색(±15~90° 스윕)이 안전망 역할을 하니 그냥 그대로 넘기면 된다.
구버전 VLM 서버(approach_direction 필드 없음)는 항상 0.0(FRONT)으로 채워져서
기존 동작과 동일하게 유지된다.

## 여러 박스를 다루는 작업(쌓기 등) — box_index 필수
`pick_object(from_scan=True)`를 여러 번 연속 호출할 때는 반드시
`box_index`를 명시하라. 생략하면 "스캔 큐 맨 앞 항목"을 집는데, 이미
다른 스캔된 박스의 (x,y) 위에 무언가를 place한 뒤 그 자리를 "다음
큐 항목"으로 착각해서 방금 쌓은 박스를 다시 집어버리는 사고가 실측
확인됐다(2026-07-22). 매 `pick_object` 응답의 `remaining_scanned_boxes`
에서 원하는 항목의 인덱스를 확인하고 다음 호출에 그대로 넘겨라.

## 놓을 곳은 반드시 pick 전에 미리 스캔/좌표 확보 (2026-08-31 추가)
**물체를 집은 뒤에 놓을 곳(바구니/박스/서랍 등)을 찾으려 하지 마라.**
그리퍼로 물체를 쥐면 카메라 시야가 그 물체나 그리퍼 자체에 가려져서
`ground_object`/`analyze_scene`/`list_detected_objects`가 잘 안 잡히거나
전혀 못 잡는 경우가 실측 반복 확인됐다. **순서를 반드시 지켜라: (1)
놓을 목표(바구니 등)의 좌표를 먼저 `ground_object`나
`list_detected_objects`로 확보 → (2) 그 다음에 집을 물체를 `pick_object`
→ (3) 미리 확보해둔 좌표로 `place_object`.** pick 이후에 놓을 곳을
재탐색해야 하는 상황이 오면(좌표를 못 구했거나 씬이 바뀐 경우), 물체를
쥔 채로 팔을 크게 움직이는 재탐색은 충돌 위험이 있으니 신중히 판단할 것.

## place_object 좌표 자동 보정
`place_object`가 성공했을 때 응답의 `place_pos`가 요청한 좌표와 다를
수 있다 (서버가 도달불가 지점을 감지해 자동으로 ±0.05m 시프트하거나
z를 낮췄을 경우 — `requested_place_pos` 필드로 원래 요청값을 확인
가능). **다음 작업(그 위에 쌓기 등)의 기준 좌표는 반드시 응답의
`place_pos`(실제 좌표)를 써야 한다.** 요청 좌표를 그대로 믿고 이어서
계산하면 어긋난다.

## place_object 폐루프 검증 (2026-08 추가)
`place_object` 응답의 `status: "success"`는 로봇 동작이 물리적으로
끝났다는 뜻이지, 물체가 실제로 목표 위치에 잘 안착했다는 보장이 아니다.
서버가 그리퍼를 열고 물러난 직후 카메라로 재확인한 결과가
`placement_verified`(true/false/null) + `verification_reason` 필드로
같이 온다.
- `placement_verified: false`가 뜨면, 이 물체를 기준으로 후속 작업
  (그 위에 쌓기 등)을 계속 진행하지 마라. `list_detected_objects`로
  실제 위치를 다시 확인하거나 사용자에게 알려라.
- `placement_verified`가 없거나 null이면 perception 데이터가 오래됐거나,
  **박스 바로 위 근접 top-down 각도라 카메라가 이 시점에 아무것도
  못 잡은 경우**다(알려진 인식 사각지대 — 2026-08 실측 확인). "물체가
  없다"고 단정하지 말고, 중요한 작업이면 `list_detected_objects`나
  `scan_for_boxes`로 다른 각도에서 재확인하라. false와 null을 같은
  걸로 취급해서 자동으로 실패 처리하지 마라 — 신뢰도가 다르다.
- z 인식 노이즈가 최대 26mm까지 관측된 적이 있어 z 오차 판정은
  관대하게 잡혀 있다. `verification_reason`에 "z가 XXmm 어긋남"이
  뜨면 심각한 게 아닐 수 있으니 xy 오차와 같이 보고 판단하라.

## 홈 복귀 직후
`go_home` 완료 응답을 받았으면 그 직후 첫 `pick_object`/`place_object`
호출까지 서버가 자체적으로 정착 시간을 두고 있다(설계상 처리됨).
타임아웃이 뜨면 원인 불명 상태로 재시도를 반복하지 말고, 1회 재시도
후에도 실패하면 사용자에게 보고하라.

## 인지 모델 서버(vlm_boxyolo.py) 포트 전환 시 주의 (2026-08-12 사고)
v5/v6 등 다른 가중치를 비교하려고 `perception_node_sim`을 커스텀 포트로
띄우면 두 가지 문제가 실측 확인됐다:
1. `BOX_SERVER_URL`과 `BOX_HEALTH_URL`은 **서로 다른 환경변수**다
   (둘 다 기본값 포트 8002). 하나만 바꾸면 헬스체크가 죽은 옛 포트만
   보다가 60초 뒤 타임아웃으로 죽는다.
2. `perception_node_sim`/`vlm_boxyolo`가 특정 터미널에서 죽으면 기본
   설정(v5, 포트 8002)으로 **자동 재기동되는 동작이 관찰됨**(정확한
   메커니즘 미확인). 이 상태에서 커스텀 포트로 별도 인스턴스를 또
   띄우면 두 인스턴스가 같은 토픽에서 충돌해 `/detected_objects`가
   아예 발행이 멈춘다.

**여러 모델을 비교할 땐 커스텀 포트로 병렬 실행하지 말고, 표준 포트
8002 하나에 모델을 바꿔 올렸다 내렸다 하면서 순서대로 비교하라.**
순수 raw 검출 개수만 비교하려면 `perception_node_sim`을 거치지 않고
프레임을 직접 캡처해서 여러 포트에 동시 POST하는 1회성 스크립트로
충분하다. 상세: `docs/wiki/perception_dev_tools.md`의 "운영 중 모델
서버 포트 전환 시 주의사항" 참고.

## estimate_object_geometry의 major_axis_yaw_deg — 신뢰 불가 (2026-08-31 확인)
`estimate_object_geometry` 툴이 반환하는 `major_axis_yaw_deg`(PCA 장축
방위각)는 **접근각 결정에 쓰지 마라.** box를 여러 각도로 재배치하며
실측한 결과, 실제 회전과 무관하게 매번 82~89도 근처로 편향 수렴하는
현상이 확인됐다(NoOp segmentation이 bbox를 그대로 마스크로 써서 배경이
섞여 들어가는 게 유력한 원인으로 추정 — 상세: `docs/wiki/grasp_geometry_pipeline.md`).
기존 `angle_base_deg`(Hough)가 상대적으로 더 낫지만 이것도 완전히
믿을 건 아니다. 접근각도는 당분간 사용자가 직접 지정한 값을 쓴다.

## 참고 문서
- 그립 각도 선택 알고리즘의 설계 이력/폐기된 접근/한계:
  `.claude/skills/grasp-kinematics-design/SKILL.md` 참고
  (`grasp_kinematics.py` 수정 시 반드시 먼저 읽을 것)
- 세션 작업 이력/설계 배경 위키: `docs/wiki/INDEX.md` — 아래 "세션 종료
  위키 정리" 절 참고

## 세션 종료 위키 정리 (트리거: "위키에 정리해줘" 등)
사용자가 세션 끝에 "이 세션 내용 위키에 정리해줘" 류의 요청을 하면
다음 절차를 따른다. 이건 자유 형식 요약이 아니라 아래 구조를 지켜야
다음 세션이 안 읽고 넘어가지 않는다.

1. **주제별로 나눠서 쓴다, 세션 로그로 쓰지 않는다.** 이 세션에서 다룬
   내용을 주제(컴포넌트/기능/버그) 단위로 쪼갠 뒤, `docs/wiki/<주제>.md`가
   이미 있으면 그 파일의 "이력" 섹션에 날짜(`YYYY-MM-DD`)를 달아 append하고,
   없으면 `docs/wiki/_TEMPLATE.md` 구조로 새로 만든다.
2. **CLAUDE.md에 이미 확정 규칙으로 있는 내용은 위키에 다시 쓰지 않는다.**
   중복되면 나중에 둘이 어긋난다 — 위키 문서에서는 해당 규칙을 링크만 걸어라.
3. **"왜"를 반드시 남긴다.** 무엇을 바꿨는지보다, 왜 그렇게 결정했는지/
   무엇을 시도했다가 버렸는지가 다음 세션에 더 값어치 있다.
4. 문서를 새로 만들거나 크게 갱신했으면 `docs/wiki/INDEX.md`에 한 줄
   (`- [제목](파일.md) — 요약`)을 추가/갱신한다. INDEX.md 자체에는 본문을
   쓰지 않는다.
5. 어느 정도로 실측/검증된 사실인지(확정 vs 임시/미검증)를 표시해서, 다음
   세션이 미검증 값을 확정값처럼 쓰지 않게 한다.
6. 정리 후 어떤 파일을 새로 만들었는지/어떤 파일을 갱신했는지 사용자에게
   짧게 보고한다.

새 세션에서 어떤 주제(그립, 스캔, place 검증, 특정 노드 등)를 다시 건드리게
되면, 작업 시작 전에 `docs/wiki/INDEX.md`를 훑어서 관련 문서가 있는지
확인하고 있으면 읽어라.
