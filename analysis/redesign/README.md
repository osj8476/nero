# NERO grasp 파이프라인 재설계 — 분석 문서

2026-09-03~04 세션. pick&place 파이프라인 재설계(5-Phase) 진행 중 스냅샷.
원본은 세션 메모리(`~/.claude/.../memory/`), 여기는 nero 레포 백업본.

- `nero-grasp-pipeline-redesign.md` — 진단(/brutal) + 5-Phase 로드맵 + 모든 의사결정
  (pick_ik 전환, PC spike 결과, seg 벤치, 그리퍼 스펙, 차용 공식 등)
- `okrobot.md` — OK-Robot 논문 Part A~D 분석 + NERO 적용점
- `논문/contact-graspnet.md` — Contact-GraspNet 방법론 A~E + 좋은/나쁜 소식 종합
- `논문/6dof-graspnet.md` — 6-DOF GraspNet 분석 + CGN 비교 + generator 방법론 결정

관련 코드: `tools/{pc_spike_capture,pc_spike_report,seg_bench,depth_noise}.py`,
`tools/{PC_SPIKE,PHASE1_VISION}.md`, `sj_pickplace/{point_cloud,segmentation_backend,
learned_grasp_backend,grasp_pose_generator}.py`
