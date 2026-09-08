# pipeline_figure — grasp 파이프라인 시각화

한 프레임에 대해 RGB → SAM/GT mask → point cloud → CGN 후보 → 선택 grasp 4패널.

## 사용

```bash
source ~/grasp/cgn_venv/bin/activate     # torch + CGN + ultralytics
export CGN_REPO=~/grasp/contact_graspnet_pytorch
PANEL_OUT=/tmp/panels python3 make_panels.py <capture.npz> <bbox.json> <label>
# npz 에 'seg' (GT segmap, HxW) 있으면 사용, 없으면 bbox → SAM
```

- `capture.npz` : `depth_m`(H,W) + `color`/`rgb`(H,W,3) + `K`(fx,fy,cx,cy) [+ `seg`]
- `bbox.json`   : `{"bboxes": [[x0,y0,x1,y1]]}`
- 출력 `PANEL_OUT/{A,B,C,D}.png` + `stats.json`

## 패널

| | 내용 |
|---|---|
| A | RGB + YOLO bbox + SAM/GT mask 오버레이 |
| B | 중력 정렬 측면도 (RANSAC 테이블 평면) + 선택 grasp 하강 |
| C | 전체 CGN 후보 (Π자 그리퍼 마커, 색 = graspness) RGB 위 투영 |
| D | 선택 grasp 강조 + approach 화살표. cost = `score − 0.5·θ⁴ − 1.4·d_com` |

그리퍼 마커: `cc = t + d·a` (contact), fingertip `cc ± w/2·b`, 손가락은 `-a` 방향(위).
CGN 4x4 의 z-col(a) = approach = 물체를 향함.

## HTML

`grasp_pipeline.html` — box(실물) + bottle(Isaac Sim) 2예제 아티팩트. base64 패널 임베드.
`make_panels.py` 로 패널 재생성 후 스크립트로 재빌드.
