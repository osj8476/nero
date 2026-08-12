#!/usr/bin/env python3
"""
build_safety_map_layer.py
--------------------------
CSV(mismatch_summary_*.csv 또는 mismatch_raw_*.csv)를 읽어서,
색깔 디스크(Cylinder)로 된 "독립적인" usd 레이어 파일을 만든다.

[2026-07 수정] z 레벨을 하나로 뭉치던 것을 그만두고, 실제 쿼리했던 높이별로
분리해서 배치함. Isaac Sim Stage 트리에서 각 높이(z_0200, z_0280, z_0310 등)
prim의 눈 아이콘을 개별적으로 켜고 꺼서 "이 스택 높이에서 워크스페이스가
어떻게 생겼는지"를 따로 확인할 수 있다. 기본은 전부 꺼진 상태(카메라/YOLO
오염 방지 + 뷰포트도 깔끔하게).

★ pxr은 Isaac Sim 전용이 아니라 Pixar가 공개한 표준 OpenUSD 라이브러리라서,
  Kit/SimulationApp 없이 `pip install usd-core` 한 순수 파이썬 venv에서
  바로 쓸 수 있다. GPU/RTX 드라이버와 완전히 무관하고, Kit 부팅이 없어
  1초 안팎으로 끝난다. start_nero_isaac_all.sh 맨 앞에서 "매번" 돌려도 부담 없음.

사용 (venv 활성화 후, pip install usd-core 되어 있어야 함):

    source ~/usd_env/bin/activate
    python3 build_safety_map_layer.py \
        --source mismatch \
        --csv "/home/bpdl/nero/analysis/convention_mismatch_output_mismatch_summary_20260729_173103.csv" \
        --out "/home/bpdl/ros2_ws/src/nero/sj_pickplace/safety_map_layer.usd"

    python3 build_safety_map_layer.py \
        --source sim_reachable \
        --csv "/home/bpdl/nero/analysis/convention_mismatch_output_mismatch_raw_20260729_173103.csv" \
        --out "/home/bpdl/ros2_ws/src/nero/sj_pickplace/safety_map_layer.usd"

--out 경로는 embed_reference_once.py로 메인 usd에 참조를 심을 때 넣은 경로와
반드시 동일해야 한다 (파일 내용만 매번 새로 덮어쓰는 것이므로 경로는 고정).

Isaac Sim에서 높이별로 켜서 보는 법:
  Stage 트리 > SafetyMapOverlay > (MismatchMap 또는 SimReachableMap) >
  z_XXXX (예: z_0200 = 0.200m) 찾아서 그 줄의 눈 아이콘 클릭.
  여러 높이 동시에 켜도 됨(비교용) — 서로 독립적으로 켜고 끌 수 있음.
"""

import argparse
import csv
from collections import defaultdict

from pxr import Usd, UsdGeom, Gf

parser = argparse.ArgumentParser()
parser.add_argument("--source", choices=["sim_reachable", "mismatch"], required=True)
parser.add_argument("--csv", required=True, help="csv 경로 (source에 맞는 파일)")
parser.add_argument("--out", required=True, help="생성할 usd 레이어 파일 경로 (고정 경로 권장)")
parser.add_argument("--z-frame", choices=["flange", "fingertip"], default="fingertip",
                     help="CSV의 z는 map_convention_mismatch.py가 /compute_ik로 조회한 "
                          "gripper_flange 기준값. 'fingertip'(기본값)을 주면 "
                          "GRIPPER_FLANGE_TO_FINGERTIP만큼 빼서 그리퍼 끝단(fingertip) "
                          "기준 높이로 라벨/배치한다. 원본 flange 값 그대로 보고 싶으면 "
                          "--z-frame flange.")
parser.add_argument("--flange-to-fingertip", type=float, default=0.1358,
                     help="flange->fingertip 오프셋(m). planning_node.py의 "
                          "GRIPPER_FLANGE_TO_FINGERTIP과 동일한 값 사용 (기본 0.1358).")
args = parser.parse_args()

Z_OFFSET_TO_FINGERTIP = args.flange_to_fingertip if args.z_frame == "fingertip" else 0.0


# ---------------------------------------------------------------
# 1. CSV 로드 & (x, y, z)별 값 집계 (z 레벨은 합치지 않고 그대로 유지 —
#    높이별로 따로 켜고 끌 수 있게 하기 위함)
# ---------------------------------------------------------------
by_z = defaultdict(lambda: defaultdict(list))  # {z: {(x,y): [val, val, ...]}}

if args.source == "sim_reachable":
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            z = round(float(row["z"]), 4)
            key = (round(float(row["x"]), 4), round(float(row["y"]), 4))
            val = 1.0 if str(row["sim_reachable"]).strip() == "True" else 0.0
            by_z[z][key].append(val)  # 각도별 여러 샘플 -> 평균
    higher_is_safe = True
    root_name = "SimReachableMap"
else:
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            z = round(float(row["z"]), 4)
            key = (round(float(row["x"]), 4), round(float(row["y"]), 4))
            by_z[z][key].append(float(row["mismatch_rate"]))  # summary는 보통 1개뿐
    higher_is_safe = False
    root_name = "MismatchMap"

z_levels = sorted(by_z.keys())
total_points = sum(len(pts) for pts in by_z.values())
print(f"[build_safety_map_layer] source={args.source}, z레벨 {len(z_levels)}개"
      f"({z_levels}), 총 {total_points}개 지점, out={args.out}")

# ---------------------------------------------------------------
# 2. 색상 계산
# ---------------------------------------------------------------
GREEN = Gf.Vec3f(0.04, 0.43, 0.18)
YELLOW = Gf.Vec3f(0.96, 0.90, 0.26)
RED = Gf.Vec3f(0.78, 0.16, 0.16)


def value_to_color(v, higher_is_safe):
    v = max(0.0, min(1.0, v))
    stops = [(0.0, RED), (0.5, YELLOW), (1.0, GREEN)] if higher_is_safe \
        else [(0.0, GREEN), (0.5, YELLOW), (1.0, RED)]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t0 <= v <= t1:
            f = (v - t0) / (t1 - t0) if t1 > t0 else 0.0
            return c0 * (1.0 - f) + c1 * f
    return stops[-1][1]


def is_critical(v, higher_is_safe):
    return (v <= 0.01) if higher_is_safe else (v >= 0.99)


def z_to_prim_name(z):
    # 0.2 -> "z_0200", 0.31 -> "z_0310" (prim 이름은 소수점/마이너스 기호 불가)
    milli = round(z * 1000)
    sign = "n" if milli < 0 else ""
    return f"z_{sign}{abs(milli):04d}"


# ---------------------------------------------------------------
# 3. 독립 usd 레이어 파일 생성 (SimulationApp 없이, 순수 USD)
#    구조: /World/{MismatchMap|SimReachableMap}/z_0200/pt_0000 ...
#    ★ USD 규칙상 부모가 invisible이면 자식은 절대 다시 visible로 못 만듦.
#      그래서 root(/World/...Map)는 기본(inherited=보임) 상태로 두고,
#      z 레벨별 그룹(z_0200 등) 각각에만 invisible을 걸어서, Stage
#      트리에서 원하는 높이 하나만 골라 켤 수 있게 함. 디스크는 실제
#      쿼리했던 높이(z) 그대로에 배치해서, 스택 높이별로 문제되는
#      구간을 직관적으로 확인할 수 있게 함(평면에 눌러 담지 않음).
# ---------------------------------------------------------------
stage = Usd.Stage.CreateNew(args.out)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

UsdGeom.Xform.Define(stage, "/World")
root_path = f"/World/{root_name}"
UsdGeom.Xform.Define(stage, root_path)  # 의도적으로 visibility 미설정 (inherited)

DISC_RADIUS = 0.018
DISC_RADIUS_CRITICAL = 0.026
DISC_HEIGHT = 0.003

# [2026-07 추가] compute_ik가 실제로 풀어준 z는 FK_LINK_NAME='gripper_flange'
# 기준이라, 시각화할 때 디스크 위치만 그리퍼 끝단 기준으로 내려서 배치한다.
# (그룹 이름/데이터/나머지 로직은 그대로 -- 표시 위치만 오프셋)
GRIPPER_FLANGE_TO_FINGERTIP = 0.1358  # planning_node.py와 동일 (실측 검증값)

for z in z_levels:
    z_group_path = f"{root_path}/{z_to_prim_name(z)}"
    z_xform = UsdGeom.Xform.Define(stage, z_group_path)
    # 기본은 안 보이게 (카메라/YOLO 오염 방지 + 뷰포트도 깔끔하게).
    # 특정 높이만 확인하고 싶으면 Stage 트리에서 이 prim의 눈 아이콘만 켜면 됨.
    z_xform.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    z_display = z - GRIPPER_FLANGE_TO_FINGERTIP  # 그리퍼 끝단 기준 높이로 변환
    points_at_z = by_z[z]
    for i, ((x, y), vals) in enumerate(points_at_z.items()):
        val = sum(vals) / len(vals)
        prim_path = f"{z_group_path}/pt_{i:04d}"
        cyl = UsdGeom.Cylinder.Define(stage, prim_path)
        radius = DISC_RADIUS_CRITICAL if is_critical(val, higher_is_safe) else DISC_RADIUS
        cyl.CreateRadiusAttr(radius)
        cyl.CreateHeightAttr(DISC_HEIGHT)
        cyl.CreateAxisAttr("Z")

        xform = UsdGeom.Xformable(cyl)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z_display))  # 끝단 기준 높이에 배치
        cyl.CreateDisplayColorAttr([value_to_color(val, higher_is_safe)])

stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
stage.GetRootLayer().Save()
print(f"[build_safety_map_layer] 저장 완료: {args.out} "
      f"(Stage 트리에서 {root_name} > z_XXXX 프림별로 눈 아이콘 켜서 높이별 확인 가능)")
