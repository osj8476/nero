#!/usr/bin/env python3
"""
align_angle_error.py
---------------------
align 단계(approach 후 재측정하는 박스 각도, pick_object 응답의
align_angle_deg)가 실제 ground truth(randomize_box_positions.py가 USD에
쓴 yaw_deg)와 얼마나 차이나는지 집계한다.

박스는 직사각형이라 모서리 방향은 90도마다 반복된다(각도 인식 파이프라인도
전부 % 90으로 다룸 -- perception_node.py, grasp_kinematics.py 참고).
그래서 GT/측정값 둘 다 mod 90으로 비교한다.

또한 world 좌표계(USD의 박스 회전 기준)와 base_link 좌표계(align_angle_deg의
기준) 사이에는 "카메라 마운트 orientation에서 비롯된 상수 오프셋"이 낄 수
있다 -- 이건 실측 노이즈가 아니라 그냥 좌표계 정의 차이라서 그립 자체에는
문제가 안 된다(grasp_kinematics의 후보 비교도 이 오프셋과 무관하게 상대적
회전량만 봄). 그래서 이 스크립트는:
  1. 전체 표본의 circular offset(상수 편향)을 추정
  2. 그 offset을 뺀 "잔차"의 분포(평균/표준편차/최대)를 진짜 노이즈로 보고
계산한다.

사용:
    python3 align_angle_error.py --gt box_gt_20260811_143000.csv \
        --trials trials.csv

    trials.csv 형식 (직접 기록): prim,align_angle_deg
        /World/TestBox,41.8
        /World/TestBox1,12.0
        ...

    또는 --pair "prim:yaw_gt:measured" 를 여러 번 줘서 즉석으로 계산 가능:
    python3 align_angle_error.py --pair "/World/TestBox:37.2:41.8" \
        --pair "/World/TestBox1:80.0:12.0"
"""
import argparse
import csv
import math


def circ_diff_90(a, b):
    """0~90 범위에서의 최소 각도 차이 (부호 없음)."""
    d = (a - b) % 90
    return min(d, 90 - d)


def circ_signed_diff_90(a, b):
    """a - b 를 (-45, 45] 범위로 감은 부호 있는 차이."""
    d = (a - b) % 90
    if d > 45:
        d -= 90
    return d


def circular_mean_offset(diffs_deg):
    """0~90 주기 각도들의 circular mean (offset 추정용)."""
    sin_sum = sum(math.sin(math.radians(d * 4)) for d in diffs_deg)
    cos_sum = sum(math.cos(math.radians(d * 4)) for d in diffs_deg)
    return (math.degrees(math.atan2(sin_sum, cos_sum)) / 4.0) % 90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', help='randomize_box_positions.py가 만든 ground truth csv')
    ap.add_argument('--trials', help='prim,align_angle_deg 형식의 csv (--gt와 prim으로 매칭)')
    ap.add_argument('--pair', action='append', default=[],
                     help='"라벨:yaw_gt_deg:measured_deg" 형식, 여러 번 지정 가능')
    args = ap.parse_args()

    samples = []  # (label, yaw_gt_mod90, measured)

    if args.gt and args.trials:
        gt = {}
        with open(args.gt) as f:
            for row in csv.DictReader(f):
                gt[row['prim']] = float(row['yaw_deg']) % 90
        with open(args.trials) as f:
            for row in csv.DictReader(f):
                prim = row['prim']
                if prim not in gt:
                    print(f'[경고] GT에 없는 prim 건너뜀: {prim}')
                    continue
                samples.append((prim, gt[prim], float(row['align_angle_deg'])))

    for p in args.pair:
        label, yaw_gt, measured = p.split(':')
        samples.append((label, float(yaw_gt) % 90, float(measured)))

    if not samples:
        ap.error('--gt/--trials 쌍이나 --pair 중 하나는 있어야 함')

    raw_diffs = [circ_signed_diff_90(m, g) % 90 for _, g, m in samples]
    offset = circular_mean_offset(raw_diffs)

    print(f'{"label":<20}{"gt(mod90)":>12}{"measured":>12}{"raw_err":>10}{"resid_err":>12}')
    residuals = []
    for label, g, m in samples:
        raw_err = circ_diff_90(m, g)
        resid = circ_diff_90(m, (g + offset) % 90)
        residuals.append(resid)
        print(f'{label:<20}{g:>12.1f}{m:>12.1f}{raw_err:>10.1f}{resid:>12.1f}')

    n = len(residuals)
    mean_r = sum(residuals) / n
    max_r = max(residuals)
    var = sum((r - mean_r) ** 2 for r in residuals) / n
    print()
    print(f'추정된 상수 오프셋(world<->base_link 기준 차이): {offset:.1f}\xb0')
    print(f'오프셋 보정 후 잔차 오차 (진짜 노이즈): '
          f'mean={mean_r:.1f}\xb0  std={math.sqrt(var):.1f}\xb0  max={max_r:.1f}\xb0  n={n}')
    print()
    print('참고: grasp_kinematics.YawCandidateSelector는 두 후보(각도, 각도+90) 중')
    print('현재 orientation과의 회전량이 더 적은 쪽을 고른다. 후보 간격이 90도이므로,')
    print('잔차 오차가 45도에 가까워질수록 "엉뚱한 후보를 고를 위험"이 커진다.')


if __name__ == '__main__':
    main()
