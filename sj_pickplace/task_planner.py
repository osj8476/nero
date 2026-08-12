#!/usr/bin/env python3
"""
task_planner.py

[신설] stack_boxes(여러 박스를 지정 좌표에 순서대로 쌓는 MCP 툴)의 pick 단계에
경량 백트래킹을 추가하는 순수 모듈. grasp_kinematics.py, placement_verification.py
와 동일한 컨벤션(ROS 의존성 없음, python3 -m unittest로 단독 검증 가능)을 따른다.

## 설계 원칙 (핵심, 절대 어기지 말 것)
pick 실패는 "어떤 박스를 골랐느냐"에 좌우되는 문제라 후보를 바꿔볼 가치가
있다. place 실패는 tier의 목표 좌표(base_pos + tier*box_height_m, 어떤
박스를 들고 있든 동일)의 문제라 박스를 바꿔도 해결되지 않는다. **따라서
이 백트래킹은 오직 pick 단계의 박스 후보 재선택에만 적용하고, place
실패 및 placement_verified is False에서의 무조건 중단은 allow_reorder
값과 무관하게 항상 동일하게 유지한다.**

## 콜백 주입 패턴
grasp_kinematics.YawCandidateSelector가 이미 쓰는 "ROS 연산을 콜백으로
주입해서 순수 유지" 패턴을 재사용한다. pick_fn/place_fn/remove_fn은
planning_node.py의 _do_pick/_do_place/self._scanned_boxes.remove를
그대로 넘겨받는 콜백이다.

## 책임 밖
IK/FK 서비스 호출, 실제 이동 실행, ROS 로깅 -- 전부 planning_node.py가
run_stack_plan()의 반환값(dict)을 보고 처리한다.
"""
import math
from dataclasses import dataclass, field, asdict

from .grasp_kinematics import (
    resolve_grasp_quat, side_reachability_check, SequenceRejected,
)

# 전체 스택 호출 동안 공유하는 pick 후보 재시도 예산 (tier별 아님).
# TIMEOUT_STACK_PER_BOX 예산이 박스당 이미 빠듯하므로 무한 재시도를 막는다.
DEFAULT_MAX_BACKTRACK_ATTEMPTS = 2


@dataclass
class PlanResult:
    """run_stack_plan()의 반환 형태. dataclasses.asdict()로 JSON화 가능한
    dict로 변환된다 (_publish_result가 extra로 그대로 병합할 수 있게)."""
    status: str
    tiers: list = field(default_factory=list)
    skipped_candidates: list = field(default_factory=list)
    backtrack_attempts_used: int = 0


def _reach_band(r: float, min_reach_r_m: float, boundary_r_m: float) -> int:
    """도달범위 밴드 판정. 0=편안(>=boundary), 1=경계링, 2=데드존(<min_reach).
    낮을수록 랭킹에서 우선한다. 하드 리젝이 아니라 순서 힌트일 뿐이다."""
    if r >= boundary_r_m:
        return 0
    if r >= min_reach_r_m:
        return 1
    return 2


def _prefilter_side_unreachable(candidates, grasp_dir, target_label,
                                 resolve_grasp_quat_fn):
    """[무료 사전 필터, 예산 안 씀] side 그립으로 확정 거부될 게 뻔한
    fallback 후보를 미리 걸러낸다 (SIDE_MIN_DIST 미만 -- 시도해봐야 100%
    실패하는 것을 알고 있으므로 예산을 낭비하지 않음).

    주의: requested_boxes[tier] 자체는 이 함수에 절대 넣지 말 것 -- 항상
    실제로 시도해서 기존 실패 경로/타이밍을 그대로 보존해야 한다 (호출부
    run_stack_plan에서 이미 분리해서 처리).

    Returns:
        (filtered: list, skipped: list[(box, reason)])
    """
    filtered = []
    skipped = []
    for box in candidates:
        pos = {'x': box['x'], 'y': box['y'], 'z': box['z']}
        _, is_side = resolve_grasp_quat_fn(grasp_dir, pos, target_label)
        if is_side:
            ok, reason = side_reachability_check(pos)
            if not ok:
                skipped.append((box, reason))
                continue
        filtered.append(box)
    return filtered, skipped


def _rank_candidates(candidates, grasp_dir, target_label,
                      min_reach_r_m, boundary_r_m, resolve_grasp_quat_fn):
    """fallback 후보를 (1) top-down 우선(side보다), (2) 도달범위 밴드
    (편안 > 경계링 > 데드존), (3) confidence 높은 순으로 정렬한다.
    이건 순서 힌트일 뿐 하드 리젝은 아니다 (사전 도달가능성 판정 툴이
    따로 없고 "시도해봐야 안다"는 기존 원칙 유지).
    """
    scored = []
    for box in candidates:
        pos = {'x': box['x'], 'y': box['y'], 'z': box['z']}
        _, is_side = resolve_grasp_quat_fn(grasp_dir, pos, target_label)
        r = math.hypot(pos['x'], pos['y'])
        band = _reach_band(r, min_reach_r_m, boundary_r_m)
        conf = box.get('confidence', 0.0) or 0.0
        # 정렬키: is_side(False가 먼저) < band(작을수록 먼저) < -conf(클수록 먼저)
        scored.append((bool(is_side), band, -conf, box))
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return [item[3] for item in scored]


def run_stack_plan(
    requested_boxes, fallback_pool, target_label, grasp_dir,
    base_pos, box_height_m,
    pick_fn, place_fn, remove_fn,
    min_reach_r_m, boundary_r_m,
    resolve_grasp_quat_fn=resolve_grasp_quat,
    allow_reorder=False,
    max_backtrack_attempts=DEFAULT_MAX_BACKTRACK_ATTEMPTS,
) -> dict:
    """box_refs(requested_boxes)를 순서대로(맨 아래부터) base_pos 위치에
    쌓는다. allow_reorder=True면 pick 실패 시 fallback_pool에서 대체
    후보를 시도한다 (place 단계는 후보 교체 없음, 설계 원칙 참고).

    Args:
        requested_boxes: 호출자가 지정한 순서대로의 박스 dict 리스트
            (self._scanned_boxes 안의 실제 레퍼런스).
        fallback_pool: allow_reorder=True일 때 대체 후보로 쓸 수 있는
            같은 라벨의 나머지 박스 dict 리스트 (requested_boxes와
            겹치지 않아야 함 -- 호출부 책임).
        target_label, grasp_dir: 스택 전체에 적용되는 라벨/자세.
        base_pos: {'x','y','z'} 맨 아래 tier가 놓일 좌표.
        box_height_m: tier당 높이 증분.
        pick_fn: (pos, quat, is_side, label, skip_reacquire, angle_deg)
            -> align_angle_deg. 실패 시 SequenceRejected 또는 Exception.
            (planning_node._do_pick과 동일 시그니처)
        place_fn: (pos, quat, is_side) -> {'actual_place_pos',
            'placement_verified', 'verification_reason'}. 실패 시
            SequenceRejected 또는 Exception. (planning_node._do_place와
            동일 시그니처)
        remove_fn: (box) -> None. 성공적으로 집은 박스를 큐에서 제거.
        min_reach_r_m, boundary_r_m: planning_node의 워크스페이스 반경
            상수 (중복정의 방지를 위해 호출부가 주입).
        resolve_grasp_quat_fn: grasp_kinematics.resolve_grasp_quat 기본값
            (테스트에서 가짜로 교체 가능).
        allow_reorder: False(기본)면 오늘 동작과 100% 동일 -- 후보 체인이
            requested_boxes[tier] 하나뿐이다.
        max_backtrack_attempts: attempt_idx>0인 후보(=fallback 대체
            시도)에만 소모되는 예산. 전체 stack 호출에서 공유(tier별
            아님). 예산 소진 시 그 이후 후보는 아예 시도하지 않는다.

    Returns:
        dict(status='success'|'partial', tiers=[...],
             skipped_candidates=[...], backtrack_attempts_used=int)
    """
    tiers = []
    skipped_candidates = []
    used_ids = set()       # id() of box dict가 이미 성공적으로 pick됨
    excluded_ids = set()   # id() of box dict가 이미 실패로 확인됨(재제시 금지)
    backtrack_attempts_used = 0

    for tier, requested_box in enumerate(requested_boxes):
        if not allow_reorder:
            # [설계 원칙] allow_reorder=False면 오늘과 100% 동일 동작 --
            # 후보 체인은 requested_boxes[tier] 하나뿐이다.
            chain = [requested_box]
        else:
            avail = [
                b for b in fallback_pool
                if id(b) not in used_ids and id(b) not in excluded_ids
                and b is not requested_box
            ]
            # [무료 사전 필터] side 그립으로 확정 거부될 fallback만 걸러냄
            # -- requested_box 자체는 이 필터를 절대 거치지 않는다.
            filtered, prefiltered_out = _prefilter_side_unreachable(
                avail, grasp_dir, target_label, resolve_grasp_quat_fn)
            for box, reason in prefiltered_out:
                skipped_candidates.append({
                    'tier': tier, 'stage': 'pick', 'box': box,
                    'reason': reason, 'prefiltered': True,
                })
                excluded_ids.add(id(box))
            ranked = _rank_candidates(
                filtered, grasp_dir, target_label,
                min_reach_r_m, boundary_r_m, resolve_grasp_quat_fn)
            chain = [requested_box] + ranked

        picked_box = None
        last_error_type = None
        last_reason = None
        for attempt_idx, candidate_box in enumerate(chain):
            if attempt_idx > 0:
                if backtrack_attempts_used >= max_backtrack_attempts:
                    # [설계] 예산 소진 -- 이후 후보는 아예 시도하지 않는다.
                    break
                backtrack_attempts_used += 1
            pos = {'x': candidate_box['x'], 'y': candidate_box['y'],
                   'z': candidate_box['z']}
            angle_deg = candidate_box.get('angle_base_deg') or 0.0
            quat, is_side = resolve_grasp_quat_fn(grasp_dir, pos, target_label)
            try:
                pick_fn(pos, quat, is_side, target_label, True, angle_deg)
                picked_box = candidate_box
                break
            except SequenceRejected as e:
                last_error_type, last_reason = 'rejected', str(e)
                skipped_candidates.append({
                    'tier': tier, 'stage': 'pick', 'box': candidate_box,
                    'reason': last_reason, 'prefiltered': False,
                })
                excluded_ids.add(id(candidate_box))
                continue
            except Exception as e:
                last_error_type, last_reason = 'failed', str(e)
                skipped_candidates.append({
                    'tier': tier, 'stage': 'pick', 'box': candidate_box,
                    'reason': last_reason, 'prefiltered': False,
                })
                excluded_ids.add(id(candidate_box))
                continue

        if picked_box is None:
            tiers.append({
                'tier': tier, 'stage': 'pick',
                'status': last_error_type or 'failed',
                'reason': last_reason or '시도 가능한 후보 없음',
                'box_used': None, 'substituted': False,
            })
            break

        used_ids.add(id(picked_box))
        remove_fn(picked_box)
        substituted = (picked_box is not requested_box)

        place_pos = {'x': base_pos['x'], 'y': base_pos['y'],
                     'z': base_pos['z'] + box_height_m * tier}
        place_quat, place_is_side = resolve_grasp_quat_fn(
            grasp_dir, place_pos, target_label)
        try:
            place_extra = place_fn(place_pos, place_quat, place_is_side)
        except SequenceRejected as e:
            # [설계 원칙] place 실패는 후보 교체 없이 그대로 중단.
            tiers.append({
                'tier': tier, 'stage': 'place', 'status': 'rejected',
                'reason': str(e), 'box_used': picked_box,
                'substituted': substituted,
            })
            break
        except Exception as e:
            tiers.append({
                'tier': tier, 'stage': 'place', 'status': 'failed',
                'reason': str(e), 'box_used': picked_box,
                'substituted': substituted,
            })
            break

        tiers.append({
            'tier': tier, 'stage': 'place', 'status': 'success',
            'box_used': picked_box, 'substituted': substituted,
            'place_pos': place_extra['actual_place_pos'],
            'placement_verified': place_extra['placement_verified'],
            'verification_reason': place_extra['verification_reason'],
        })
        if place_extra['placement_verified'] is False:
            # [설계 원칙] placement_verified is False의 무조건 중단은
            # allow_reorder 값과 절대 무관하다 -- 여기를 조건부로 만들지 말 것.
            break

    all_ok = (len(tiers) == len(requested_boxes)
              and tiers and tiers[-1]['status'] == 'success')
    overall = 'success' if all_ok else 'partial'
    result = PlanResult(
        status=overall, tiers=tiers, skipped_candidates=skipped_candidates,
        backtrack_attempts_used=backtrack_attempts_used,
    )
    return asdict(result)
