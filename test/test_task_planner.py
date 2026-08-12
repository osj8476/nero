#!/usr/bin/env python3
"""
test_task_planner.py

sj_pickplace/task_planner.py 순수 로직 단위테스트. ROS/colcon 불필요,
독립 실행 가능:
    cd <repo> && python3 -m unittest test.test_task_planner -v

pick_fn/place_fn/remove_fn을 페이크로 주입해 시나리오별로 검증한다
(설계 문서 "테스트" 절 1~8번 시나리오와 1:1 대응).
"""
import os
import sys
import unittest

# sj_pickplace 패키지(__init__.py가 비어있어 ROS 없이도 import 가능)를
# 상대import 없이 바로 찾을 수 있도록 리포지토리 루트를 sys.path에 추가.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sj_pickplace.task_planner import (  # noqa: E402
    run_stack_plan, _rank_candidates, DEFAULT_MAX_BACKTRACK_ATTEMPTS,
)
from sj_pickplace.grasp_kinematics import SequenceRejected  # noqa: E402


MIN_REACH_R_M = 0.20
BOUNDARY_R_M = 0.30


def _box(x, y, z=0.03, confidence=0.9, angle=0.0):
    return {'x': x, 'y': y, 'z': z, 'confidence': confidence,
            'angle_base_deg': angle, 'label': 'box'}


def _key(pos):
    return (round(pos['x'], 6), round(pos['y'], 6), round(pos['z'], 6))


def _topdown_resolver(grasp_dir, pos, label=''):
    """모든 후보를 top-down으로 취급하는 기본 fake resolver."""
    return [0.0, 0.0, 0.0, 1.0], False


class FakePick:
    """pos의 (x,y,z)를 키로 하는 behavior map에 따라 성공/거부/실패를
    시뮬레이션하는 pick_fn 페이크. 기본은 'success'."""

    def __init__(self, behavior_by_xyz=None):
        self.calls = []
        self.behavior = behavior_by_xyz or {}

    def __call__(self, pos, quat, is_side, label, skip_reacquire, angle_deg):
        self.calls.append(dict(pos))
        behavior = self.behavior.get(_key(pos), 'success')
        if behavior == 'reject':
            raise SequenceRejected(f'pick 거부: {_key(pos)}')
        if behavior == 'fail':
            raise RuntimeError(f'pick 실패: {_key(pos)}')
        return angle_deg


class FakePlace:
    """pos의 (x,y,z)를 키로 하는 behavior/verified map에 따라 성공/거부/
    실패 또는 placement_verified 값을 시뮬레이션하는 place_fn 페이크."""

    def __init__(self, behavior_by_xyz=None, verified_by_xyz=None):
        self.calls = []
        self.behavior = behavior_by_xyz or {}
        self.verified = verified_by_xyz or {}

    def __call__(self, pos, quat, is_side):
        self.calls.append(dict(pos))
        key = _key(pos)
        behavior = self.behavior.get(key, 'success')
        if behavior == 'reject':
            raise SequenceRejected(f'place 거부: {key}')
        if behavior == 'fail':
            raise RuntimeError(f'place 실패: {key}')
        return {
            'actual_place_pos': pos,
            'placement_verified': self.verified.get(key, True),
            'verification_reason': 'test-ok',
        }


class FakeRemove:
    def __init__(self, queue):
        self.queue = queue
        self.calls = []

    def __call__(self, box):
        self.calls.append(box)
        try:
            self.queue.remove(box)
        except ValueError:
            pass


class TestRunStackPlan(unittest.TestCase):
    def setUp(self):
        self.base_pos = {'x': 0.30, 'y': 0.0, 'z': 0.02}
        self.box_height_m = 0.05

    def _run(self, requested_boxes, fallback_pool, pick_fn, place_fn,
              allow_reorder=False, max_backtrack_attempts=DEFAULT_MAX_BACKTRACK_ATTEMPTS,
              resolver=_topdown_resolver):
        queue = list(requested_boxes) + list(fallback_pool)
        remove_fn = FakeRemove(queue)
        result = run_stack_plan(
            requested_boxes=requested_boxes,
            fallback_pool=fallback_pool,
            target_label='box',
            grasp_dir=None,
            base_pos=self.base_pos,
            box_height_m=self.box_height_m,
            pick_fn=pick_fn,
            place_fn=place_fn,
            remove_fn=remove_fn,
            min_reach_r_m=MIN_REACH_R_M,
            boundary_r_m=BOUNDARY_R_M,
            resolve_grasp_quat_fn=resolver,
            allow_reorder=allow_reorder,
            max_backtrack_attempts=max_backtrack_attempts,
        )
        return result, remove_fn

    # 1. 전부 성공
    def test_all_success(self):
        box0 = _box(0.30, 0.0)
        box1 = _box(0.30, 0.10)
        pick_fn = FakePick()
        place_fn = FakePlace()
        result, remove_fn = self._run([box0, box1], [], pick_fn, place_fn)

        self.assertEqual(result['status'], 'success')
        self.assertEqual(len(result['tiers']), 2)
        for tier_idx, tier in enumerate(result['tiers']):
            self.assertEqual(tier['status'], 'success')
            self.assertFalse(tier['substituted'])
        self.assertEqual(result['tiers'][0]['box_used'], box0)
        self.assertEqual(result['tiers'][1]['box_used'], box1)
        self.assertEqual(result['skipped_candidates'], [])
        self.assertEqual(result['backtrack_attempts_used'], 0)
        self.assertEqual(len(pick_fn.calls), 2)
        self.assertEqual(len(place_fn.calls), 2)

    # 2. pick 거부 -> fallback 후보로 대체 성공
    def test_pick_rejected_falls_back_to_substitute(self):
        bad_box = _box(0.30, 0.0)
        good_box = _box(0.30, 0.10)
        pick_fn = FakePick({_key(bad_box): 'reject'})
        place_fn = FakePlace()
        result, remove_fn = self._run(
            [bad_box], [good_box], pick_fn, place_fn, allow_reorder=True)

        self.assertEqual(result['status'], 'success')
        tier0 = result['tiers'][0]
        self.assertEqual(tier0['status'], 'success')
        self.assertTrue(tier0['substituted'])
        self.assertEqual(tier0['box_used'], good_box)
        self.assertEqual(len(result['skipped_candidates']), 1)
        self.assertEqual(result['skipped_candidates'][0]['box'], bad_box)
        self.assertEqual(result['backtrack_attempts_used'], 1)
        # 대체된 박스(good_box)만 큐에서 제거됨, bad_box는 그대로.
        self.assertIn(bad_box, remove_fn.queue)
        self.assertNotIn(good_box, remove_fn.queue)

    # 3. 백트래킹 예산 소진 -- 예산 넘는 후보는 아예 시도 안 함
    def test_backtrack_budget_exhausted_stops_trying_further_candidates(self):
        bad_box = _box(0.30, 0.0)
        fb1 = _box(0.30, 0.10)
        fb2 = _box(0.30, 0.20)
        fb3 = _box(0.30, 0.30)  # 예산(기본 2) 초과분 -- 시도조차 안 돼야 함
        pick_fn = FakePick({
            _key(bad_box): 'reject', _key(fb1): 'reject', _key(fb2): 'reject',
            _key(fb3): 'success',
        })
        place_fn = FakePlace()
        result, _ = self._run(
            [bad_box], [fb1, fb2, fb3], pick_fn, place_fn,
            allow_reorder=True, max_backtrack_attempts=2)

        # bad_box(예산 미소모) + fb1,fb2(예산 2 소모) = 3회 호출, fb3은 호출 안 됨.
        self.assertEqual(len(pick_fn.calls), 3)
        called_keys = {_key(p) for p in pick_fn.calls}
        self.assertNotIn(_key(fb3), called_keys)
        self.assertEqual(result['backtrack_attempts_used'], 2)
        self.assertEqual(result['tiers'][0]['status'], 'rejected')
        self.assertEqual(result['status'], 'partial')

    # 4. placement_verified=False는 예산/allow_reorder와 무관하게 무조건 중단
    def test_placement_verified_false_stops_unconditionally(self):
        box0 = _box(0.30, 0.0)
        box1 = _box(0.30, 0.10)
        pick_fn = FakePick()
        place_pos_tier0 = {'x': self.base_pos['x'], 'y': self.base_pos['y'],
                            'z': self.base_pos['z']}
        place_fn = FakePlace(verified_by_xyz={_key(place_pos_tier0): False})
        result, _ = self._run(
            [box0, box1], [], pick_fn, place_fn, allow_reorder=True)

        self.assertEqual(result['status'], 'partial')
        self.assertEqual(len(result['tiers']), 1)
        self.assertFalse(result['tiers'][0]['placement_verified'])
        # tier 1의 pick_fn은 아예 호출되지 않아야 함.
        self.assertEqual(len(pick_fn.calls), 1)

    # 5. allow_reorder=False면 fallback_pool이 있어도 완전히 무시됨
    def test_allow_reorder_false_ignores_fallback_pool(self):
        bad_box = _box(0.30, 0.0)
        fb = _box(0.30, 0.10)
        pick_fn = FakePick({_key(bad_box): 'reject'})
        place_fn = FakePlace()
        result, _ = self._run(
            [bad_box], [fb], pick_fn, place_fn, allow_reorder=False)

        self.assertEqual(len(pick_fn.calls), 1)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(len(result['skipped_candidates']), 1)
        self.assertEqual(result['skipped_candidates'][0]['box'], bad_box)
        self.assertEqual(result['tiers'][0]['status'], 'rejected')
        self.assertIsNone(result['tiers'][0]['box_used'])
        self.assertEqual(result['backtrack_attempts_used'], 0)

    # 6. place 단계 실패(SequenceRejected/Exception 둘 다)는 후보 교체 없이 중단
    def test_place_failure_does_not_trigger_candidate_replacement(self):
        for behavior in ('reject', 'fail'):
            with self.subTest(behavior=behavior):
                box0 = _box(0.30, 0.0)
                fb = _box(0.30, 0.10)
                pick_fn = FakePick()
                place_pos_tier0 = {'x': self.base_pos['x'], 'y': self.base_pos['y'],
                                    'z': self.base_pos['z']}
                place_fn = FakePlace({_key(place_pos_tier0): behavior})
                result, _ = self._run(
                    [box0], [fb], pick_fn, place_fn, allow_reorder=True)

                expected_status = 'rejected' if behavior == 'reject' else 'failed'
                self.assertEqual(result['tiers'][0]['status'], expected_status)
                self.assertEqual(result['status'], 'partial')
                # place가 실패해도 pick_fn은 원래 박스 1번만 호출됨 (후보 교체 없음).
                self.assertEqual(len(pick_fn.calls), 1)
                self.assertEqual(len(place_fn.calls), 1)

    # 8. side 그립 사전 필터가 예산을 소모하지 않음
    def test_side_prefilter_does_not_consume_backtrack_budget(self):
        bad_box = _box(0.30, 0.0)  # requested, top-down으로 취급, pick 실패
        side_unreachable = _box(0.02, 0.0)  # 원점 근처 -- side로 취급하면 즉시 거부 대상

        def resolver(grasp_dir, pos, label=''):
            # side_unreachable 위치만 side로 분류하는 테스트 전용 규칙.
            is_side = (pos['x'] < 0.1)
            return [0.0, 0.0, 0.0, 1.0], is_side

        pick_fn = FakePick({_key(bad_box): 'reject'})
        place_fn = FakePlace()
        result, _ = self._run(
            [bad_box], [side_unreachable], pick_fn, place_fn,
            allow_reorder=True, resolver=resolver)

        # side_unreachable은 사전 필터에서 걸러져 pick_fn이 아예 호출되지 않음.
        self.assertEqual(len(pick_fn.calls), 1)
        self.assertEqual(result['backtrack_attempts_used'], 0)
        prefiltered = [sc for sc in result['skipped_candidates'] if sc.get('prefiltered')]
        self.assertEqual(len(prefiltered), 1)
        self.assertEqual(prefiltered[0]['box'], side_unreachable)


class TestRankCandidates(unittest.TestCase):
    """7. _rank_candidates 랭킹 로직 단위 테스트
    (top-down > side, 도달범위 밴드, confidence 순서)."""

    def _resolver(self, grasp_dir, pos, label=''):
        # 테스트 전용 규칙: x < 0 이면 side로 취급.
        return [0.0, 0.0, 0.0, 1.0], pos['x'] < 0

    def test_topdown_before_side_reach_band_and_confidence_order(self):
        side_box = _box(-0.5, 0.0, confidence=0.99)          # side (최하위)
        deadzone_box = _box(0.05, 0.05, confidence=0.99)      # top-down, r<0.20
        boundary_box = _box(0.25, 0.0, confidence=0.5)        # top-down, 0.20<=r<0.30
        comfortable_low_conf = _box(0.35, 0.0, confidence=0.1)   # top-down, r>=0.30
        comfortable_high_conf = _box(0.32, 0.0, confidence=0.95)  # top-down, r>=0.30

        candidates = [side_box, deadzone_box, boundary_box,
                      comfortable_low_conf, comfortable_high_conf]
        ranked = _rank_candidates(
            candidates, grasp_dir=None, target_label='box',
            min_reach_r_m=MIN_REACH_R_M, boundary_r_m=BOUNDARY_R_M,
            resolve_grasp_quat_fn=self._resolver)

        # side는 항상 top-down 후보들보다 뒤에 온다.
        self.assertEqual(ranked[-1], side_box)
        # comfortable(high/low conf) 두 개가 boundary/deadzone보다 앞에 온다.
        idx = {id(b): i for i, b in enumerate(ranked)}
        self.assertLess(idx[id(comfortable_high_conf)], idx[id(boundary_box)])
        self.assertLess(idx[id(comfortable_low_conf)], idx[id(boundary_box)])
        self.assertLess(idx[id(boundary_box)], idx[id(deadzone_box)])
        self.assertLess(idx[id(deadzone_box)], idx[id(side_box)])
        # 같은 밴드(comfortable) 안에서는 confidence 높은 쪽이 먼저.
        self.assertLess(idx[id(comfortable_high_conf)], idx[id(comfortable_low_conf)])


if __name__ == '__main__':
    unittest.main()
