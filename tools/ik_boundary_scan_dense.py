#!/usr/bin/env python3
"""
side 그립(roll=0, pitch=90, yaw=atan2(y,x)) 도달 가능 영역 정밀 스캔.
x, y 전 영역(양수/음수) x 0.02m 간격, z 3개 레벨.
결과를 텍스트 파일로 직접 저장한다 (CSV + 요약).
"""
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from pymoveit2 import MoveIt2
import threading, time, math

def euler_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return [sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy, cr*cp*cy + sr*sp*sy]

rclpy.init()
node = rclpy.create_node('ik_boundary_scan_dense')
cb = ReentrantCallbackGroup()
m = MoveIt2(
    node=node,
    joint_names=['joint1','joint2','joint3','joint4','joint5','joint6','joint7'],
    base_link_name='base_link',
    end_effector_name='gripper_flange',
    group_name='arm',
    callback_group=cb,
)
ex = MultiThreadedExecutor()
ex.add_node(node)
t = threading.Thread(target=ex.spin, daemon=True)
t.start()
time.sleep(2.0)

# 물체 몸통 중앙 높이 대표값 3개 (낮음/중간/높음)
Z_LEVELS = [0.09, 0.15, 0.25]

# 0.02m 간격, x/y 모두 양수음수 전체
X_RANGE = [round(-0.45 + 0.02*i, 2) for i in range(46)]   # -0.45 ~ 0.45
Y_RANGE = [round(-0.40 + 0.02*i, 2) for i in range(41)]   # -0.40 ~ 0.40

OUT_PATH = "/home/bpdl/sj/ik_side_reachable_map.txt"

results = []
total = len(Z_LEVELS) * len(X_RANGE) * len(Y_RANGE)
count = 0
start_time = time.time()

with open(OUT_PATH, 'w') as f:
    f.write("x,y,z,ok\n")

    for z in Z_LEVELS:
        for x in X_RANGE:
            for y in Y_RANGE:
                count += 1
                # 베이스 바로 위(특이점 구간)는 스킵
                if abs(x) < 0.05 and abs(y) < 0.05:
                    continue
                yaw = math.atan2(y, x)
                quat = euler_to_quat(0.0, math.radians(90), yaw)
                result = m.compute_ik(position=[x, y, z], quat_xyzw=quat)
                ok = result is not None
                f.write(f"{x},{y},{z},{ok}\n")
                f.flush()  # 중간에 끊겨도 데이터 보존
                results.append(ok)

                if count % 200 == 0:
                    elapsed = time.time() - start_time
                    rate = count / elapsed
                    remaining = (total - count) / rate if rate > 0 else 0
                    print(f"[{count}/{total}] 진행중... "
                          f"(경과 {elapsed:.0f}s, 예상잔여 {remaining:.0f}s)")

    elapsed_total = time.time() - start_time
    ok_count = sum(results)
    summary = (f"\n=== 요약 ===\n"
              f"전체 스캔: {len(results)}개\n"
              f"OK: {ok_count}개 ({ok_count/len(results)*100:.1f}%)\n"
              f"FAIL: {len(results)-ok_count}개\n"
              f"소요시간: {elapsed_total:.0f}초\n")
    f.write(summary)
    print(summary)

print(f"\n결과 저장 완료: {OUT_PATH}")
rclpy.shutdown()
