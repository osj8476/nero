#!/usr/bin/env python3
"""
side 그립(roll=0, pitch=90, yaw=atan2(y,x)) 기준으로
x, y, z 전 영역에 대해 IK 가능 여부를 스캔하는 스크립트.

결과를 CSV로 저장해서 표/히트맵으로 분석 가능하게 한다.
"""
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from pymoveit2 import MoveIt2
import threading, time, math, csv

def euler_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return [sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy, cr*cp*cy + sr*sp*cy]

rclpy.init()
node = rclpy.create_node('ik_full_scan')
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

# 스캔 범위 (필요시 조정)
X_RANGE = [0.15, 0.20, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40, 0.45]
Y_RANGE = [-0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30]
Z_RANGE = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

results = []
total = len(X_RANGE) * len(Y_RANGE) * len(Z_RANGE)
count = 0

for x in X_RANGE:
    for y in Y_RANGE:
        yaw = math.atan2(y, x)
        quat = euler_to_quat(0.0, math.radians(90), yaw)
        for z in Z_RANGE:
            count += 1
            result = m.compute_ik(position=[x, y, z], quat_xyzw=quat)
            ok = result is not None
            results.append({'x': x, 'y': y, 'z': z, 'ok': ok})
            print(f"[{count}/{total}] x={x:.2f} y={y:.2f} z={z:.2f} -> {'OK' if ok else 'FAIL'}")

# CSV 저장
with open('/tmp/ik_scan_result.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['x','y','z','ok'])
    writer.writeheader()
    writer.writerows(results)

print("\n=== 요약 ===")
ok_count = sum(1 for r in results if r['ok'])
print(f"전체 {total}개 중 OK {ok_count}개 ({ok_count/total*100:.1f}%)")
print("결과 저장: /tmp/ik_scan_result.csv")

rclpy.shutdown()
