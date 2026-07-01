#!/usr/bin/env python3
"""
top_down 그립(quat=[0.008,0.999,0.023,0.037]) 도달 가능 높이 스캔.
x=0.2, y=-0.1 고정, z만 변화시켜 최소 안전 높이를 찾는다.
"""
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from pymoveit2 import MoveIt2
import threading, time

rclpy.init()
node = rclpy.create_node('ik_top_scan')
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

quat = [0.008, 0.999, 0.023, 0.037]

print("=== x=0.2, y=-0.1 고정, z 스캔 ===")
for z in [0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
    result = m.compute_ik(position=[0.2, -0.1, z], quat_xyzw=quat)
    print(f"z={z:.2f} -> {'OK' if result else 'FAIL'}")

print("")
print("=== 여러 x,y 위치에서 z 최소값 탐색 ===")
for x, y in [(0.2, -0.1), (0.25, -0.2), (0.3, 0.0), (0.2, 0.0)]:
    for z in [0.05, 0.08, 0.10, 0.13, 0.15, 0.18, 0.20]:
        result = m.compute_ik(position=[x, y, z], quat_xyzw=quat)
        print(f"x={x},y={y},z={z:.2f} -> {'OK' if result else 'FAIL'}")
    print("")

rclpy.shutdown()
