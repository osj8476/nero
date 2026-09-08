#!/usr/bin/env python3
"""C2/C3: CGN grasp -> pick_ik reachability -> STOMP plan. move_group 이 떠 있어야 함."""
import json, sys, math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (RobotState, Constraints, PositionConstraint, OrientationConstraint,
                             MotionPlanRequest, WorkspaceParameters, BoundingVolume)
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

GRASPS = json.load(open("/home/bpdl/grasp/isaac_grasps_base2.json"))["grasps"]
ARM_JOINTS = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
SEED = [0.0,-0.9991,0.0,1.3986,0.0,0.0,1.5491]

def mk_pose(g):
    p = PoseStamped(); p.header.frame_id = "base_link"
    p.pose.position = Point(x=g["position_xyz"][0], y=g["position_xyz"][1], z=g["position_xyz"][2])
    q = g["quat_xyzw"]
    p.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
    return p

def main():
    rclpy.init()
    n = rclpy.create_node("c2c3_test")
    ik = n.create_client(GetPositionIK, "/compute_ik")
    ik.wait_for_service(timeout_sec=10)
    seed_state = RobotState()
    seed_state.joint_state = JointState(name=ARM_JOINTS, position=SEED)

    reachable = []
    for g in GRASPS:
        req = GetPositionIK.Request()
        req.ik_request.group_name = "arm"
        req.ik_request.robot_state = seed_state
        req.ik_request.pose_stamped = mk_pose(g)
        req.ik_request.timeout.sec = 1
        req.ik_request.avoid_collisions = True
        fut = ik.call_async(req); rclpy.spin_until_future_complete(n, fut, timeout_sec=6)
        res = fut.result()
        code = res.error_code.val if res else None
        sol = None
        if code == 1:
            js = res.solution.joint_state
            sol = [round(js.position[js.name.index(j)],3) for j in ARM_JOINTS]
            reachable.append((g, sol))
        print("  grasp#%d score=%.3f  IK code=%s  sol=%s" % (g["rank"], g["score"], code, sol))

    if not reachable:
        print("\nNo reachable grasp. C2 FAIL.")
        n.destroy_node(); rclpy.shutdown(); return
    print(f"\nC2 OK: {len(reachable)}/{len(GRASPS)} reachable")

    # C3: STOMP plan to the best reachable
    g, sol = reachable[0]
    ac = ActionClient(n, MoveGroup, "/move_action")
    ac.wait_for_server(timeout_sec=10)
    goal = MoveGroup.Goal()
    r = MotionPlanRequest()
    r.group_name = "arm"
    r.pipeline_id = "stomp"
    r.num_planning_attempts = 3
    r.allowed_planning_time = 10.0
    r.max_velocity_scaling_factor = 0.3
    r.max_acceleration_scaling_factor = 0.3
    r.start_state = seed_state
    r.workspace_parameters = WorkspaceParameters()
    r.workspace_parameters.header.frame_id = "base_link"
    r.workspace_parameters.min_corner.x = -2.0; r.workspace_parameters.min_corner.y = -2.0; r.workspace_parameters.min_corner.z = -2.0
    r.workspace_parameters.max_corner.x = 2.0; r.workspace_parameters.max_corner.y = 2.0; r.workspace_parameters.max_corner.z = 2.0
    c = Constraints()
    jc_names = ARM_JOINTS
    from moveit_msgs.msg import JointConstraint
    for jn, jv in zip(jc_names, sol):
        j = JointConstraint(); j.joint_name = jn; j.position = jv
        j.tolerance_above = 0.01; j.tolerance_below = 0.01; j.weight = 1.0
        c.joint_constraints.append(j)
    r.goal_constraints.append(c)
    goal.request = r
    goal.planning_options.plan_only = True
    fut = ac.send_goal_async(goal); rclpy.spin_until_future_complete(n, fut, timeout_sec=15)
    gh = fut.result()
    if not gh.accepted:
        print("STOMP goal rejected"); n.destroy_node(); rclpy.shutdown(); return
    rf = gh.get_result_async(); rclpy.spin_until_future_complete(n, rf, timeout_sec=30)
    result = rf.result().result
    traj = result.planned_trajectory.joint_trajectory
    print(f"\nC3 STOMP: error_code={result.error_code.val}  waypoints={len(traj.points)}  "
          f"duration={traj.points[-1].time_from_start.sec + traj.points[-1].time_from_start.nanosec*1e-9:.2f}s" if traj.points else f"\nC3 STOMP error_code={result.error_code.val} (no traj)")
    if traj.points:
        # dump trajectory for C4 (PC replay into Isaac Sim)
        out = {"joint_names": list(traj.joint_names),
               "points": [{"positions": list(p.positions),
                           "time": p.time_from_start.sec + p.time_from_start.nanosec*1e-9}
                          for p in traj.points],
               "grasp": g}
        json.dump(out, open("/home/bpdl/grasp/c3_trajectory.json","w"), indent=1)
        print("  -> /home/bpdl/grasp/c3_trajectory.json")
    n.destroy_node(); rclpy.shutdown()

if __name__ == "__main__":
    main()
