"""헤드리스 planning 테스트: rsp + jsp + move_group (STOMP/pick_ik). 실행 브리지 없음."""
from moveit_configs_utils import MoveItConfigsBuilder
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    mc = (
        MoveItConfigsBuilder("nero", package_name="nero_gripper_moveit_config")
        .planning_pipelines(pipelines=["ompl", "stomp", "pilz_industrial_motion_planner"],
                            default_planning_pipeline="stomp")
        .to_moveit_configs()
    )
    return LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             output="screen", parameters=[mc.robot_description]),
        Node(package="joint_state_publisher", executable="joint_state_publisher",
             output="screen", parameters=[mc.robot_description]),
        Node(package="tf2_ros", executable="static_transform_publisher", output="log",
             arguments=["0","0","0","0","0","0","world","base_link"]),
        Node(package="moveit_ros_move_group", executable="move_group", output="screen",
             parameters=[mc.to_dict()]),
    ])
