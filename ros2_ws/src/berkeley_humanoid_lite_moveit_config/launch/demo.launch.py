"""Bring up MoveIt and RViz for the Berkeley Humanoid Lite arms, in simulation.

Starts move_group, robot_state_publisher, RViz with the MotionPlanning panel,
and the simulated trajectory controllers.  No hardware and no CAN bus is touched.

    ros2 launch berkeley_humanoid_lite_moveit_config demo.launch.py
    ros2 launch berkeley_humanoid_lite_moveit_config demo.launch.py model:=stock

Prefer ../run_demo.sh, which also pins ROS_DOMAIN_ID away from the dexhand
session on the default domain.

Models:
    v1arm    V1 forearm + 3-DOF wrist + DexHand v2 fingers on BOTH arms.
             9 DOF per arm, so this is the only model with full 6-DOF pose IK.
             Groups: left_arm, right_arm, both_arms, left_hand, right_hand
    dexhand  DexHand v2 welded to the right arm only; 5 DOF, position-only IK
    stock    original end effectors; 5 DOF, position-only IK

Drag the interactive marker on either hand and hit "Plan & Execute".  On v1arm
the marker's rotation rings work too; on the 5-DOF models orientation is ignored.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

PACKAGE = "berkeley_humanoid_lite_moveit_config"
DESCRIPTION_PACKAGE = "berkeley_humanoid_lite_description"

MODELS = {
    "v1arm": {
        "urdf": "berkeley_humanoid_lite_v1arm.urdf",
        "srdf": "config/berkeley_humanoid_lite_v1arm.srdf",
        "kinematics": "config/kinematics_v1arm.yaml",
        "limits": "config/joint_limits_v1arm.yaml",
        "controllers": "config/moveit_controllers_v1arm.yaml",
        "server_args": ["--model", "v1arm"],
    },
    "dexhand": {
        "urdf": "berkeley_humanoid_lite_dexhand.urdf",
        "srdf": "config/berkeley_humanoid_lite_dexhand.srdf",
        "kinematics": "config/kinematics.yaml",
        "limits": "config/joint_limits_dexhand.yaml",
        "controllers": "config/moveit_controllers_dexhand.yaml",
        "server_args": ["--model", "dexhand"],
    },
    "stock": {
        "urdf": "berkeley_humanoid_lite.urdf",
        "srdf": "config/berkeley_humanoid_lite.srdf",
        "kinematics": "config/kinematics.yaml",
        "limits": "config/joint_limits.yaml",
        "controllers": "config/moveit_controllers.yaml",
        "server_args": ["--model", "stock"],
    },
}


def launch_setup(context, *_args, **_kwargs):
    model = LaunchConfiguration("model").perform(context)
    if model not in MODELS:
        raise RuntimeError(f"unknown model '{model}'; choose one of {sorted(MODELS)}")
    spec = MODELS[model]

    urdf = (
        Path(get_package_share_directory(DESCRIPTION_PACKAGE)) / "urdf" / spec["urdf"]
    )
    if not urdf.is_file():
        raise RuntimeError(f"{urdf} is missing; run generate_urdf.py and rebuild")

    moveit_config = (
        MoveItConfigsBuilder("berkeley-humanoid-lite", package_name=PACKAGE)
        .robot_description(file_path=str(urdf))
        .robot_description_semantic(file_path=spec["srdf"])
        .robot_description_kinematics(file_path=spec["kinematics"])
        .joint_limits(file_path=spec["limits"])
        .trajectory_execution(file_path=spec["controllers"])
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    rviz_config = str(
        Path(get_package_share_directory(PACKAGE)) / "config" / "moveit.rviz"
    )

    return [
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[moveit_config.to_dict()],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            output="screen",
            condition=IfCondition(LaunchConfiguration("use_rviz")),
            arguments=["-d", rviz_config],
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                moveit_config.joint_limits,
                moveit_config.planning_pipelines,
            ],
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[moveit_config.robot_description],
        ),
        Node(
            package=PACKAGE,
            executable="sim_trajectory_server.py",
            output="screen",
            arguments=spec["server_args"],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            "model",
            default_value="v1arm",
            description=f"which robot to load: {', '.join(sorted(MODELS))}",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            description="set false to run move_group headless, e.g. for tests",
        ),
        OpaqueFunction(function=launch_setup),
    ])
