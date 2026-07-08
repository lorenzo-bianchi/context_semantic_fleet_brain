import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    simulator_launch_file = os.path.join(
        get_package_share_directory('semantic_sim_env'),
        'launch',
        'simulator.launch.py'
    )

    bridge_launch_file = os.path.join(
        get_package_share_directory('redis_ros_bridge'),
        'launch',
        'bridge.launch.py'
    )

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(simulator_launch_file)
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(bridge_launch_file)
        )
    ])
