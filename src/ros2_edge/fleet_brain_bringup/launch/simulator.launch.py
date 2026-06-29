from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='semantic_sim_env',
            executable='simulator_node',
            output='screen'
        ),

        Node(
            package='redis_ros_bridge',
            executable='bridge',
            output='screen',
        )
    ])