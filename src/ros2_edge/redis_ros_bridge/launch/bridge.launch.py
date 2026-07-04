from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Path to the config file
    config = os.path.join(
        get_package_share_directory('redis_ros_bridge'),
        'config',
        'bridge.yaml'
    )

    return LaunchDescription([
        Node(
            package='redis_ros_bridge',
            executable='bridge',
            name='bridge_node',
            emulate_tty=True,
            parameters=[config],
            output='both',
            log_cmd=True,
        )
    ])
