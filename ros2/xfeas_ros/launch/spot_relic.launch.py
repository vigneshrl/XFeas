"""XFeas on the real robot: ros2 launch xfeas_ros spot_relic.launch.py [config:=/path/to/spot_relic.yaml]"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default = os.path.join(get_package_share_directory("xfeas_ros"), "config", "spot_relic.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default),
        Node(package="xfeas_ros", executable="xfeas_node", name="xfeas", output="screen",
             parameters=[{"config": LaunchConfiguration("config")}]),
    ])
