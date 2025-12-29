from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    panda_pkg = FindPackageShare('moveit_resources_panda_description')
    robot_desc = Command([
        'xacro ', PathJoinSubstitution([panda_pkg, 'urdf', 'panda.urdf'])
    ])

    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}]
    )
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'panda_link0']
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', PathJoinSubstitution([panda_pkg, 'config', 'panda.rviz'])]
    )

    return LaunchDescription([rsp_node, static_tf, rviz_node])