from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent
from launch.events import Shutdown
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression

from launch_ros.actions import Node

def generate_launch_description():
    package_name = "pgr_stereo_camera"

    ld = LaunchDescription()

    # arguments
    arg_ns = DeclareLaunchArgument('ns',
                                    default_value="camera",
                                    description="The ROS namespace to launch the camera node(s) in."
    )
    
    arg_sync = DeclareLaunchArgument('syncStereo',
                                     default_value="false",
                                     choices=['true', 'false'],
                                     description="Whether to run stereo camera's synchronized or not."
    )
    
    arg_cam_l_idx = DeclareLaunchArgument('cameraLeftIndex',
                                          default_value="0",
                                          description="Index of the left camera"
    )
    arg_cam_r_idx = DeclareLaunchArgument('cameraRightIndex',
                                          default_value="1",
                                          description="Index of the right camera"
    )

    # events
    event_shutdown_argcheck = EmitEvent(
        event=Shutdown( reason="cameraLeftIndex == cameraRightIndex" ),
        condition=LaunchConfigurationEquals( 'cameraLeftIndex', LaunchConfiguration('cameraRightIndex') )
    )

    # nodes
    node_stereo_sync = Node(
            package=package_name,
            namespace=LaunchConfiguration('ns'),
            executable='pgr_stereo_camera',
            output='screen',
            emulate_tty=True,
            condition=LaunchConfigurationEquals( 'syncStereo', 'true' ),
            )

    node_mono_left = Node(
        package=package_name,
        namespace=PathJoinSubstitution( [LaunchConfiguration('ns'), 'left'] ),
        name="MonocularCameraNode",
        executable='pgr_mono_camera',
        output='screen',
        emulate_tty=True,
        condition=LaunchConfigurationEquals( 'syncStereo', 'false' ),
        parameters=[{
            'camera.index': LaunchConfiguration('cameraLeftIndex'),
        }],
    )

    node_mono_right = Node(
        package=package_name,
        namespace=PathJoinSubstitution( [LaunchConfiguration('ns'), 'right'] ),
        name="MonocularCameraNode",
        executable='pgr_mono_camera',
        output='screen',
        emulate_tty=True,
        condition=LaunchConfigurationEquals( 'syncStereo', 'false' ),
        parameters=[{
            'camera.index': LaunchConfiguration('cameraRightIndex'),
        }],
    )

    # configure launch description
    # - arguments
    ld.add_action( arg_ns )
    ld.add_action( arg_sync )
    ld.add_action( arg_cam_l_idx )
    ld.add_action( arg_cam_r_idx )

    # - nodes
    ld.add_action( node_stereo_sync )
    ld.add_action( node_mono_left )
    ld.add_action( node_mono_right )

    # - events
    ld.add_action( event_shutdown_argcheck )

    return ld

# generate_launch_description