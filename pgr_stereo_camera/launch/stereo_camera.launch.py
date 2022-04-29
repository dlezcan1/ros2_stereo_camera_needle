from matplotlib import container
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent
from launch.events import Shutdown
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare



def generate_launch_description():
    package_name = "pgr_stereo_camera"

    pkg_share = FindPackageShare( package_name )

    ld = LaunchDescription()

    # arguments
    arg_ns = DeclareLaunchArgument('ns',
                                    default_value="camera",
                                    description="The ROS namespace to launch the camera node(s) in."
    )

    arg_rosparam_l = DeclareLaunchArgument('leftParamFile',
                                         default_value='left_camera.yaml',
                                         description="The ROS Yaml parameter file for loading in the share config directory for left camera."
    )

    arg_rosparam_r = DeclareLaunchArgument('rightParamFile',
                                         default_value='right_camera.yaml',
                                         description="The ROS Yaml parameter file for loading in the share config directory for right camera."
    )

    arg_rosparam_s = DeclareLaunchArgument('stereoParamFile',
                                         default_value='stereo_camera.yaml',
                                         description="The ROS Yaml parameter file for loading in the share config directory for stereo camera setup."
    )
    
    arg_sync = DeclareLaunchArgument('syncStereo',
                                     default_value="false",
                                     choices=['true', 'false'],
                                     description="Whether to run stereo camera's synchronized or not."
    )

    arg_imageproc = DeclareLaunchArgument('useImageProc',
                                          default_value='true',
                                          choices=['true', 'false'],
                                          description="Whether to use image processing or not."
    )

    # nodes
    node_stereo_sync = Node(
            package=package_name,
            namespace=LaunchConfiguration('ns'),
            executable='pgr_stereo_camera',
            output='screen',
            emulate_tty=True,
            condition=LaunchConfigurationEquals( 'syncStereo', 'true' ),
            parameters=[
                PathJoinSubstitution( [ pkg_share, 'config', LaunchConfiguration('stereoParamFile') ] ),
            ]
    )

    node_mono_left = Node(
        package=package_name,
        namespace=PathJoinSubstitution( [ LaunchConfiguration('ns'), 'left' ] ),
        name="LeftMonocularCameraNode",
        executable='pgr_mono_camera',
        output='screen',
        emulate_tty=True,
        condition=LaunchConfigurationEquals( 'syncStereo', 'false' ),
        parameters=[
            PathJoinSubstitution( [ pkg_share, 'config', LaunchConfiguration('leftParamFile') ] ),
        ],
    )

    node_mono_right = Node(
        package=package_name,
        namespace=PathJoinSubstitution( [LaunchConfiguration('ns'), 'right'] ),
        name="RightMonocularCameraNode",
        executable='pgr_mono_camera',
        output='screen',
        emulate_tty=True,
        condition=LaunchConfigurationEquals( 'syncStereo', 'false' ),
        parameters=[
            PathJoinSubstitution( [ pkg_share, 'config', LaunchConfiguration('rightParamFile') ] ),
        ],
    )

    container_img_proc_l = ComposableNodeContainer(
        name="left_image_proc_container",
        package='rclcpp_components',
        executable='component_container',
        namespace=PathJoinSubstitution([LaunchConfiguration('ns'), 'left']),
        condition=LaunchConfigurationEquals('useImageProc', 'true'),
        composable_node_descriptions=[
            ComposableNode(
                package='image_proc',
                plugin='image_proc::DebayerNode',
                name='left_debayer_node',
                namespace=PathJoinSubstitution([LaunchConfiguration('ns'), 'left']),
            ),
            # Example of rectifying an image
            ComposableNode(
                package='image_proc',
                plugin='image_proc::RectifyNode',
                name='left_rectify_mono_node',
                namespace=PathJoinSubstitution([LaunchConfiguration('ns'), 'left']),
                # Remap subscribers and publishers
                remappings=[
                    # Subscriber remap
                    ('image', 'image_mono'),
                    ('camera_info', 'camera_info'),
                    ('image_rect', 'image_rect')
                ],
            ),
            # Example of rectifying an image
            ComposableNode(
                package='image_proc',
                plugin='image_proc::RectifyNode',
                name='left_rectify_color_node',
                namespace=PathJoinSubstitution([LaunchConfiguration('ns'), 'left']),
                # Remap subscribers and publishers
                remappings=[
                    # Subscriber remap
                    ('image', 'image_color'),
                    # Publisher remap
                    ('image_rect', 'image_rect_color')
                ],
            )],
        output='screen'
    )
    
    container_img_proc_r = ComposableNodeContainer(
        name="right_image_proc_container",
        package='rclcpp_components',
        executable='component_container',
        namespace=PathJoinSubstitution([LaunchConfiguration('ns'), 'right']),
        condition=LaunchConfigurationEquals('useImageProc', 'true'),
        composable_node_descriptions=[
            ComposableNode(
                package='image_proc',
                plugin='image_proc::DebayerNode',
                name='right_debayer_node',
                namespace=PathJoinSubstitution([LaunchConfiguration('ns'), 'right']),
            ),
            # Example of rectifying an image
            ComposableNode(
                package='image_proc',
                plugin='image_proc::RectifyNode',
                name='right_rectify_mono_node',
                namespace=PathJoinSubstitution([LaunchConfiguration('ns'), 'right']),
                # Remap subscribers and publishers
                remappings=[
                    # Subscriber remap
                    ('image', 'image_mono'),
                    ('camera_info', 'camera_info'),
                    ('image_rect', 'image_rect')
                ],
            ),
            # Example of rectifying an image
            ComposableNode(
                package='image_proc',
                plugin='image_proc::RectifyNode',
                name='right_rectify_color_node',
                namespace=PathJoinSubstitution([LaunchConfiguration('ns'), 'right']),
                # Remap subscribers and publishers
                remappings=[
                    # Subscriber remap
                    ('image', 'image_color'),
                    # Publisher remap
                    ('image_rect', 'image_rect_color')
                ],
            )],
        output='screen'
    )
    

    # configure launch description
    # - arguments
    ld.add_action( arg_ns )

    ld.add_action( arg_rosparam_l )
    ld.add_action( arg_rosparam_r )
    ld.add_action( arg_rosparam_s )
    
    ld.add_action( arg_sync )
    ld.add_action( arg_imageproc )
    
    # - nodes
    ld.add_action( node_stereo_sync )
    ld.add_action( node_mono_left )
    ld.add_action( node_mono_right )

    # - containers
    ld.add_action( container_img_proc_l )
    ld.add_action( container_img_proc_r )

    return ld

# generate_launch_description