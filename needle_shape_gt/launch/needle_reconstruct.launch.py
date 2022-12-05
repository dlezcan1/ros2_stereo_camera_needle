import sys, os
import json
from ament_index_python.packages import get_package_share_directory


from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node



pkg_needle_shape_gt = get_package_share_directory('needle_shape_gt')

def generate_launch_description():
    ld = LaunchDescription()

    default_needle_reconstruction_param_file = "reconstruction_params.json"
    needleStereoParamFile = default_needle_reconstruction_param_file
    for arg in sys.argv:
        if arg.startswith("stereoNeedleParamFile:="):
            needleStereoParamFile = arg.split(":=")[1]

            break
        # if        
    # for

    # arguments
    arg_params = DeclareLaunchArgument( 'needleStereoParamFile',
                                        default_value=default_needle_reconstruction_param_file,
                                        description="The stereo needle reconstruction parameter json file." )

    node_needle_stereo_gt = Node(
            package='needle_shape_gt',
            namespace='',
            executable='stereo_needle_shape_node',
            output='screen',
            emulate_tty=True,
            parameters=[ {
                    'needle.stereoReconstructionParamFile' : os.path.join(pkg_needle_shape_gt, "config", default_needle_reconstruction_param_file)
                    # 'needle.stereoReconstructionParamFile' : LaunchConfiguration( 'needleStereoParamFile')
                    } ]
            )
    node_needle_transformed_gt = Node(
            package='needle_shape_gt',
            namespace='',
            executable='transformed_shape_node',
            output='screen',
            emulate_tty=True,
            parameters=[]
            )

   	# configure launch description
    ld.add_action(arg_params)
    ld.add_action(node_needle_stereo_gt)
    ld.add_action(node_needle_transformed_gt)

    return ld

# generate_launch_descrtiption