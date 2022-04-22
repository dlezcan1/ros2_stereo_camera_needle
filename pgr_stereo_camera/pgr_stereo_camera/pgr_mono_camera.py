# standard libraries


# ROS standard libraries
import rclpy

# custom imports
from .nodes import CameraNode


def main(args=None):
    rclpy.init(args=args)

    node = CameraNode( "MonocularCameraNode" )

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()

# main


if __name__ == '__main__':
    main()

# if __main__