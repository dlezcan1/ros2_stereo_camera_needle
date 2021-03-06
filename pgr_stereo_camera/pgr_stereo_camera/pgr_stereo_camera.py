# standard libraries


# ROS standard libraries
import rclpy

# custom imports
from .nodes import StereoCameraNode


def main(args=None):
    rclpy.init(args=args)

    node = StereoCameraNode( "StereoCameraNode" )

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()

# main


if __name__ == '__main__':
    main()

# if __main__