# standard libraries

# ROS libraries
from cv2 import PARAM_ALGORITHM
from rclpy.node import Node

# 3rd party libraries
import numpy as np
import cv2 as cv

# messages
from std_msgs.msg import Bool, Header
from sensor_msgs.msg import Image

# services

# custom imports
from .PgrCamera import PgrCamera
from .PgrStereo import PgrStereo
from .utilities import ImageConversions

# fields
CAMERA_TOPIC_NAMES = {
    # monocular vision
    "image"          : "image/raw",
    "connected"      : "connected",
    
    # stereo vision
    'left_connected' : "left/connected",
    'right_connected': "right/connected",

    'left_image'     : "left/image/raw",
    'right_image'    : "right/image/raw", 
}

CAMERA_TIMER_TIMES = {
    'connected': 0.5,
    'image'    : 1/60,
}

# nodes/classes
class CameraNode( Node ):
    
    PARAM_CAMERA = "camera"
    PARAM_CAMERA_IDX = f"{PARAM_CAMERA}.index"

    def __init__( self, name="CameraNode" ):
        super().__init__( name )
        
        self.camera = PgrCamera()
        
        # ROS parameters
        self.camera_index = self.declare_parameter( self.PARAM_CAMERA_IDX, 0 ).get_parameter_value().integer_value

        # connect the camera
        if self.connect_camera():
            self.get_logger().info( f"Connected to camera at index: {self.camera_index}" )
            self.get_logger().info( self.camera.get_camera_info() )

        # if
        else:
            self.get_logger().warn( f"Unable to connect to camera at index: {self.camera_index}" )

        # else

        # start camera capturing
        if self.camera.start_capture():
            self.get_logger().info( f"Camera has begun capturing images." )

        else:
            self.get_logger().warn( f"Camera was unable to start capturing." )

        # ROS publishers
        self.pub_connected = self.create_publisher( Bool,  CAMERA_TOPIC_NAMES['connected'], 10 )
        self.pub_image     = self.create_publisher( Image, CAMERA_TOPIC_NAMES['image'],     10 )

        # ROS timers
        self.timer_pub_connected = self.create_timer( CAMERA_TIMER_TIMES["connected"], self.publish_cameraConnected )
        self.timer_pub_image     = self.create_timer( CAMERA_TIMER_TIMES["image"],     self.publish_image )

    # __init__

    def connect_camera( self ):
        """ Connect the camera """
        
        return self.camera.connect( self.camera_index )

    # connect_camera

    def publish_cameraConnected(self):
        """ Publish if the camera is connected or not """
        self.pub_connected.publish( Bool( data=self.camera.isConnected ) )

    # publish_cameraConnected

    def publish_image(self):
        """ Publish the camera's image """
        img = self.camera.retrieve_image()

        # convert to message
        header = Header( stamp=self.get_clock().now().to_msg(), frame_id="camera" )
        msg = ImageConversions.cvToImageMsg( img.astype(np.uint8), encoding="calculate", header=header )

        # publish the message
        if msg is not None:
            self.pub_image.publish( msg )

        # if

    # publish_image

# class: CameraNode

class StereoCameraNode(Node):
    """ Single instance of stereo camera node"""
    
    def __init__( self, name="StereoCameraNode" ):
        super().__init__( name )

        self.stereo_camera = PgrStereo()

        # connect the cameras
        if self.connect_cameras():
            self.get_logger().info( "Connected to stereo camera pair" )
            self.get_logger().info( "PGR Stereo Cameras: ")
            self.get_logger().info( self.stereo_camera.get_camera_info() )

        # if
        else:
            self.get_logger().warn( "Unable to connect to stereo camera pair." )

        # start capturing images from the cameras
        if self.stereo_camera.isConnected:
            self.stereo_camera.startCapture()
            self.get_logger().info( "Started capture on stereo cameras." )

        # if

        # ROS parameters
        
        # ROS subscribers
        
        # ROS publishers
        self.pub_connected_l = self.create_publisher(Bool, CAMERA_TOPIC_NAMES['left_connected'],  10)
        self.pub_connected_r = self.create_publisher(Bool, CAMERA_TOPIC_NAMES['right_connected'], 10)

        self.pub_image_l = self.create_publisher( Image, CAMERA_TOPIC_NAMES['left_image'],  10 )
        self.pub_image_r = self.create_publisher( Image, CAMERA_TOPIC_NAMES['right_image'], 10 )

        # ROS services

        # ROS timers
        self.timer_pub_connected = self.create_timer( CAMERA_TIMER_TIMES["connected"], self.publish_camerasConnected )
        self.timer_pub_image     = self.create_timer( CAMERA_TIMER_TIMES["image"],     self.publish_images )
        

    # __init__

    def connect_cameras(self):
        """ Connect to the cameras """
        self.stereo_camera.connect( 0, 1 )

        return self.stereo_camera.isConnected

    # connect_cameras

    def publish_camerasConnected(self):
        self.pub_connected_l.publish( Bool( data=self.stereo_camera.cam_left.isConnected ) )
        self.pub_connected_r.publish( Bool( data=self.stereo_camera.cam_right.isConnected ) )

    # publish_cameraConnected

    def publish_images(self):
        # get the image pair
        if not self.stereo_camera.isConnected:
            return

        # return

        img_l, img_r = self.stereo_camera.grab_image_pair()

        # convert to image messages
        header = Header( stamp=self.get_clock().now().to_msg(), frame_id='camera' )
        
        msg_l = ImageConversions.cvToImageMsg( img_l.astype(np.uint8), encoding='calculate', header=header )
        msg_r = ImageConversions.cvToImageMsg( img_r.astype(np.uint8), encoding='calculate', header=header )

        # publish the messages
        if msg_l is not None:
            self.pub_image_l.publish( msg_l )

        if msg_r is not None:
            self.pub_image_r.publish( msg_r )

    # publish_images


# class: StereoCameraNode