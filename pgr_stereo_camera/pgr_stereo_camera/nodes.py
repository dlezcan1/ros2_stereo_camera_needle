# standard libraries
from abc import ABC

# ROS libraries
from rclpy.node import Node

# 3rd party libraries
import numpy as np
import cv2 as cv

# messages
from std_msgs.msg import Bool, Header
from sensor_msgs.msg import CameraInfo, Image

# services

# custom imports
from .FlyCamera import PgrCamera
from .FlyStereo import PgrStereo
from .utilities import ImageConversions, duplicate_CameraInfomsg

# constants
CAMERA_DEFAULTS = {
    'height': 1080, #768,
    'width' : 1440, #1024,
}

CAMERA_TOPIC_NAMES = {
    # monocular vision
    "info"           : "camera_info",
    "image"          : "image_raw",
    "connected"      : "connected",
}

CAMERA_TIMER_TIMES = {
    'connected': 0.5,
    'image'    : 1/60,
}

# nodes/classes
class CameraNode( Node ):
    PARAM_CAMERA              = "camera"
    PARAM_CAMERA_IDX          = f"{PARAM_CAMERA}.index"
    
    PARAM_CAMERA_INFO         = f"{PARAM_CAMERA}.info"
    PARAM_CAMERA_INFO_STEREO  = f"{PARAM_CAMERA_INFO}.stereo"

    PARAM_CAMERA_INFO_INT     = f"{PARAM_CAMERA_INFO}.intrinsics"         # 9 element (3x3 row-major) float
    
    PARAM_CAMERA_INFO_DST     = f"{PARAM_CAMERA_INFO}.distortion"         # number of parameters depend on distortion model
    PARAM_CAMERA_INFO_DST_MDL = f"{PARAM_CAMERA_INFO_DST}_model"          # string of distortion model
      
    PARAM_CAMERA_INFO_PROJCT  = f"{PARAM_CAMERA_INFO}.projection"         # 12 element (3x4 row-major) float
    PARAM_CAMERA_INFO_HEIGHT  = f"{PARAM_CAMERA_INFO}.image_height"       # uint32
    PARAM_CAMERA_INFO_WIDTH   = f"{PARAM_CAMERA_INFO}.image_width"        # uint32

    PARAM_CAMERA_INFO_ROT     = f"{PARAM_CAMERA_INFO_STEREO}.rotation"    # 9 element (3x3 row-major) float
    PARAM_CAMERA_INFO_TRANS   = f"{PARAM_CAMERA_INFO_STEREO}.translation" # 3 element float

    
    def __init__( self, name="CameraNode" ):
        super().__init__( name )
        
        self.camera = PgrCamera()
        
        # ROS parameters
        self.camera_index = self.declare_parameter( self.PARAM_CAMERA_IDX, 0 ).get_parameter_value().integer_value

        self.get_logger().info(f"Current camera index: {self.camera_index}")
        
        # - camera info paramters
        self.msg_camerainfo = CameraInfo()

        pv = self.declare_parameter( self.PARAM_CAMERA_INFO_INT ).get_parameter_value()
        intrinsic_mtx = np.array( pv.double_array_value ).reshape(3,3)
        rotation_mtx  = np.array( self.declare_parameter( self.PARAM_CAMERA_INFO_ROT,   np.eye(3).ravel().tolist() ).get_parameter_value().double_array_value) .reshape(3,3)
        translation   = np.array( self.declare_parameter( self.PARAM_CAMERA_INFO_TRANS, np.zeros(3).tolist() ).get_parameter_value().double_array_value )
        
        projection_mtx = np.hstack( ( intrinsic_mtx, translation.reshape(-1,1) ) ) # defaulted as per interface message
        projection_mtx = np.array( self.declare_parameter( self.PARAM_CAMERA_INFO_PROJCT, projection_mtx.ravel().tolist() ).get_parameter_value().double_array_value ).reshape(3,4)

        distortion_coeffs = self.declare_parameter( self.PARAM_CAMERA_INFO_DST , [0] * 5 ).get_parameter_value().double_array_value 
        distortion_model  = self.declare_parameter( self.PARAM_CAMERA_INFO_DST_MDL, "plumb_bob" ).get_parameter_value().string_value

        height = self.declare_parameter( self.PARAM_CAMERA_INFO_HEIGHT, CAMERA_DEFAULTS['height'] ).get_parameter_value().integer_value
        width  = self.declare_parameter( self.PARAM_CAMERA_INFO_WIDTH,  CAMERA_DEFAULTS['width']  ).get_parameter_value().integer_value
        
        # -- set camera info message
        self.msg_camerainfo.height = height
        self.msg_camerainfo.width  = width

        self.msg_camerainfo.k = intrinsic_mtx.ravel().tolist() 
        self.msg_camerainfo.r = rotation_mtx.ravel().tolist()  
        self.msg_camerainfo.p = projection_mtx.ravel().tolist()
        
        self.msg_camerainfo.distortion_model = distortion_model
        self.msg_camerainfo.d                = list( distortion_coeffs )
            
        # connect the camera
        if self.connect_camera():
            self.get_logger().info( f"Connected to camera at index: {self.camera_index}" )
            for msg in self.camera.get_camera_info().split('\n'):
                self.get_logger().info( msg )

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
        self.pub_caminfo   = self.create_publisher( CameraInfo, CAMERA_TOPIC_NAMES['info'], 10 )

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
        header   = Header( stamp=self.get_clock().now().to_msg(), frame_id="camera" )
        msg_img  = ImageConversions.cvToImageMsg( img.astype(np.uint8), encoding="calculate", header=header )
        msg_info = duplicate_CameraInfomsg( self.msg_camerainfo )
        msg_info.header = header

        # publish the message(s)
        if msg_img is not None:
            self.pub_image.publish( msg_img )
            self.pub_caminfo.publish( msg_info )

        # if

    # publish_image

# class: MonoCameraNode

class StereoCameraNode( Node ):
    """ Single instance of stereo camera node"""
    PARAM_CAMERA       = "camera.{}"
    PARAM_CAMERA_INFO  = f"{PARAM_CAMERA}.index"
    
    PARAM_CAMERA_INFO         = f"{PARAM_CAMERA}.info"
    PARAM_CAMERA_INFO_STEREO  = f"{PARAM_CAMERA_INFO}.stereo"

    PARAM_CAMERA_INFO_INT     = f"{PARAM_CAMERA_INFO}.intrinsics"         # 9 element (3x3 row-major) float
    
    PARAM_CAMERA_INFO_DST     = f"{PARAM_CAMERA_INFO}.distortion"         # number of parameters depend on distortion model
    PARAM_CAMERA_INFO_DST_MDL = f"{PARAM_CAMERA_INFO_DST}_model"          # string of distortion model
      
    PARAM_CAMERA_INFO_PROJCT  = f"{PARAM_CAMERA_INFO}.projection"         # 12 element (3x4 row-major) float
    PARAM_CAMERA_INFO_HEIGHT  = f"{PARAM_CAMERA_INFO}.image_height"       # uint32
    PARAM_CAMERA_INFO_WIDTH   = f"{PARAM_CAMERA_INFO}.image_width"        # uint32

    PARAM_CAMERA_INFO_ROT     = f"{PARAM_CAMERA_INFO_STEREO}.rotation"    # 9 element (3x3 row-major) float
    PARAM_CAMERA_INFO_TRANS   = f"{PARAM_CAMERA_INFO_STEREO}.translation" # 3 element float
    
    
    def __init__( self, name="StereoCameraNode" ):
        Node.__init__( self, name )

        self.stereo_camera = PgrStereo()

        # ROS parameters
        # - left camera
        self.msg_camerainfo_l = CameraInfo()

        intrinsic_mtx_l = np.array( self.declare_parameter( self.PARAM_CAMERA_INFO_INT.format('left') ).get_parameter_value().double_array_value ).reshape(3,3)
        rotation_mtx_l  = np.array( self.declare_parameter( self.PARAM_CAMERA_INFO_ROT.format('left'),   np.eye(3).ravel().tolist() ).get_parameter_value().double_array_value) .reshape(3,3)
        translation_l   = np.array( self.declare_parameter( self.PARAM_CAMERA_INFO_TRANS.format('left'), np.zeros(3).tolist() ).get_parameter_value().double_array_value )
        
        projection_mtx_l = np.hstack( ( intrinsic_mtx_l, translation_l.reshape(-1,1) ) ) # defaulted as per interface message
        projection_mtx_l = np.array( self.declare_parameter( self.PARAM_CAMERA_INFO_PROJCT.format('left'), projection_mtx_l.ravel().tolist() ).get_parameter_value().double_array_value ).reshape(3,4)

        distortion_coeffs_l = self.declare_parameter( self.PARAM_CAMERA_INFO_DST.format('left') , [0] * 5 ).get_parameter_value().double_array_value 
        distortion_model_l  = self.declare_parameter( self.PARAM_CAMERA_INFO_DST_MDL.format('left'), "plumb_bob" ).get_parameter_value().string_value

        height_l = self.declare_parameter( self.PARAM_CAMERA_INFO_HEIGHT.format('left'), CAMERA_DEFAULTS['height'] ).get_parameter_value().integer_value
        width_l  = self.declare_parameter( self.PARAM_CAMERA_INFO_WIDTH.format('left'),  CAMERA_DEFAULTS['width']  ).get_parameter_value().integer_value

        # -- set camera info message
        self.msg_camerainfo_l.height = height_l
        self.msg_camerainfo_l.width  = width_l

        self.msg_camerainfo_l.k = intrinsic_mtx_l.ravel().tolist() 
        self.msg_camerainfo_l.r = rotation_mtx_l.ravel().tolist()  
        self.msg_camerainfo_l.p = projection_mtx_l.ravel().tolist()

        self.msg_camerainfo_l.distortion_model = distortion_model_l
        self.msg_camerainfo_l.d                = list( distortion_coeffs_l )

        # - right camera
        self.msg_camerainfo_r = CameraInfo()

        intrinsic_mtx_r = np.array( self.declare_parameter( self.PARAM_CAMERA_INFO_INT.format('right') ).get_parameter_value().double_array_value ).reshape(3,3)
        rotation_mtx_r  = np.array( self.declare_parameter( self.PARAM_CAMERA_INFO_ROT.format('right'),   np.eye(3).ravel().tolist() ).get_parameter_value().double_array_value) .reshape(3,3)
        translation_r   = np.array( self.declare_parameter( self.PARAM_CAMERA_INFO_TRANS.format('right'), np.zeros(3).tolist() ).get_parameter_value().double_array_value )
        
        projection_mtx_r = np.hstack( ( intrinsic_mtx_r, translation_r.reshape(-1,1) ) ) # defaulted as per interface message
        projection_mtx_r = np.array( self.declare_parameter( self.PARAM_CAMERA_INFO_PROJCT.format('right'), projection_mtx_r.ravel().tolist() ).get_parameter_value().double_array_value ).reshape(3,4)

        distortion_coeffs_r = self.declare_parameter( self.PARAM_CAMERA_INFO_DST.format('right') , [0] * 5 ).get_parameter_value().double_array_value 
        distortion_model_r  = self.declare_parameter( self.PARAM_CAMERA_INFO_DST_MDL.format('right'), "plumb_bob" ).get_parameter_value().string_value

        height_r = self.declare_parameter( self.PARAM_CAMERA_INFO_HEIGHT.format('right'), CAMERA_DEFAULTS['height'] ).get_parameter_value().integer_value
        width_r  = self.declare_parameter( self.PARAM_CAMERA_INFO_WIDTH.format('right'),  CAMERA_DEFAULTS['width']  ).get_parameter_value().integer_value

        # -- set camera info message
        self.msg_camerainfo_r.height = height_r
        self.msg_camerainfo_r.width  = width_r

        self.msg_camerainfo_r.k = intrinsic_mtx_r.ravel().tolist() 
        self.msg_camerainfo_r.r = rotation_mtx_r.ravel().tolist()  
        self.msg_camerainfo_r.p = projection_mtx_r.ravel().tolist()
        
        self.msg_camerainfo_r.distortion_model = distortion_model_r
        self.msg_camerainfo_r.d                = list( distortion_coeffs_r )

        # connect the cameras
        if self.connect_cameras():
            self.get_logger().info( "Connected to stereo camera pair" )
            self.get_logger().info( "PGR Stereo Cameras: ")
            for msg in self.stereo_camera.get_camera_info().split('\n'):
                self.get_logger().info( msg )

        # if
        else:
            self.get_logger().warn( "Unable to connect to stereo camera pair." )

        # start capturing images from the cameras
        if self.stereo_camera.isConnected:
            self.stereo_camera.startCapture()
            self.get_logger().info( "Started capture on stereo cameras." )

        # if
        
        # ROS subscribers
        
        # ROS publishers/
        self.pub_connected_l = self.create_publisher(Bool, f"left/{CAMERA_TOPIC_NAMES['connected']}",  10)
        self.pub_connected_r = self.create_publisher(Bool, f"right/{CAMERA_TOPIC_NAMES['connected']}", 10)

        self.pub_image_l = self.create_publisher( Image, f"left/{CAMERA_TOPIC_NAMES['image']}",  10 )
        self.pub_image_r = self.create_publisher( Image, f"right/{CAMERA_TOPIC_NAMES['image']}", 10 )

        self.pub_caminfo_l = self.create_publisher( CameraInfo, f"left/{CAMERA_TOPIC_NAMES['info']}",  10)
        self.pub_caminfo_r = self.create_publisher( CameraInfo, f"right/{CAMERA_TOPIC_NAMES['info']}", 10)

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
        self.pub_connected_l.publish( Bool( data=self.stereo_camera.cam_left.IsInitialized() ) )
        self.pub_connected_r.publish( Bool( data=self.stereo_camera.cam_right.IsInitialized()) )

    # publish_cameraConnected

    def publish_images(self):
        # get the image pair
        if not self.stereo_camera.isConnected:
            return

        # return

        img_l, img_r = self.stereo_camera.grab_image_pair()

        # convert to image messages
        header = Header( stamp=self.get_clock().now().to_msg(), frame_id='camera' )
        
        msg_img_l = ImageConversions.cvToImageMsg( img_l.astype(np.uint8), encoding='calculate', header=header )
        msg_img_r = ImageConversions.cvToImageMsg( img_r.astype(np.uint8), encoding='calculate', header=header )

        msg_caminfo_l = duplicate_CameraInfomsg( self.msg_camerainfo_l )
        msg_caminfo_l.header = header

        msg_caminfo_r = duplicate_CameraInfomsg( self.msg_camerainfo_r )
        msg_caminfo_r.header = header

        # publish the messages
        if msg_img_l is not None: # left
            self.pub_image_l.publish( msg_img_l )
            self.pub_caminfo_l.publish( msg_caminfo_l )

        # if

        if msg_img_r is not None: # right 
            self.pub_image_r.publish( msg_img_r )
            self.pub_caminfo_r.publish( msg_caminfo_r )

        # if

    # publish_images


# class: StereoCameraNode