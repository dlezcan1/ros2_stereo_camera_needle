# ROS2 packages
import rclpy
from rclpy.node import Node
from rclpy import Parameter
# messages
from std_msgs.msg import Bool, Header
from geometry_msgs.msg import PoseArray, PoseStamped, Pose
from sensor_msgs.msg import CameraInfo, Image
from rcl_interfaces.msg import ParameterDescriptor
# ROS2 services
from std_srvs.srv import Trigger

#numpy
import numpy as np
# cv
import cv2
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError


import matplotlib.pyplot as plt

#needle_reconstruction
from .needle_reconstruction import StereoRefInsertionExperiment

class NeedleStereoReconstructionNode(Node):
	"""Node for needle 3D shape reconstruction using stereo vision image pairs"""
	
	# PARAMETER NAMES
	PARAM_NEEDLE = "needle"
	
	# - needle stereo shape reconstruction parameters
	PARAM_STEREORECONSTRUCTIONFILE = ".".join( [ PARAM_NEEDLE, "stereoReconstructionParamFile" ] )  # needle parameter json file
	# PARAM_STEREOPARAMFILE = ".".join( [ PARAM_NEEDLE, "stereoParamFile" ] )  # needle parameter json file
	# PARAM_LEFTROI = ".".join( [ PARAM_NEEDLE, "leftROI" ] )  # region of interest array for left camera
	# PARAM_RIGHTROI = ".".join( [ PARAM_NEEDLE, "rightROI" ] )  # region of interest array for right camera
	# PARAM_LEFTCONTRASTENHANCE = ".".join( [ PARAM_NEEDLE, "leftContrastEnhance" ] )  # constrast enhance parameter for left camera
	# PARAM_RIGHTCONTRASTENHANCE = ".".join( [ PARAM_NEEDLE, "rightContrastEnhance" ] )  # constrast enhance parameter for right camera
	# PARAM_ZOOM = ".".join( [ PARAM_NEEDLE, "zoom" ] )  # template matching zoom
	# PARAM_WINDOWSIZE = ".".join( [ PARAM_NEEDLE, "windowSize" ] )  # template matching window size
	# PARAM_ALPHA = ".".join( [ PARAM_NEEDLE, "alpha" ] )  # stereo rectification alpha parameter
	# PARAM_SUBTRACTTHRESHOLD = ".".join( [ PARAM_NEEDLE, "subtractThreshold" ] )  # reference image subtraction threshold

	def __init__(self, name="NeedleStereoReconstruction"):
		super().__init__(name)

		# declare member properties for camera image acquisition
		self.is_left_camera_connected = False
		self.is_right_camera_connected = False
		self.ref_left_img = None
		self.ref_right_img = None
		self.left_camera_img = None
		self.right_camera_img = None
		self.insertion_depth = 0.0
		self.bridge = CvBridge()

		# declare and get parameters
		pd_stereoneedleparams = ParameterDescriptor( name=self.PARAM_STEREORECONSTRUCTIONFILE, type=Parameter.Type.STRING.value,
											  description='stereo needle shape parameters json file.', read_only=True )
		stereo_reconstruction_param_file = self.declare_parameter( pd_stereoneedleparams.name, descriptor=pd_stereoneedleparams ). \
			get_parameter_value().string_value


		# get Stereo Needle Shape Reconstruction object
		try:
			self.needle_reconstructor = StereoRefInsertionExperiment.load_json( stereo_reconstruction_param_file )

			self.get_logger().info( "Successfully loaded Stereo Needle Shape Reconstructor: \n" + str( self.needle_reconstructor ) )

		# try
		except Exception as e:
			self.get_logger().error( f"{stereo_reconstruction_param_file} is not a valid stereo reconstruction parameter file." )
			raise e

		# except



		# # declare stereo parameters
		# pd_stereoParamFile = ParameterDescriptor( name=self.PARAM_STEREOPARAMFILE, type=Parameter.Type.STRING.value,
		# 									  description='stereo parameters .mat file.', read_only=True )
		# # declare image region of interest parameters
		# pd_leftROI = ParameterDescriptor( name=self.PARAM_LEFTROI, type=Parameter.Type.INTEGER_ARRAY.value,
		# 									  description='The left image ROI to use of format [TOP_Y, TOP_X, BOTTOM_Y, BOTTOM_X', read_only=True )
		# pd_rightROI = ParameterDescriptor( name=self.PARAM_RIGHTROI, type=Parameter.Type.INTEGER_ARRAY.value,
		# 									  description='The right image ROI to use of format [TOP_Y, TOP_X, BOTTOM_Y, BOTTOM_X', read_only=True )
		# pd_leftContrastEnhance = ParameterDescriptor( name=self.PARAM_LEFTCONTRASTENHANCE, type=Parameter.Type.DOUBLE_ARRAY.value,
		# 									  description='The left image contrast enhancement of format [ALPHA, BETA]', read_only=True )
		# pd_rightContrastEnhance = ParameterDescriptor( name=self.PARAM_RIGHTCONTRASTENHANCE, type=Parameter.Type.DOUBLE_ARRAY.value,
		# 									  description='The right image contrast enhancement of format [ALPHA, BETA]', read_only=True )
		# # declare reconstruction parameters
		# pd_zoom = ParameterDescriptor( name=self.PARAM_ZOOM, type=Parameter.Type.DOUBLE.value,
		# 									  description='The zoom for stereo template matching', read_only=True )
		# pd_windowSize = ParameterDescriptor( name=self.PARAM_WINDOWSIZE, type=Parameter.Type.INTEGER_ARRAY.value,
		# 									  description='The window size for stereo template matching of format [WIDTH, DEPTH]', read_only=True )
		# pd_alpha = ParameterDescriptor( name=self.PARAM_ALPHA, type=Parameter.Type.DOUBLE.value,
		# 									  description='The alpha parameter for stereo rectification', read_only=True )
		# pd_subtractThreshold = ParameterDescriptor( name=self.PARAM_SUBTRACTTHRESHOLD, type=Parameter.Type.DOUBLE.value,
		# 									  description='The threshold for reference image subtraction', read_only=True )

		# # declaration of parameters
		# stereo_params_file = self.declare_parameter( pd_stereoParamFile.name,
		# 										descriptor=pd_stereoParamFile ).get_parameter_value().string_value
		# left_roi = self.declare_parameter( pd_leftROI.name,
		# 										descriptor=pd_leftROI ).get_parameter_value().integer_array_value
		# right_roi = self.declare_parameter( pd_rightROI.name,
		# 										descriptor=pd_rightROI ).get_parameter_value().integer_array_value
		# left_contrast_enhance = self.declare_parameter( pd_leftContrastEnhance.name,
		# 										descriptor=pd_leftContrastEnhance ).get_parameter_value().double_array_value
		# right_contrast_enhance = self.declare_parameter( pd_rightContrastEnhance.name,
		# 										descriptor=pd_rightContrastEnhance ).get_parameter_value().double_array_value
		# zoom = self.declare_parameter( pd_zoom.name,
		# 										descriptor=pd_zoom ).get_parameter_value().double_value
		# windows_size = self.declare_parameter( pd_windowSize.name,
		# 										descriptor=pd_windowSize ).get_parameter_value().integer_array_value
		# alpha = self.declare_parameter( pd_alpha.name,
		# 										descriptor=pd_alpha ).get_parameter_value().double_value
		# subtractThreshold = self.declare_parameter( pd_subtractThreshold.name,
		# 										descriptor=pd_subtractThreshold ).get_parameter_value().double_value


		# create subscriptions
		self.sub_connected_l = self.create_subscription( Bool, 'camera/left/connected', lambda msg: self.sub_connected_callback(msg, "left"), 1)
		self.sub_connected_r = self.create_subscription( Bool, 'camera/right/connected', lambda msg: self.sub_connected_callback(msg, "right"), 1)
		self.sub_image_l = self.create_subscription( Image, 'camera/left/image_raw', lambda msg: self.sub_image_callback(msg, "left"),10)
		self.sub_image_r = self.create_subscription( Image, 'camera/right/image_raw', lambda msg: self.sub_image_callback(msg, "right"), 10)
		self.sub_needle_pose = self.create_subscription( PoseStamped, 'stage/state/needle_pose', self.sub_needle_pose_callback, 10)

		# create publishers
		self.pub_needle_gt_shape = self.create_publisher(PoseArray, '/needle/state/gt_shape', 10)

		#create publisher timers
		self.timer_pub_gt_shape = self.create_timer(1, self.publish_gt_shape)

		# create services
		self.srv_ref_img = self.create_service(Trigger, 'camera/get_reference', self.srv_ref_img_callback)

	# __init__

	def destroy_node( self ) -> bool:
		""" Destroy the node override"""
		self.get_logger().info( "Shutting down..." )
		retval = super().destroy_node()
		self.get_logger().info( "Shut down complete." )
		return retval

	# destroy_node


	def sub_connected_callback(self, msg, cameraName):
		if (cameraName == "left"):
			if msg.data==True:
				self.is_left_camera_connected = True
			else:
				self.is_left_camera_connected = False
		else:
			if msg.data==True:
				self.is_right_camera_connected = True
			else:
				self.is_right_camera_connected = False

	# sub_connected_callback

	def sub_needle_pose_callback(self, msg):
		self.insertion_depth = msg.pose.position.y # in meters


	#sub_needle_pose_callback

	def sub_image_callback(self, msg, cameraName):
		if (cameraName == "left"):
			self.left_camera_img = np.array(self.bridge.imgmsg_to_cv2(msg, "bgr8"))
		else:
			self.right_camera_img = np.array(self.bridge.imgmsg_to_cv2(msg, "bgr8"))

	#sub_image_callback

	def publish_gt_shape(self):

		if  (self.insertion_depth > 0.02) and  (self.ref_left_img is not None) and (self.ref_right_img is not None):
			# image_pair_ref = (self.ref_left_img, self.ref_right_img)
			# image_pair_target = (self.left_camera_img, self.right_camera_img)
			# dataset = {"ref": image_pair_ref, "target": image_pair_target}
			
			# depth=self.insertion_depth%0.005
			# print(depth)
			# if depth<0.0001:
				# cv2.imwrite('left_'+str(self.insertion_depth)+'.png', self.left_camera_img)
				# cv2.imwrite('right_'+str(self.insertion_depth)+'.png', self.right_camera_img)
			# cv2.imwrite('check_left.png', self.left_camera_img)
			# cv2.imwrite('check_right.png', self.right_camera_img)
			self.needle_reconstructor.needle_reconstructor.load_image_pair(self.left_camera_img, self.right_camera_img, reference=False)


			try:
				pts_3d = self.needle_reconstructor.needle_reconstructor.reconstruct_needle()
			except:
				print("[NEEDLE_SHAPE_GT] EXCEPTION: NEEDLE WAS NOT DETECTED ON THE IMAGE")
				return
			

			#plot
			# fig = plt.figure()
			# ax1 = fig.add_subplot(111, projection='3d')
			# ax1.scatter(pts_3d[:,0], pts_3d[:,1], pts_3d[:,2], s=10, c='g', marker="o", label='shape_gt_untransformed')
			# ax1.axis('equal')
			# plt.legend(loc='upper left')
			# plt.xlabel('x')
			# plt.ylabel('y')
			# # plt.zlabel('z')
			# plt.show()


			header = Header( stamp=self.get_clock().now().to_msg(), frame_id='/robot' )
			msg_gt_shape = poses2msg(pts_3d,header=header)

			self.pub_needle_gt_shape.publish(msg_gt_shape)

	# publish_gt_shape

	def srv_ref_img_callback(self, request: Trigger.Request, response: Trigger.Response):
		""" Service to get reference image for needle shape reconstruction """

		if self.is_left_camera_connected and self.is_right_camera_connected:
			self.ref_left_img = self.left_camera_img
			self.ref_right_img = self.right_camera_img
			cv2.imwrite('ref_left.png', self.ref_left_img)
			cv2.imwrite('ref_right.png', self.ref_right_img)
			# print(self.ref_left_img.shape)
			self.needle_reconstructor.needle_reconstructor.load_image_pair(self.ref_left_img, self.ref_right_img, reference=True)
			response.success = True
		# if
		else:
			response.success = False
			response.message = "Cameras are not connected. Please check camera connections."

		# else

		return response

	# srv_ref_img_callback

# class: NeedleStereoReconstructionNode	

def pose2msg( pos: np.ndarray):
    """ Turn a pose into a Pose message """
    msg = Pose()

    # handle position
    msg.position.x = pos[ 0 ]
    msg.position.y = pos[ 1 ]
    msg.position.z = pos[ 2 ]

    return msg


# pose2msg

def poses2msg( pmat: np.ndarray, header: Header = Header() ):
    """ Turn a sequence of poses into a PoseArray message"""
    # determine number of elements in poses
    N = len(pmat)
    
    # generate the message and add the individual poses
    msg = PoseArray( header=header )
    for i in range( N ):
        msg.poses.append( pose2msg( pmat[ i ] ) )

    # for

    return msg

# poses2msg





def main( args=None ):
	rclpy.init( args=args )

	stereo_needle_node = NeedleStereoReconstructionNode()

	try:
		rclpy.spin( stereo_needle_node )

	except KeyboardInterrupt:
		pass

	# clean up
	stereo_needle_node.destroy_node()
	rclpy.shutdown()


# main


if __name__ == "__main__":
	main()

# if: main