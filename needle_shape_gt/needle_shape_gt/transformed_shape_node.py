# ROS2 packages
import rclpy
from rclpy.node import Node
from rclpy import Parameter
# messages
from std_msgs.msg import Header
from geometry_msgs.msg import PoseArray, PoseStamped, Pose
import message_filters

#numpy
import numpy as np
#scipy interpolateion functions
from scipy import interpolate
import matplotlib.pyplot as plt


#misc
from scipy.spatial.transform import Rotation as Rot
from needle_shape_sensing import geometry


class NeedleTransformShapeNode(Node):
	"""Node for needle shape transfom with FBG-based needle shape using point cloud registration """
	
	def __init__(self, name="NeedleTransformShapeNode"):
		super().__init__(name)



		# declare member properties
		self.insertion_depth = 0.0 # in mm
		self.ds = 0.5 # increments along arclength in mm
		self.gt_shape_inserted = None
		self.fbg_shape_inserted = None
		self.Rmat = None
		self.pmat = None


		# ### TEST
		# self.a=100*np.random.rand(3,10) # 3xN

		# self.a = np.array([[7.5854,   16.2182,   45.0542],
  #  [5.3950   ,79.4285,    8.3821],
  #  [53.0798   ,31.1215,   22.8977],
  #  [77.9167   ,52.8533,   91.3337],
  #  [93.4011   ,16.5649,   15.2378],
  #  [12.9906   ,60.1982,   82.5817],
  #  [56.8824   ,26.2971,   53.8342],
  #  [46.9391   ,65.4079,   99.6135],
  #   [1.1902   ,68.9215,    7.8176],
  #  [33.7123   ,74.8152 ,  44.2678]]).T

		# rotation_matrix = Rot.from_rotvec(np.pi/2 * np.array([0, 0, 1])).as_matrix() # 3x3
		# self.b = rotation_matrix @ self.a # 3x3 * 3xN =  3xN

		# p=np.reshape( np.array([10,-10,0]), (3,1) )
		# print(rotation_matrix)
		# for i in range(self.b.shape[1]):
		# 	self.b[:,i] =  self.b[:,i]+p.T
		# print("**************** A: ", self.a.shape, self.a)
		# print("**************** B: ", self.b.shape, self.b)

		# create subscriptions
		self.sub_needle_gt_shape = self.create_subscription( PoseArray, '/needle/state/gt_shape', self.sub_needle_gt_shape_callback, 10)
		self.sub_needle_fbg_shape = self.create_subscription( PoseArray, '/needle/state/current_shape', self.sub_needle_fbg_shape_callback, 10)
		self.sub_needle_pose = self.create_subscription( PoseStamped, 'stage/state/needle_pose', self.sub_needle_pose_callback, 10)

		# self.sub_needle_gt_shape = message_filters.Subscriber(self, PoseArray, '/needle/state/gt_shape')
		# self.sub_needle_fbg_shape = message_filters.Subscriber(self, PoseArray,'/needle/state/current_shape')

		# ts = message_filters.ApproximateTimeSynchronizer([self.sub_needle_gt_shape, self.sub_needle_fbg_shape], queue_size=10, slop=1)
		# ts.registerCallback(self.ts_callback)

		# create publishers
		self.pub_transform_camera2needle = self.create_publisher(Pose, '/needle/state/transf_camera2needle', 10)
		self.pub_needle_gt_shape_transformed = self.create_publisher(PoseArray, '/needle/state/gt_shape_transformed', 10)

		#create publisher timers
		self.timer_pub_gt_shape_transformed = self.create_timer(0.5, self.publish_gt_shape_transformed)
		self.timer_pub_transform_camera2needle = self.create_timer(0.5, self.publish_transform)


	# __init__

	def destroy_node( self ) -> bool:
		""" Destroy the node override"""
		self.get_logger().info( "Shutting down..." )
		retval = super().destroy_node()
		self.get_logger().info( "Shut down complete." )
		return retval	

	# destroy_node

	def sub_needle_pose_callback(self, msg):
		self.insertion_depth = 1000*msg.pose.position.y 


	#sub_needle_pose_callback

	def ts_callback(self, msg_gt, msg_fbg):
		shape_gt = msg2poses(msg_gt)
		x,y,z = np.flip(shape_gt[:,0]),np.flip(shape_gt[:,1]),np.flip(shape_gt[:,2])
		squared_diff_x, squared_diff_y, squared_diff_z = np.diff([x,y,z]) ** 2
		arclen = np.sqrt(squared_diff_x + squared_diff_y + squared_diff_z)
		arclen_cum = np.concatenate(([0], arclen.cumsum()))

		tck,u = interpolate.splprep([x,y,z], u=arclen_cum, k=3)
		# u_fine = np.linspace(0, arclen_cum[-1], 50)
		n=self.numPts(self.ds)
		u_fine = np.linspace(0, 0+(self.ds*n),n,endpoint=False)
		x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
		# print(20*"#", "gt shape", len(x_fine))
		self.gt_shape_inserted = np.array([x_fine, y_fine, z_fine]).T
		# print("************* gt:\n", self.gt_shape_inserted[:10,:])

		shape_fbg = msg2poses(msg_fbg)
		if self.insertion_depth>10:
			num_pts = self.numPts(self.ds)
			# print(20*"*","fbg shape ", num_pts)
			self.fbg_shape_inserted = np.flip(shape_fbg[-num_pts:, :])
			# print("############## fbg:\n", self.fbg_shape_inserted[:10,:])


	def sub_needle_gt_shape_callback(self, msg):
		# print(["SHAPE_GT:", msg.header.stamp])
		shape_gt = msg2poses(msg)

		x,y,z = np.flip(shape_gt[:,0]),np.flip(shape_gt[:,1]),np.flip(shape_gt[:,2])
		squared_diff_x, squared_diff_y, squared_diff_z = np.diff([x,y,z]) ** 2
		arclen = np.sqrt(squared_diff_x + squared_diff_y + squared_diff_z)
		arclen_cum = np.concatenate(([0], arclen.cumsum()))

		tck,u = interpolate.splprep([x,y,z], u=arclen_cum, k=3)
		# u_fine = np.linspace(0, arclen_cum[-1], 50)
		n=self.numPts(self.ds)
		u_fine = np.linspace(0, 0+(self.ds*n),n,endpoint=False)
		x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
		# print(20*"#", "gt shape", len(x_fine))
		self.gt_shape_inserted = np.array([x_fine, y_fine, z_fine]).T
		# print("************* gt:\n", self.gt_shape_inserted[:10,:])

	# sub_needle_gt_shape_callback

	def sub_needle_fbg_shape_callback(self, msg):
		# print(["FBG_GT:", msg.header.stamp])
		shape_fbg = msg2poses(msg)
		#print(20*"$"+"	fbg_shape", len(shape_fbg))
		
		if self.insertion_depth>10:
			num_pts = self.numPts(self.ds)
			# print(20*"*","fbg shape ", num_pts)
			self.fbg_shape_inserted = np.flip(shape_fbg[-num_pts:, :])
			# print("############## fbg:\n", self.fbg_shape_inserted[:10,:])

	
	# sub_needle_fbg_shape_callback

	def publish_gt_shape_transformed(self):
		
		# #test
		# if (self.Rmat is not None and self.pmat is not None):
		# 	transformed = self.Rmat@self.a
		# 	for i in range(transformed.shape[1]):
		# 		transformed[:,i] = transformed[:,i] +self.pmat.flatten()

		if (self.Rmat is not None and self.pmat is not None):
			transformed = self.Rmat @ (self.gt_shape_inserted).T
			for i in range(transformed.shape[1]):
				transformed[:,i] = transformed[:,i] + self.pmat.flatten()


			


			# print(20*"#","transformed:",transformed[:,:10])
			# print(20*"&", "fbg", self.fbg_shape_inserted[:10,:].T)

			print("TRANSFORMED",transformed.shape) # (3xN)
			print("FBG", self.fbg_shape_inserted.shape) # (Nx3)
			
			
			#plotting
			# fig = plt.figure()
			# ax1 = fig.add_subplot(111, projection='3d')
			# ax1.scatter(self.fbg_shape_inserted[:,0], self.fbg_shape_inserted[:,1], self.fbg_shape_inserted[:,2], s=10, c='b', marker="o", label='fbg')
			# ax1.scatter(transformed[0,:], transformed[1,:], transformed[2,:], s=10 , c='r', marker="o", label='gt_stereo')
			# ax1.axis('equal')
			# plt.legend(loc='upper left')
			# plt.xlabel('x')
			# plt.ylabel('y')
			# # plt.zlabel('z')
			# plt.show()




			header = Header( stamp=self.get_clock().now().to_msg(), frame_id='/robot' )
			msg_gt_transformed_shape = poses2msg(transformed,header=header)
			self.pub_needle_gt_shape_transformed.publish(msg_gt_transformed_shape)

	# publish_gt_shape_transformed

	def publish_transform(self):
		# #test 
		# self.Rmat, self.pmat = cloud_reg((self.a).T,(self.b).T)
		# print(self.Rmat, self.pmat)

		if (self.gt_shape_inserted is not None and self.fbg_shape_inserted is not None):
			# print("###################### shape a:", self.gt_shape_inserted.shape, self.fbg_shape_inserted.shape)
			self.Rmat, self.pmat = cloud_reg(self.gt_shape_inserted, self.fbg_shape_inserted)
			# print("Rmat, pmat: ",self.Rmat, self.pmat)
			if (self.pmat is not None and self.Rmat is not None):
				msg_camera2needle = pose2msg(self.pmat, self.Rmat)
				self.pub_transform_camera2needle.publish(msg_camera2needle)


	# publish_transform

	def numPts(self, ds=0.5):
		return int(self.insertion_depth//ds)

# class: NeedleTransformShapeNode	

def pose2msg( pos: np.ndarray, R: np.ndarray = None ):
    """ Turn a pose into a Pose message """
    msg = Pose()

    # handle position
    msg.position.x = pos[ 0 ]
    msg.position.y = pos[ 1 ]
    msg.position.z = pos[ 2 ]
    if R is not None:
	    # handle orientation
	    quat = geometry.rotm2quat( R )
	    msg.orientation.w = quat[ 0 ]
	    msg.orientation.x = quat[ 1 ]
	    msg.orientation.y = quat[ 2 ]
	    msg.orientation.z = quat[ 3 ]

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

def msg2pose( msg: Pose):
	""" Convert a Pose message into a pose"""
	pos = np.array( [ msg.position.x, msg.position.y, msg.position.z ] )
	# quat = np.array( [ msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z ] )
	# R = geometry.quat2rotm( quat )

	return pos

# msg2pose

def msg2poses( msg: PoseArray):
	""" Convert a PoseArray message into nd.array of poses"""
	size = len(msg.poses)
	poses=np.zeros((size,3))

	for i in range(size):
		poses[i,:] = msg2pose(msg.poses[i])

	return poses

# msg2pose

def cloud_reg(camera_shape_data: np.ndarray, fbg_shape_data :np.ndarray):
	'''
	Function for performing poiont cloud registration between 2 point clouds a and b such that F*a=b.
	Inputs: np.ndarray's of size (Nx3)
	Outputs: R (3x3), p (3x1) np.ndarray's
	'''

	# Get data shapes
	size_a = len(camera_shape_data)
	size_b = len(fbg_shape_data)
	
	# Do initial checks of data size
	if (size_a != size_b):
		# print("Error. Cloud sizes should be equal")
		return (None, None)
	elif (size_a<3):
		print("Error. Cloud registration requires at least 3 point pairs")
		return (None, None)

	# Transfering to zero-mean points
	a_mean = np.reshape(np.mean(camera_shape_data, axis=0), (1,3))
	b_mean = np.reshape(np.mean(fbg_shape_data, axis=0), (1,3))
	a1 = camera_shape_data-a_mean
	b1 = fbg_shape_data-b_mean

	# print("################### a1:", a1)
	# print("################### b1:", b1)

	# Constructing matrix H
	H=np.zeros((3,3))
	Hd=np.zeros((3,3))
	for i in range(size_a):
		Hd[0,0] = np.dot(a1[i,0],b1[i,0])
		Hd[0,1] = np.dot(a1[i,0],b1[i,1])
		Hd[0,2] = np.dot(a1[i,0],b1[i,2])
		Hd[1,0] = np.dot(a1[i,1],b1[i,0])
		Hd[1,1] = np.dot(a1[i,1],b1[i,1])
		Hd[1,2] = np.dot(a1[i,1],b1[i,2])
		Hd[2,0] = np.dot(a1[i,2],b1[i,0])
		Hd[2,1] = np.dot(a1[i,2],b1[i,1])
		Hd[2,2] = np.dot(a1[i,2],b1[i,2])
		H = H+Hd

	# print("######################## H:", H)
	# Applying SVD on H
	U,_,Vh = np.linalg.svd(H)

	V=Vh.T
	# print("######################## U:", U)
	# print("######################## V:",V)

	Rmat = V@U.T

	# print("######################## Rmat:", Rmat)

	# If det(R) = -1, get V' matrix which is V with 3rd column vector negated and set R = V' * U.T 
	if (abs(np.linalg.det(Rmat)+1.0) < 0.01):
		V[:,2] = -V[:,2] 
		Rmat = V@U.T

	# Calculate translational part p
	# pmat = b_mean.T-Rmat@(a_mean.T)
	pmat = b_mean.flatten() - Rmat@a_mean.flatten()

	return Rmat,pmat

# cloud_reg

def main( args=None ):
	rclpy.init( args=args )

	transformed_needle_node = NeedleTransformShapeNode()

	try:
		rclpy.spin( transformed_needle_node )

	except KeyboardInterrupt:
		pass

	# clean up
	transformed_needle_node.destroy_node()
	rclpy.shutdown()


# main


if __name__ == "__main__":
	main()

# if: main