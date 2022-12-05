import PySpin
import numpy as np
import cv2



class PgrStereo():
	def __init__(self, cam_left_idx: int = 0, cam_right_idx: int = 1):
		''' Constructor '''
		self.system = PySpin.System.GetInstance()
		self.capture_started = False # capture has not been started


		try:
			# Retrieve list of cameras from the system
			#
			# *** NOTES ***
			# Camera lists can be retrieved from an interface or the system object.
			# Camera lists retrieved from the system, such as this one, return all
			# cameras available on the system.
			#
			self.cam_list = self.system.GetCameras()

			self.num_cams = self.cam_list.GetSize()

			print('STEREO: Number of cameras detected: %i' % self.num_cams)

			

			# Finish if there are not enough cameras
			if self.num_cams < 2:

				# Clear camera list before releasing system
				self.cam_list.Clear()

				# Release system instance
				self.system.ReleaseInstance()

				print('Not enough cameras!')
				return
		
			# Print device vendor and model name
			self.cam_left = self.cam_list[cam_left_idx]
			self.cam_right = self.cam_list[cam_right_idx]
			

			#Initializing the cameras
			# connect camera if applicable
			if (cam_left_idx >= 0) and (cam_right_idx >= 0):
				self.connect( cam_left_idx, cam_right_idx)

			# if



			# # Retrieve TL device nodemap; please see NodeMapInfo example for
			# # additional comments on transport layer nodemaps
			# nodemap_tldevice = self.camera.GetTLDeviceNodeMap()

			# # Print device vendor name and device model name
			# #
			# # *** NOTES ***
			# # Grabbing node information requires first retrieving the node and
			# # then retrieving its information. There are two things to keep in
			# # mind. First, a node is distinguished by type, which is related
			# # to its value's data type. Second, nodes should be checked for
			# # availability and readability/writability prior to making an
			# # attempt to read from or write to the node.
			# node_device_vendor_name = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceVendorName'))

			# if PySpin.IsAvailable(node_device_vendor_name) and PySpin.IsReadable(node_device_vendor_name):
			#   device_vendor_name = node_device_vendor_name.ToString()

			# node_device_model_name = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceModelName'))

			# if PySpin.IsAvailable(node_device_model_name) and PySpin.IsReadable(node_device_model_name):
			#   device_model_name = node_device_model_name.ToString()

			# print('\tDevice %s %s \n' % ( device_vendor_name, device_model_name))
			





			
			




			# # Retrieve TL device nodemap and print device information
			# nodemap_tldevice = self.camera.GetTLDeviceNodeMap()

			# self.print_device_info(nodemap_tldevice)

			

			# Start capture
			# self.start_capture()


			# Acquire images
			# self.grab_image_pair()


			# End capture
			# self.stop_capture()


			
		except PySpin.SpinnakerException as ex:
			print('Error: %s' % ex)
		


	# __init__

	def __del__(self):
		''' Destructor '''
		# disconnect the cameras
		try:
			self.stopCapture()
			
		except Exception as e:
			pass
			
		self.disconnect()

	# __del__

	@property
	def isConnected(self):
		return self.cam_left.IsInitialized() and self.cam_right.IsInitialized()

	# property: isConnected

	def connect(self, cam_left_idx: int = 0, cam_right_idx: int = 1):
		""" Connect to the camera """
		print("Connecting...")

		if cam_left_idx == cam_right_idx:
			raise IndexError("Need 2 different camera indices for stereo")
		
		if (self.num_cams < cam_left_idx + 1) or (cam_left_idx < 0):
			raise IndexError("Left camera index is out of range")

		if (self.num_cams < cam_right_idx + 1) or (cam_right_idx < 0):
			raise IndexError("Right camera index is out of range")


		# if

		self.cam_left.Init()
		self.cam_right.Init()

		print("STEREO: Is connected to the cameras:", self.isConnected)

		return (self.cam_left.IsInitialized(), self.cam_right.IsInitialized())

	# connect

	def disconnect(self):
		""" Disconnect the camera """
		print("STEREO: Disconnecting...")
		self.cam_left.DeInit()
		self.cam_right.DeInit()

		print("STEREO: Is connected to the cameras:", self.isConnected)

		del self.cam_left, self.cam_right

		print("Check1")
		self.cam_list.Clear()
		print("Check2")
		self.system.ReleaseInstance()
		print("Check3")

	# disconnect


	def get_camera_info(self):
		left_nodemap_tldevice = self.cam_left.GetTLDeviceNodeMap()
		left_node_device_information = PySpin.CCategoryPtr(left_nodemap_tldevice.GetNode('DeviceInformation'))
		
		right_nodemap_tldevice = self.cam_right.GetTLDeviceNodeMap()
		right_node_device_information = PySpin.CCategoryPtr(right_nodemap_tldevice.GetNode('DeviceInformation'))

		msg = ""

		for name, cam_info in zip(['Left', 'Right'], [left_node_device_information, right_node_device_information]):
			info_str = f'***[{name}] CAMERA INFORMATION ***\n'
			if PySpin.IsAvailable(cam_info) and PySpin.IsReadable(cam_info):
				features = cam_info.GetFeatures()
				for feature in features:
					node_feature = PySpin.CValuePtr(feature)
					info_str+=('%s: %s \n' % (node_feature.GetName(),
									node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

			else:
				info_str.append('Device control information not available.\n')
			
			msg += info_str

		return msg
	
	# get_camera_info

	def print_camera_info(self):
		"""
		This function prints the device information of the camera from the transport
		layer; please see NodeMapInfo example for more in-depth comments on printing
		device information from the nodemap.

		:param nodemap: Transport layer device nodemap.
		:type nodemap: INodeMap
		:returns: True if successful, False otherwise.
		:rtype: bool
		"""
		print( "PgrStereo Camera:" )
		print( self.get_camera_info() )
		# print('*** DEVICE INFORMATION ***\n')

		# try:
		#   result = True
		#   node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

		#   if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
		#       features = node_device_information.GetFeatures()
		#       for feature in features:
		#           node_feature = PySpin.CValuePtr(feature)
		#           print('%s: %s' % (node_feature.GetName(),
		#                             node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

		#   else:
		#       print('Device control information not available.')

		# except PySpin.SpinnakerException as ex:
		#   print('Error: %s' % ex)
		#   return False

		# return result

	def grab_image_pair(self):
		''' Grabs an image pair from each camera. Returns None if there is an error '''
		image_left_np, image_right_np = None, None
		try:
			# Create ImageProcessor instance for post processing images
			processor = PySpin.ImageProcessor()

			# Set default image processor color processing method
			#
			# *** NOTES ***
			# By default, if no specific color processing algorithm is set, the image
			# processor will default to NEAREST_NEIGHBOR method.
			processor.SetColorProcessing(PySpin.HQ_LINEAR)

			try: 
				img_left = self.cam_left.GetNextImage(1000)

				if img_left.IsIncomplete():
					print('Image incomplete with image status %d ...' % img_left.GetImageStatus())

				else:
					image_left_np = img_left.GetData().reshape(img_left.GetHeight(),img_left.GetWidth() ,-1)
					img_left.Release()
			except PySpin.SpinnakerException as ex:
				print('Error: %s' % ex)
				img_left_np = None

			

			try:
				img_right = self.cam_right.GetNextImage(1000)
				if img_right.IsIncomplete():
					print('Image incomplete with image status %d ...' % img_right.GetImageStatus())

				else:
					image_right_np = img_right.GetData().reshape(img_right.GetHeight(),img_right.GetWidth() ,-1)
					img_right.Release()
			except PySpin.SpinnakerException as ex:
				print('Error: %s' % ex)
				img_right_np = None

		except PySpin.SpinnakerException as ex:
			print('Error: %s' % ex)
			img_left_np, img_right_np = None, None
		
		return (image_left_np, image_right_np)

	# retrieve_image

	def startCapture(self):
		""" Begin camera capture """
		try:
			# Retrieve GenICam nodemap
			left_nodemap = self.cam_left.GetNodeMap()

			# Set acquisition mode to continuous
			# In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
			left_node_acquisition_mode = PySpin.CEnumerationPtr(left_nodemap.GetNode('AcquisitionMode'))
			if not PySpin.IsAvailable(left_node_acquisition_mode) or not PySpin.IsWritable(left_node_acquisition_mode):
				print('Unable to set acquisition mode to continuous for left camera (enum retrieval). Aborting...')
			# Retrieve entry node
			left_node_acquisition_mode_continuous = left_node_acquisition_mode.GetEntryByName('Continuous')
			if not PySpin.IsAvailable(left_node_acquisition_mode_continuous) or not PySpin.IsReadable(left_node_acquisition_mode_continuous):
				print('Unable to set acquisition mode to continuous for left camera (entry retrieval). Aborting...')

			# Retrieve integer value from entry node
			left_acquisition_mode_continuous = left_node_acquisition_mode_continuous.GetValue()

			# Set integer value from entry node as new value of enumeration node
			left_node_acquisition_mode.SetIntValue(left_acquisition_mode_continuous)
			print('[LEFT] Camera  acquisition mode set to continuous...')


			#  Begin acquiring images
			self.cam_left.BeginAcquisition()
			print("[LEFT] Began Acquisition")



			right_nodemap = self.cam_right.GetNodeMap()
			right_node_acquisition_mode = PySpin.CEnumerationPtr(right_nodemap.GetNode('AcquisitionMode'))
			if not PySpin.IsAvailable(right_node_acquisition_mode) or not PySpin.IsWritable(right_node_acquisition_mode):
				print('Unable to set acquisition mode to continuous for right camera (enum retrieval). Aborting...')
			# Retrieve entry node
			right_node_acquisition_mode_continuous = right_node_acquisition_mode.GetEntryByName('Continuous')
			if not PySpin.IsAvailable(right_node_acquisition_mode_continuous) or not PySpin.IsReadable(right_node_acquisition_mode_continuous):
				print('Unable to set acquisition mode to continuous for right camera (entry retrieval). Aborting...')

			right_acquisition_mode_continuous = right_node_acquisition_mode_continuous.GetValue()

			right_node_acquisition_mode.SetIntValue(right_acquisition_mode_continuous)
			print('[RIGHT] Camera  acquisition mode set to continuous...')

			self.cam_right.BeginAcquisition()
			print("[RIGHT] Began Acquisition")

		except PySpin.SpinnakerException as ex:
			print('Error in start_capture: %s' % ex)


	# start_capture

	def stop_capture(self):
		""" Stop camera capture """
		try: 
			self.cam_left.EndAcquisition()
			print('[LEFT]: Capture stopped')

			self.cam_right.EndAcquisition()
			print('[RIGHT]: Capture stopped')

		except PySpin.SpinnakerException as ex:
			print('Error: %s' % ex)

	# stop_capture

# class: PgrStereo


# camera=PgrStereo(cam_left_idx=0, cam_right_idx=1)


def live_capture():
    pgr_stereo = PgrStereo()
    
    pgr_stereo.connect(0,1) # where to change the stereo camera order

    pgr_stereo.startCapture()

    counter = 0
    file_base = "{}-{:04d}.png"    
    while True:
        try:
            img_left, img_right = pgr_stereo.grab_image_pair()

            img_cat = np.concatenate((img_left, img_right), axis=1)
            cv2.putText(img_cat, 'LEFT', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,0], 4)
            cv2.putText(img_cat, 'RIGHT', (img_left.shape[1]+20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,0], 4)

            # cv2.imshow('left', img_left[:,:,::-1])
            # cv2.imshow('right', img_right[:,:,::-1])
            cv2.imshow('left-right', img_cat[:,:,::-1])
          

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

            # if
            
            elif key & 0xFF == ord('c'):
                cv2.imwrite(file_base.format("left", counter), cv2.cvtColor(img_left, cv2.COLOR_RGB2BGR))
                print("Captured image: ", file_base.format('left', counter))
                
                cv2.imwrite(file_base.format("right", counter), cv2.cvtColor(img_right, cv2.COLOR_RGB2BGR))
                print("Captured image: ", file_base.format('right', counter))

                counter += 1
            # elif
            

        except:
            break

    # while
    
    cv2.destroyAllWindows()
    pgr_stereo.stopCapture()
    pgr_stereo.disconnect()


# live_capture


#========================== MAIN =================================

if __name__ == "__main__":
    # test_PgrStereo()
    # prompted_capture()
    # stereo_viewer()
	live_capture()
# if: main
