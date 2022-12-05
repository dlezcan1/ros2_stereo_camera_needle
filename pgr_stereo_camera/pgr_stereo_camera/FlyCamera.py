import PySpin
import numpy as np
import cv2 as cv



class PgrCamera():
	def __init__(self, cam_idx: int=-1):
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

			print('Number of cameras detected: %i' % self.num_cams)

			

			# Finish if there are no cameras
			if self.num_cams == 0:

				# Clear camera list before releasing system
				self.cam_list.Clear()

				# Release system instance
				self.system.ReleaseInstance()

				print('Not enough cameras!')
				return
		
			# Print device vendor and model name
			self.camera = self.cam_list[cam_idx]
			



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
			# 	device_vendor_name = node_device_vendor_name.ToString()

			# node_device_model_name = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceModelName'))

			# if PySpin.IsAvailable(node_device_model_name) and PySpin.IsReadable(node_device_model_name):
			# 	device_model_name = node_device_model_name.ToString()

			# print('\tDevice %s %s \n' % ( device_vendor_name, device_model_name))
			





			#Initializing the camera
			# connect camera if applicable
			if cam_idx >= 0:
				self.connect( cam_idx)

			# if
			




			# Retrieve TL device nodemap and print device information
			# nodemap_tldevice = self.camera.GetTLDeviceNodeMap()

			# self.print_device_info(nodemap_tldevice)

			

			#Start capture
			# self.start_capture()


			#Acquire images
			# self.retrieve_image()


			#End capture
			# self.stop_capture()



			# self.disconnect()

			
		except PySpin.SpinnakerException as ex:
			print('Error: %s' % ex)
		


	# __init__

	@property
	def isConnected(self):
		return self.camera.IsInitialized()

	# property: isConnected

	def connect(self, cam_idx: int = -1):
		print("Connecting...")
		""" Connect to the camera """
		if (cam_idx < 0) or (len(self.cam_list) < cam_idx + 1):
			raise IndexError("Camera index is out of range!") # do nothing

		# if

		self.camera.Init()
		print("Is connected to the camera:", self.isConnected)

		return self.isConnected

	# connect

	def disconnect(self):
		""" Disconnect the camera """
		print("Disconnecting...")
		self.camera.DeInit()
		result = self.isConnected
		print("Is connected to the camera:", result)

		del self.camera
		print("Check1")
		self.cam_list.Clear()
		print("Check2")
		self.system.ReleaseInstance()
		print("Check3")

		return result

	# disconnect

	def print_device_info(self, nodemap):
		"""
		This function prints the device information of the camera from the transport
		layer; please see NodeMapInfo example for more in-depth comments on printing
		device information from the nodemap.

		:param nodemap: Transport layer device nodemap.
		:type nodemap: INodeMap
		:returns: True if successful, False otherwise.
		:rtype: bool
		"""

		print('*** DEVICE INFORMATION ***\n')

		try:
			result = True
			node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

			if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
				features = node_device_information.GetFeatures()
				for feature in features:
					node_feature = PySpin.CValuePtr(feature)
					print('%s: %s' % (node_feature.GetName(),
									  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

			else:
				print('Device control information not available.')

		except PySpin.SpinnakerException as ex:
			print('Error: %s' % ex)
			return False

		return result


	def get_camera_info(self):
		nodemap_tldevice = self.camera.GetTLDeviceNodeMap()
		node_device_information = PySpin.CCategoryPtr(nodemap_tldevice.GetNode('DeviceInformation'))
		
		info_str = '*** CAMERA INFORMATION ***\n'
		if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
			features = node_device_information.GetFeatures()
			for feature in features:
				node_feature = PySpin.CValuePtr(feature)
				info_str += ('%s: %s \n' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

		else:
			print('Device control information not available.')
		
		msg = (info_str)

		# msg = (f'*** CAMERA INFORMATION ***\n'
		#        f'Serial number - {cam_info.serialNumber:d}\n'
		#        f'Camera model - {cam_info.modelName.decode():s}\n'
		#        f'Camera vendor - {cam_info.vendorName.decode():s}\n'
		#        f'Sensor - {cam_info.sensorInfo.decode():s}\n'
		#        f'Resolution - {cam_info.sensorResolution.decode():s}\n'
		#        f'Firmware version - {cam_info.firmwareVersion.decode():s}\n'
		#        f'Firmware build time - {cam_info.firmwareBuildTime.decode():s}'
		#     )

		return msg
	
	# get_camera_info

	def retrieve_image(self):
		""" Retrieves an image from the camera. Returns None if there is an error """
		# make sure we have started the capture
		if not self.capture_started:
			return None

		# if

		image_np=None
		try:
			# Create ImageProcessor instance for post processing images
			processor = PySpin.ImageProcessor()

			# Set default image processor color processing method
			#
			# *** NOTES ***
			# By default, if no specific color processing algorithm is set, the image
			# processor will default to NEAREST_NEIGHBOR method.
			processor.SetColorProcessing(PySpin.HQ_LINEAR)

			image = self.camera.GetNextImage(1000)

			if image.IsIncomplete():
				print('Image incomplete with image status %d ...' % image.GetImageStatus())

			else:
				image_np = image.GetData().reshape(image.GetHeight(),image.GetWidth() ,-1)
				#  Release image
				#
				#  *** NOTES ***
				#  Images retrieved directly from the camera (i.e. non-converted
				#  images) need to be released in order to keep from filling the
				#  buffer.
				image.Release()

		except PySpin.SpinnakerException as ex:
			print('Error: %s' % ex)
			return None
		
		return image_np

	# retrieve_image

	def start_capture(self):
		""" Begin camera capture """
		try:
			# Retrieve GenICam nodemap
			nodemap = self.camera.GetNodeMap()
			
			# Set acquisition mode to continuous
			# In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
			node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
			if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
				print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
				return False

			# Retrieve entry node from enumeration node
			node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
			if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(node_acquisition_mode_continuous):
				print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
				return False

			# Retrieve integer value from entry node
			acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

			# Set integer value from entry node as new value of enumeration node
			node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

			#  Begin acquiring images
			#
			#  *** NOTES ***
			#  What happens when the camera begins acquiring images depends on the
			#  acquisition mode. Single frame captures only a single image, multi
			#  frame catures a set number of images, and continuous captures a
			#  continuous stream of images. Because the example calls for the
			#  retrieval of 10 images, continuous mode has been set.
			#
			#  *** LATER ***
			#  Image acquisition must be ended when no more images are needed.
			self.camera.BeginAcquisition()
			self.capture_started = True
		except PySpin.SpinnakerException as ex:
			print('Error: %s' % ex)
			return False

		return self.capture_started

	# start_capture

	def stop_capture(self):
		""" Stop camera capture """
		try: 
			self.camera.EndAcquisition()
			self.capture_started = False
		except PySpin.SpinnakerException as ex:
			print('Error: %s' % ex)
			return False

		return not self.capture_started

	# stop_capture

# class: PgrCamera


# camera=PgrCamera(cam_idx=0)
