import PyCapture2 as pycap
import numpy as np
import cv2 as cv



class PgrCamera():
    def __init__(self, cam_idx: int=-1):
        self.bus = pycap.BusManager()

        self.camera = pycap.Camera()
        self.camera_index = -1       # not connected
        self.capture_started = False # capture has not been started

        # connect camera if applicable
        if cam_idx >= 0:
            self.connect( cam_idx )

        # if

    # __init__

    @property
    def isConnected(self):
        return self.camera.isConnected

    # property: isConnected

    def connect(self, cam_idx: int = -1):
        """ Connect to the camera """
        if (cam_idx < 0) or (self.bus.getNumOfCameras() < cam_idx + 1):
            raise IndexError("Camera index is out of range!") # do nothing

        # if

        self.camera.connect(self.bus.getCameraFromIndex(cam_idx))

        return self.isConnected

    # connect

    def disconnect(self):
        """ Disconnect the camera """
        self.camera.disconnect()

        return not self.isConnected

    # disconnect

    def enable_timestamp(self, enable_timestamp: bool):
        """ Enable timestamping for data gathering """
        embedded_info = self.camera.getEmbeddedImageInfo()

        if embedded_info.available.timestamp:
            self.camera.setEmbeddedImageInfo( timestamp=enable_timestamp )
            successful = True
        
        # if
        else:
            successful = False

        # else

        return successful
    
    # enable_timestamp

    def get_camera_info(self):
        cam_info = self.camera.getCameraInfo()
        msg = (f'*** CAMERA INFORMATION ***\n'
               f'Serial number - {cam_info.serialNumber:d}\n'
               f'Camera model - {cam_info.modelName.decode():s}\n'
               f'Camera vendor - {cam_info.vendorName.decode():s}\n'
               f'Sensor - {cam_info.sensorInfo.decode():s}\n'
               f'Resolution - {cam_info.sensorResolution.decode():s}\n'
               f'Firmware version - {cam_info.firmwareVersion.decode():s}\n'
               f'Firmware build time - {cam_info.firmwareBuildTime.decode():s}'
            )

        return msg
    
    # get_camera_info

    def retrieve_image(self):
        """ Retrieves an image from the camera. Returns None if there is an error """
        # make sure we have started the capture
        if not self.capture_started:
            return None

        # if
        
        try:
            image = self.camera.retrieveBuffer()
            image_np = image.getData().reshape( image.getRows(), image.getCols(), -1 )

        # try
        except pycap.Fc2error as fc2err:
            image_np = None

        # except

        return image_np

    # retrieve_image

    def start_capture(self):
        """ Begin camera capture """
        self.camera.startCapture()
        self.capture_started = True

        return self.capture_started

    # start_capture

    def stop_capture(self):
        """ Stop camera capture """
        self.camera.stopCapture()
        self.capture_started = False

        return not self.capture_started

    # stop_capture

# class: PgrCamera