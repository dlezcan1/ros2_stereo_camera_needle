# 3rd party libraries
import numpy as np
from cv_bridge import CvBridge

# ROS messages
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, Image


class ImageConversions:
    bridge = CvBridge() 

    @classmethod
    def cvToImageMsg(cls, image_np: np.ndarray, encoding: str=None, header: Header=Header()):
        """ Convert numpy image to an sensor_msgs/msg/Image message. Returns none if image_np is None

            :param image_np: numpy array of the image
            :param encoding: string value of the type of encoding. If "calculate" is reported, will return:
                                    - 'rgb8' if image has 3 channels
                                    - 'mono8' if image has 1 channel
                            If None, just use default bridge conversion
                            (Default=None)
            :param Header: std_msgs.msg.Header for the image message (Default=Header())

            :returns: None if image_np is None
                      sensor_msgs.msg.Image message of image_np image
        
        """
        if image_np is None:
            return None
        
        msg = cls.bridge.cv2_to_imgmsg( image_np )
        msg.header = header
        if encoding is not None and encoding == "calculate": # determine the encoding
            msg.encoding = "rgb8" if image_np.ndim == 3 else "mono8"

        elif encoding is not None:
            msg.encoding = encoding

        return msg

    # cvToImageMsg

    @staticmethod
    def numpyToMsgImpl(image_np: np.ndarray, header: Header=Header()):
        """ Self-implementation of converting numpy image to Image message (untested)

            encoding is calculated as:
                'rgb8' if image has 3 color channels
                'mono8' if image has 1 color channel

            :param image_np: numpy array of the image
            :param Header: std_msgs.msg.Header for the image message (Default=Header())

            :returns: None if image_np is None
                      sensor_msgs.msg.Image message of image_np image
        """
        if image_np is None:
            return None
        
        msg = Image()

        if image_np.ndim not in [2,3]:
            raise ValueError( f"Numpy image is not a 2D or 3D array. Is of size {image_np.size()}" )

        # if

        # set the header
        msg.header = header

        # set the image
        msg.height, msg.width, *_ = image_np.shape
        msg.encoding = "rgb8" if image_np.ndim == 3 else "mono8"
        msg.step = msg.width * image_np.dtype.itemsize * ( 3 if image_np.ndim == 3 else 1) # number of bytes in a row
        msg.is_bigendian = False

        msg.data = image_np.ravel().tolist()

        return msg

    # numpyToMsgImpl

    @classmethod
    def ImageMsgTocv( cls, image_msg: Image ):
        return cls.bridge.imgmsg_to_cv2( image_msg )

    # ImageMsgTocv

# class: ImageConversions

def duplicate_CameraInfomsg( msg_old: CameraInfo ):
    msg_new = CameraInfo()

    msg_new.header = msg_old.header

    msg_new.height = msg_old.height
    msg_new.width = msg_old.width

    msg_new.k = msg_old.k
    msg_new.r = msg_old.r
    msg_new.p = msg_old.p

    msg_new.distortion_model = msg_old.distortion_model
    msg_new.d                = msg_old.d

    # - optional operational parameters
    msg_new.roi = msg_old.roi
    
    msg_new.binning_x = msg_old.binning_x
    msg_new.binning_y = msg_old.binning_y

    return msg_new

# duplicate_CameraInfo_msg