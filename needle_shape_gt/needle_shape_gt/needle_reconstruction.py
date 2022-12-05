"""
Created on Aug 26, 2021

This is a library/script to perform stereo needle reconstruciton and process datasets


@author: Dimitri Lezcano

"""
import argparse
import json
import glob
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Any, Union

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from . import stereo_needle_proc as stereo_needle


@dataclass
class ImageROI:
    image: np.ndarray
    roi: List[ int ] = field( default_factory=list )
    blackout: List[ int ] = field( default_factory=list )


# dataclass: ImageROI

@dataclass
class StereoPair:
    left: Any = None
    right: Any = None

    def __eq__( self, other ):
        if isinstance( other, StereoPair ):
            retval = (self.left == other.left, self.right == other.right)

        elif isinstance( other, (tuple, list) ):
            retval = (self.left == other[ 0 ], self.right == other[ 1 ])

        else:
            retval = (self.left == other, self.right == other)

        return retval

    # __eq__

    @staticmethod
    def from_image( left: str = None, right: str = None ):
        """ Generate a StereoPair object from to image files"""
        sp = StereoPair()
        sp.load_image_pair( left=left, right=right )

        return sp

    # from_image

    def load_image_pair( self, left: str = None, right: str = None ):
        """ Load image pair of image files"""
        if left is not None:
            self.left = cv.imread( left, cv.IMREAD_COLOR )

        if right is not None:
            self.right = cv.imread( right, cv.IMREAD_COLOR )

    # load_image_pair

    def set( self, left, right ):
        """ Function to set the stereo pair"""
        self.left = left
        self.right = right

    # set_pair


# dataclass: StereoPair

@dataclass
class StereoImagePair:
    left: ImageROI
    right: ImageROI


# dataclass: StereoImagePair

class StereoRefInsertionExperiment:
    directory_pattern = r".*[/,\\]Insertion([0-9]+)[/,\\]([0-9]+).*"  # data directory pattern

    def __init__( self, stereo_param_file: str, insertion_depths: list = None,
                  insertion_numbers: list = None, roi: tuple = None, blackout: tuple = None, contrast: tuple = None,
                  window_size: np.ndarray = None,  alpha: float = None, zoom: float = None, sub_thresh: float = None):
        stereo_params = stereo_needle.load_stereoparams_matlab( stereo_param_file )
        
        # self.data_directory = os.path.normpath( data_dir )
        # self.insertion_numbers = insertion_numbers
        
        # if insertion_depths is None:
        #     self.insertion_depths = None

        # else:  # make sure non-negative insertion depth
        #     self.insertion_depths = [ 0 ] + list( filter( lambda d: d > 0, insertion_depths ) )

        self.needle_reconstructor = StereoNeedleRefReconstruction( stereo_params, None, None, None, None )

        # set the datasets ROI
        if roi is not None:
            self.needle_reconstructor.roi.left = roi[ 0 ]
            self.needle_reconstructor.roi.right = roi[ 1 ]

        # if

        # set the image blackout regions
        if blackout is not None:
            self.needle_reconstructor.blackout.left = blackout[ 0 ]
            self.needle_reconstructor.blackout.right = blackout[ 1 ]

        # if

        # set the image contrast enhancements
        if contrast is not None:
            self.needle_reconstructor.contrast.left = contrast[ 0 ]
            self.needle_reconstructor.contrast.right = contrast[ 1 ]

        # if

        if window_size is not None:
            self.needle_reconstructor.window_size = window_size

        # if

        if alpha is not None:
            self.needle_reconstructor.alpha = alpha

        # if

        if zoom is not None:
            self.needle_reconstructor.zoom = zoom
            
        # if

        if sub_thresh is not None:
            self.needle_reconstructor.sub_thresh = sub_thresh
            
        # if



        # configure the dataset
        # self.dataset, self.processed_data = self.configure_dataset( self.data_directory, self.insertion_depths,
        #                                                             self.insertion_numbers )

    # __init__

    @property
    def processed_images( self ):
        return self.needle_reconstructor.processed_images

    # processed_images

    @property
    def processed_figures( self ):
        return self.needle_reconstructor.processed_figures

    # processed_figures

    @property
    def stereo_params( self ):
        return self.needle_reconstructor.stereo_params

    # stereo_params

    @classmethod
    def configure_dataset( cls, directory: str, insertion_depths: list, insertion_numbers: list ) -> (list, list):
        """
            Configure a dataset based on the directory:

            :param directory: string of the main data directory
            :param insertion_depths: a list of insertion depths that are to be processed.
            :param insertion_numbers: a list of insertion numbers to process

        """
        dataset = [ ]
        processed_dataset = [ ]

        if directory is None:
            return dataset

        # if

        directories = glob.glob( os.path.join( directory, 'Insertion*/*/' ) )

        # iterate over the potential directories
        for d in directories:
            res = re.search( cls.directory_pattern, d )

            if res is not None:
                insertion_num, insertion_depth = res.groups()
                insertion_num = int( insertion_num )
                insertion_depth = float( insertion_depth )

                # only include insertion depths that we want to process
                if (insertion_depths is None) and (insertion_numbers is None):  # take all data
                    dataset.append( (d, insertion_num, insertion_depth) )

                elif insertion_depths is None:  # take all depths
                    if insertion_num in insertion_numbers:
                        dataset.append( (d, insertion_num, insertion_depth) )

                elif insertion_numbers is None:
                    if insertion_depth in insertion_depths:  # take all insertion trials
                        dataset.append( (d, insertion_num, insertion_depth) )

                elif insertion_depth in insertion_depths and insertion_num in insertion_numbers:  # be selective
                    dataset.append( (d, insertion_num, insertion_depth) )

                if os.path.isfile( os.path.join( d, 'left-right_3d-pts.csv' ) ):
                    # load the processed data
                    pts_3d = np.loadtxt( os.path.join( d, 'left-right_3d-pts.csv' ), delimiter=',' )
                    processed_dataset.append( (d, insertion_num, insertion_depth, pts_3d) )  # processed_dataset

                # if

            # if
        # for

        return dataset, processed_dataset

    # configure_dataset

    # def process_dataset( self, dataset: list, save: bool = True, overwrite: bool = False):
    #     """ Process the dataset

    #         :param dataset:   List of the data (data_dir, insertion_hole, insertion_depth
    #         :param save:      (Default = True) whether to save the processed data or not
    #         :param overwrite: (Default = False) whether to overwrite already processed datasets
    #         :param kwargs:    the keyword arguments to pass to StereoNeedleRefReconstruction.reconstruct_needle
    #     """
    #     # iterate over the insertion holes
    #     # insertion_holes = set( [ data[ 1 ] for data in dataset ] )
    #     # proc_show = kwargs.get( 'proc_show', False )

    #     for insertion_hole in insertion_holes:
    #         # # grab all of the relevant insertion holes
    #         # dataset_hole = filter( lambda data: (data[ 1 ] == insertion_hole) and (data[ 2 ] > 0), dataset )
    #         # dataset_hole_ref = \
    #         #     list( filter( lambda data: (data[ 1 ] == insertion_hole) and (data[ 2 ] == 0), dataset ) )[ 0 ]

    #         # load the reference images
    #         left_ref_file = os.path.join( dataset_hole_ref[ 0 ], 'left.png' )
    #         right_ref_file = os.path.join( dataset_hole_ref[ 0 ], 'right.png' )
    #         self.needle_reconstructor.load_image_pair( left_ref_file, right_ref_file, reference=True )

    #         # iterate over the datasets
    #         for sub_dataset in dataset_hole:
    #             # # see if the data has been processed already
    #             # d, _, insertion_depth = sub_dataset
    #             # idx = np.argwhere( list(
    #             #         map(
    #             #                 lambda row: all(
    #             #                         (row[ 0 ] == d, row[ 1 ] == insertion_hole, row[ 2 ] == insertion_depth) ),
    #             #                 self.processed_data ) ) ).flatten()

    #             # # check if data is already processed
    #             # if len( idx ) > 0 and not overwrite and save:
    #             #     continue

    #             # # if

    #             print( f"Processing dataset: {d}" )

    #             proc_dataset = self.process_trial()

    #             # # show the processed images
    #             # if proc_show:
    #             #     plt.show()

    #             # # if

    #             # # check if we are a new dataset or overwriting
    #             # if len( idx ) == 0:
    #             #     self.processed_data.append( proc_dataset )

    #             # # if
    #             # elif overwrite:  # overwrite the dataset
    #             #     self.processed_data[ idx[ 0 ] ] = proc_dataset

    #             # # else

    #         # for
    #     # for

    # # process_dataset

    # def process_trial( self):
    #     """ Process a single insertion trial"""
        
    #     # directory, insertion_hole, insertion_depth = dataset  # unpack the dataset

    #     # # load the next image pair
    #     # left_file = os.path.join( directory, 'left.png' )
    #     # right_file = os.path.join( directory, 'right.png' )
    #     # self.needle_reconstructor.load_image_pair( left_file, right_file, reference=False )

    #     # perform the 3D reconstruction
    #     # pts_3d = self.needle_reconstructor.reconstruct_needle()

    #     # # save the data (if you would like to save it)
    #     # if save:
    #     #     self.needle_reconstructor.save_3dpoints( directory=directory )
    #     #     self.needle_reconstructor.save_processed_images( directory=directory )

    #     # # if

    #     return pts_3d

    # # process_trial



    @staticmethod
    def load_json( filename: str ):
        """ 
        This function is used to load a StereoRefInsertionExperiment class from a saved JSON file.
        
        Args:
            - filename: str, the input json file to be loaded.
            
        Returns:
            A StereoRefInsertionExperiment Class object with the loaded json parameters.
        
        """
        # load the data from the json file to a dict
        print(filename)
        with open( filename, 'r' ) as json_file:
            data = json.load( json_file )

        # with

        stereoParamFile = data.get( 'stereo parameters mat file location', None )

        if 'ROI' in data.keys():
            keysROI=data['ROI'].keys()
            if ('left' in keysROI) and ('right' in keysROI):
                leftROI = [data["ROI"]["left"][0:2],data["ROI"]["left"][2:4]]
                rightROI = [data["ROI"]["right"][0:2],data["ROI"]["right"][2:4]]

            # if

        # if
        else:
            leftROI = []
            rightROI = []

        # else

        if 'blackout' in data.keys():
            keysBlackout=data['blackout'].keys()
            if ('left' in keysBlackout) and ('right' in keysBlackout):
                leftBlackout = data["blackout"]["left"]
                rightBlackout = data["blackout"]["right"]

            # if

        # if
        else:
            leftBlackout = []
            rightBlackout = []

        # else

        

        # else

        if 'contrast enhance' in data.keys():
            keysContrast=data['contrast enhance'].keys()
            if ('left' in keysContrast) and ('right' in keysContrast):
                leftContrastEnhance = tuple(data["contrast enhance"]["left"])
                rightContrastEnhance = tuple(data["contrast enhance"]["right"])

            # if
            
        # if
        else:
            leftContrastEnhance = None
            rightContrastEnhance = None

        # else

        if 'window size' in data.keys():
            windowSize = tuple(data['window size'])

        # if

        else:
            windowSize = None

        # else

        zoom = data.get( 'zoom', None )
        alpha = data.get( 'alpha', None )
        subtractThr = data.get( 'subtract threshold', None )

        # instantiate the StereoRefInsertionExperiment class object
        needle_reconstructor = StereoRefInsertionExperiment( stereoParamFile, 
                                                                roi=(leftROI, rightROI), 
                                                                blackout=(leftBlackout,rightBlackout),
                                                                contrast=(leftContrastEnhance, rightContrastEnhance),
                                                                window_size=windowSize,
                                                                zoom=zoom,
                                                                alpha=alpha,
                                                                sub_thresh = subtractThr
                                                            )

        # return the instantiation
        return needle_reconstructor

    # load_json


# class: StereoRefInsertionExperiment


class StereoNeedleReconstruction( ABC ):
    """ Basic class for stereo needle reconstruction"""
    save_fbase = 'left-right_{:s}'

    def __init__( self, stereo_params: dict, img_left: np.ndarray = None, img_right: np.ndarray = None ):
        self.stereo_params = stereo_params
        self.image = StereoPair( img_left, img_right )

        self.roi = StereoPair( [ ], [ ] )
        self.blackout = StereoPair( [ ], [ ] )
        self.contrast = StereoPair( (1, 0), (1, 0) )  # (alpha, beta): alpha * image + beta

        self.needle_shape = None
        self.img_points = StereoPair( None, None )
        self.img_bspline = StereoPair( None, None )
        self.processed_images = { }
        self.processed_figures = { }

    # __init__

    @staticmethod
    def contrast_enhance( image: np.ndarray, alpha: float, beta: float ):
        """ Perform contrast enhancement of an image

            :param image: the input image
            :param alpha: the scaling term for contrast enhancement
            :param beta:  the offset term for contrast enhancement

            :returns: the contrast enhanced image as a float numpy array
        """
        return np.clip( alpha * (image.astype( float )) + beta, 0, 255 )

    # contrast_enhance

    def load_image_pair( self, left_img: np.ndarray = None, right_img: np.ndarray = None ):
        """ Load the image pair. If the one of the images is none, that image will not be loaded

            :param left_img: (Default = None) np.ndarray of the left image
            :param right_img: (Default = None) np.ndarray of the right image

        """

        if left_img is not None:
            self.image.left = left_img

        # if

        if right_img is not None:
            self.image.right = right_img

        # if

    # load_image_pair

    @abstractmethod
    def reconstruct_needle( self, **kwargs ) -> np.ndarray:
        """
            Reconstruct the 3D needle shape from the left and right image pair

        """
        pass

    # reconstruct_needle

    def save_3dpoints( self, outfile: str = None, directory: str = '', verbose: bool = False ):
        """ Save the 3D reconstruction to a file """

        if self.needle_shape is not None:
            if outfile is None:
                outfile = self.save_fbase.format( '3d-pts' ) + '.csv'

            # if

            outfile = os.path.join( directory, outfile )

            np.savetxt( outfile, self.needle_shape, delimiter=',' )
            if verbose:
                print( "Saved reconstructed shape:", outfile )

            # if

        # if

    # save_3dpoints

    def save_processed_images( self, directory: str = '.' ):
        """ Save the images that have now been processed

            :param directory: (Default = '.') string of the directory to save the processed images to.
        """
        # the format string for saving the figures
        save_fbase = os.path.join( directory, self.save_fbase )

        if self.processed_images is not None:
            for key, img in self.processed_images.items():
                cv.imwrite( save_fbase.format( key ) + '.png', img )
                print( "Saved figure:", save_fbase.format( key ) + '.png' )

            # for
        # if

        if self.processed_figures is not None:
            for key, fig in self.processed_figures.items():
                fig.savefig( save_fbase.format( key + '-fig' ) + '.png' )
                print( "Saved figure:", save_fbase.format( key + '-fig' ) + '.png' )

            # for
        # if

    # save_processed_images


# class: StereoNeedleReconstruction


class StereoNeedleRefReconstruction( StereoNeedleReconstruction ):
    """ Class for Needle Image Reference Reconstruction """

    def __init__( self, stereo_params: dict, img_left: np.ndarray = None, img_right: np.ndarray = None,
                  ref_left: np.ndarray = None, ref_right: np.ndarray = None ):
        super().__init__( stereo_params, img_left, img_right )
        self.reference = StereoPair( ref_left, ref_right )

    # __init__

    def load_image_pair( self, left_img: np.ndarray = None, right_img: np.ndarray = None, reference: bool = False ):
        """ Load the image pair. If the one of the images is none, that image will not be loaded

            :param left_img: (Default = None) np.ndarray of the left image
            :param right_img: (Default = None) np.ndarray of the right image
            :param reference: (Default = False) whether we are loading the reference image or not
        """
        if not reference:
            super().load_image_pair( left_img, right_img )

        # if
        else:
            if left_img is not None:
                self.reference.left = left_img

            # if

            if right_img is not None:
                self.reference.right = right_img

            # if

        # else

    # load_image_pair

    def reconstruct_needle( self ) -> np.ndarray:
        """
            Reconstruct the needle shape

            Keyword arguments:
                window size: 2-tuple of for window size of the stereo template matching (must be odd)
                zoom:        the zoom value for for the template maching algorithm
                alpha:       the alpha parameter in stereo rectification
                sub_thresh:  the threshold value for the reference image subtraction

        """
        # keyword argument parsing
        # window_size = kwargs.get( 'window_size', (201, 51) )
        # zoom = kwargs.get( 'zoom', 1.0 )
        # alpha = kwargs.get( 'alpha', 0.6 )
        # sub_thresh = kwargs.get( 'sub_thresh', 60 )
        # proc_show = kwargs.get( 'proc_show', False )

        window_size = self.window_size
        zoom = self.zoom
        alpha = self.alpha
        sub_thresh = self.sub_thresh

        # perform contrast enhancement
        ref_left = self.contrast_enhance( self.reference.left, self.contrast.left[ 0 ],
                                          self.contrast.left[ 0 ] ).astype( np.uint8 )
        img_left = self.contrast_enhance( self.image.left, self.contrast.left[ 0 ],
                                          self.contrast.left[ 1 ] ).astype( np.uint8 )
        ref_right = self.contrast_enhance( self.reference.right, self.contrast.right[ 0 ],
                                           self.contrast.left[ 0 ] ).astype( np.uint8 )
        img_right = self.contrast_enhance( self.image.right, self.contrast.right[ 0 ],
                                           self.contrast.right[ 1 ] ).astype( np.uint8 )

        # perform stereo reconstruction
        pts_3d, pts_l, pts_r, bspline_l, bspline_r, imgs, figs = \
            stereo_needle.needle_reconstruction_ref( img_left, ref_left,
                                                     img_right, ref_right,
                                                     stereo_params=self.stereo_params, recalc_stereo=True,
                                                     bor_l=self.blackout.left, bor_r=self.blackout.right,
                                                     roi_l=self.roi.left, roi_r=self.roi.right,
                                                     alpha=alpha, winsize=window_size, zoom=zoom,
                                                     sub_thresh=sub_thresh)
        # set the current fields
        self.needle_shape = pts_3d[ :, 0:3 ]  # remove 4-th axis

        self.img_points.left = pts_l
        self.img_points.right = pts_r

        self.img_bspline.left = pts_l
        self.img_bspline.right = pts_r

        self.processed_images = imgs
        self.processed_figures = figs

        return pts_3d[ :, 0:3 ]

    # reconstruct_needle


# class:StereoNeedleRefReconstruction

# def __get_parser() -> argparse.ArgumentParser:
#     """ Configure the argument parser"""
#     parser = argparse.ArgumentParser(
#             description='Perform 3D needle reconstruction of the needle insertion experiments.' )

#     # stereo parameters
#     expmt_group = parser.add_argument_group( 'Experiment', 'The experimental parameters' )
#     expmt_group.add_argument( 'stereoParamFile', type=str, help='Stereo Calibration parameter file' )

#     # data directory 
#     expmt_group.add_argument( 'dataDirectory', type=str, help='Needle Insertion Experiment directory' )
#     expmt_group.add_argument( '--insertion-numbers', type=int, nargs='+', default=None )
#     expmt_group.add_argument( '--insertion-depths', type=float, nargs='+', default=None,
#                               help="The insertion depths of the needle to be parsed." )
#     expmt_group.add_argument( '--show-processed', action='store_true', help='Show the processed data' )
#     expmt_group.add_argument( '--save', action='store_true', help='Save the processed data or not' )
#     expmt_group.add_argument( '--force-overwrite', action='store_true', help='Overwrite previously processed data.' )

#     # image region of interestes
#     imgproc_group = parser.add_argument_group( 'Image Processing and Stereo',
#                                                'The image processing and stereo vision parameters.' )
#     imgproc_group.add_argument( '--left-roi', nargs=4, type=int, default=[ ], help='The left image ROI to use',
#                                 metavar=('TOP_Y', 'TOP_X', 'BOTTOM_Y', 'BOTTOM_X') )
#     imgproc_group.add_argument( '--right-roi', nargs=4, type=int, default=[ ], help='The right image ROI to use',
#                                 metavar=('TOP_Y', 'TOP_X', 'BOTTOM_Y', 'BOTTOM_X') )

#     imgproc_group.add_argument( '--left-blackout', nargs='+', type=int, default=[ ],
#                                 help='The blackout regions for the left image' )
#     imgproc_group.add_argument( '--right-blackout', nargs='+', type=int, default=[ ],
#                                 help='The blackout regions for the right image' )

#     imgproc_group.add_argument( '--left-contrast-enhance', nargs=2, type=float, default=[ ],
#                                 help='The left image contrast enhancement', metavar=('ALPHA', 'BETA') )
#     imgproc_group.add_argument( '--right-contrast-enhance', nargs=2, type=float, default=[ ],
#                                 help='The left image contrast enhancement', metavar=('ALPHA', 'BETA') )

#     # reconstruction parameters
#     imgproc_group.add_argument( '--zoom', type=float, default=1.0, help="The zoom for stereo template matching" )
#     imgproc_group.add_argument( '--window-size', type=int, nargs=2, default=(201, 51), metavar=('WIDTH', 'HEIGHT'),
#                                 help='The window size for stereo template matching' )
#     imgproc_group.add_argument( '--alpha', type=float, default=0.6,
#                                 help='The alpha parameter for stereo rectification.' )
#     imgproc_group.add_argument( '--subtract-thresh', type=float, default=60,
#                                 help='The threshold for reference image subtraction.' )

#     # video processing
#     video_group = parser.add_argument_group( 'Video', 'Process needle shape of video images' )
#     video_group.add_argument( '--video', action='store_true', help="Process stereo videos" )

#     # aruco processing
#     aruco_group = parser.add_argument_group( 'ARUCO', 'Process needle shape with ARUCO marker present' )
#     aruco_group.add_argument( '--aruco-id', type=int, default=None, help="The ARUCO ID to detect." )
#     aruco_group.add_argument( '--aruco-size', type=float, default=None,
#                               help="The size of the ARUCO side length (in mm)" )

#     aruco_group.add_argument( '--aruco-thresh', type=int, default=50,
#                               help="The thresholding for ARUCO Image processing." )
#     aruco_group.add_argument( '--aruco-contrast', nargs=2, type=float, default=[ 1, 0 ],
#                               help="Aruco contrast enhancement", metavar=('ALPHA', 'BETA') )

#     aruco_group.add_argument( '--aruco-left-roi', nargs=4, type=int, default=None, help='Left image ARUCO ROI',
#                               metavar=('TOP_Y', 'TOP_X', 'BOTTOM_Y', 'BOTTOM_X') )
#     aruco_group.add_argument( '--aruco-right-roi', nargs=4, type=int, default=None, help='Right image ARUCO ROI',
#                               metavar=('TOP_Y', 'TOP_X', 'BOTTOM_Y', 'BOTTOM_X') )

#     aruco_group.add_argument( '--aruco-left-blackout', nargs=4, type=int, default=None,
#                               help='Left image ARUCO blackout regions',
#                               metavar=('TOP_Y', 'TOP_X', 'BOTTOM_Y', 'BOTTOM_X') )
#     aruco_group.add_argument( '--aruco-right-blackout', nargs=4, type=int, default=None,
#                               help='Right image ARUCO blackout regions',
#                               metavar=('TOP_Y', 'TOP_X', 'BOTTOM_Y', 'BOTTOM_X') )

#     return parser


# # __get_parser

def main( args=None ):
    # # parse the arguments
    # parser = __get_parser()
    # pargs = parser.parse_args( args )

    # # image ROI parsing
    # if len( pargs.left_roi ) > 0:
    #     left_roi = [ pargs.left_roi[ 0:2 ], pargs.left_roi[ 2:4 ] ]

    # # if
    # else:
    #     left_roi = [ ]

    # # else

    # if len( pargs.right_roi ) > 0:
    #     right_roi = [ pargs.right_roi[ 0:2 ], pargs.right_roi[ 2:4 ] ]

    # # if
    # else:
    #     right_roi = [ ]

    # # else

    # # image blackout region parsing
    # if len( pargs.left_blackout ) > 0:
    #     assert (len( pargs.left_blackout ) % 4 == 0)  # check if there are adequate pairs
    #     left_blackout = [ ]
    #     for i in range( 0, len( pargs.left_blackout ), 4 ):
    #         left_blackout.append( [ pargs.left_blackout[ i:i + 2 ],
    #                                 pargs.left_blackout[ i + 2:i + 4 ] ] )

    #     # for

    # # if
    # else:
    #     left_blackout = [ ]

    # # else

    # if len( pargs.right_blackout ) > 0:
    #     assert (len( pargs.right_blackout ) % 4 == 0)  # check if there are adequate pairs
    #     right_blackout = [ ]
    #     for i in range( 0, len( pargs.right_blackout ), 4 ):
    #         right_blackout.append( [ pargs.right_blackout[ i:i + 2 ],
    #                                  pargs.right_blackout[ i + 2:i + 4 ] ] )

    #     # for

    # # if
    # else:
    #     right_blackout = [ ]

    # # else

    # # contrast enhancement
    # left_contrast = tuple( pargs.left_contrast_enhance ) if len( pargs.left_contrast_enhance ) > 0 else None
    # right_contrast = tuple( pargs.right_contrast_enhance ) if len( pargs.right_contrast_enhance ) > 0 else None

    # # instantiate the Insertion Experiment data processor
    # # process the dataset
    # stereo_kwargs = { 'zoom'                : pargs.zoom,
    #                   'window_size'         : pargs.window_size,
    #                   'alpha'               : pargs.alpha,
    #                   'sub_thresh'          : pargs.subtract_thresh,
    #                   'proc_show'           : pargs.show_processed,
    #                   'aruco_left_roi'      : pargs.aruco_left_roi,
    #                   'aruco_left_blackout' : pargs.aruco_left_blackout,
    #                   'aruco_right_roi'     : pargs.aruco_right_roi,
    #                   'aruco_right_blackout': pargs.aruco_right_blackout,
    #                   'aruco_threshold'     : pargs.aruco_thresh,
    #                   'aruco_contrast_alpha': pargs.aruco_contrast[ 0 ],
    #                   'aruco_contrast_beta' : pargs.aruco_contrast[ 1 ]
    #                   }
    # if pargs.video:
    #     image_processor = StereoRefInsertionExperimentVideo( pargs.stereoParamFile, pargs.dataDirectory,
    #                                                          pargs.insertion_depths, pargs.insertion_numbers,
    #                                                          roi=(left_roi, right_roi),
    #                                                          blackout=(left_blackout, right_blackout),
    #                                                          contrast=(left_contrast, right_contrast) )
    #     image_processor.process_video( save=pargs.save, overwrite=pargs.force_overwrite, **stereo_kwargs )
    # # if
    # elif (pargs.aruco_id is not None) and (pargs.aruco_size is not None):
    #     image_processor = StereoRefInsertionExperimentARUCO( pargs.stereoParamFile, pargs.dataDirectory,
    #                                                          pargs.insertion_depths, pargs.insertion_numbers,
    #                                                          aruco_id=pargs.aruco_id, aruco_size=pargs.aruco_size,
    #                                                          roi=(left_roi, right_roi),
    #                                                          blackout=(left_blackout, right_blackout),
    #                                                          contrast=(left_contrast, right_contrast) )
    # # elif
    # else:  # Insertion Images
    #     image_processor = StereoRefInsertionExperiment( pargs.stereoParamFile, pargs.dataDirectory,
    #                                                     pargs.insertion_depths, pargs.insertion_numbers,
    #                                                     roi=(left_roi, right_roi),
    #                                                     blackout=(left_blackout, right_blackout),
    #                                                     contrast=(left_contrast, right_contrast) )
    # # else

    # image_processor.process_dataset( image_processor.dataset, save=pargs.save, overwrite=pargs.force_overwrite,
    #                                  **stereo_kwargs )

    print( "Program completed." )


# main

if __name__ == "__main__":
    main()

# if __main__
