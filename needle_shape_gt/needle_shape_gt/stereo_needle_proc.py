"""
Created on Nov 6, 2020

This is a file for building image processing to segment the needle in stereo images


@author: dlezcan1

"""

import glob
import os
import re
import time
import warnings
# helper
from typing import Union

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
# NURBS
from geomdl import fitting
from geomdl.visualization import VisMPL
# plotting
from matplotlib import colors as pltcolors
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.neighbors import LocalOutlierFactor

from .BSpline1D import BSpline1D
# custom functions
from .needle_segmentation_functions import find_hsv_image

# color HSV ranges
COLOR_HSVRANGE_RED = ((0, 50, 50), (10, 255, 255))
COLOR_HSVRANGE_BLUE = ((110, 50, 50), (130, 255, 255))
COLOR_HSVRANGE_GREEN = ((40, 50, 50), (75, 255, 255))
COLOR_HSVRANGE_YELLOW = ((25, 50, 50), (35, 255, 255))

# image size
IMAGE_SIZE = (768, 1024)


def axisEqual3D( ax ):
    """ taken from online """
    extents = np.array( [ getattr( ax, 'get_{}lim'.format( dim ) )() for dim in 'xyz' ] )
    sz = extents[ :, 1 ] - extents[ :, 0 ]
    centers = np.mean( extents, axis=1 )
    maxsize = max( abs( sz ) )
    r = maxsize / 2
    for ctr, dim in zip( centers, 'xyz' ):
        getattr( ax, 'set_{}lim'.format( dim ) )( ctr - r, ctr + r )


# axisEqual3D


def blackout( img, tl, br ):
    # handle negative indices
    for i, (tl_i, br_i) in enumerate( zip( tl, br ) ):
        if tl_i < 0:
            tl[ i ] = img.shape[ i ] + tl_i + 1

        if br_i < 0:
            br[ i ] = img.shape[ i ] + br_i + 1

    # for

    img[ tl[ 0 ]:br[ 0 ], tl[ 1 ]:br[ 1 ] ] = 0

    return img


# blackout


def blackout_image( bo_regions: list, image_size ):
    bo_bool = np.ones( image_size[ :2 ], dtype=bool )

    bo_bool = blackout_regions( bo_bool, bo_regions )

    return bo_bool


# blackout_image


def blackout_regions( img, bo_regions: list ):
    img = img.copy()
    for tl, br in bo_regions:
        img = blackout( img, tl, br )

    # for

    return img


# blackout_regions


def bin_close( left_bin, right_bin, ksize=(6, 6) ):
    kernel = np.ones( ksize )
    left_close = cv.morphologyEx( left_bin, cv.MORPH_CLOSE, kernel )
    right_close = cv.morphologyEx( right_bin, cv.MORPH_CLOSE, kernel )

    return left_close, right_close


# bin_close


def bin_dilate( left_bin, right_bin, ksize=(3, 3) ):
    kernel = np.ones( ksize )
    left_dil = cv.dilate( left_bin, kernel )
    right_dil = cv.dilate( right_bin, kernel )

    return left_dil, right_dil


# bin_dilate


def bin_erode( left_bin, right_bin, ksize=(3, 3) ):
    kernel = np.ones( ksize )
    left_erode = cv.erode( left_bin, kernel )
    right_erode = cv.erode( right_bin, kernel )

    return left_erode, right_erode


# bin_erode


def bin_open( left_bin, right_bin, ksize=(6, 6) ):
    kernel = np.ones( ksize )
    left_open = cv.morphologyEx( left_bin, cv.MORPH_OPEN, kernel )
    right_open = cv.morphologyEx( right_bin, cv.MORPH_OPEN, kernel )

    return left_open, right_open


# bin_open


def canny( left_img, right_img, lo_thresh=150, hi_thresh=200 ):
    """ Canny edge detection """

    canny_left = cv.Canny( left_img, lo_thresh, hi_thresh )
    canny_right = cv.Canny( right_img, lo_thresh, hi_thresh )

    return canny_left, canny_right


# canny


def centerline_from_contours( contours, len_thresh: int = -1, bspline_k: int = -1,
                              outlier_thresh: float = -1, num_neighbors: int = -1,
                              scale: tuple = (1, 1) ):
    """ This is to determine the centerline points from the contours using outlier detection
        and bspline smoothing

        bspline fits on the y-axis as the independent variable (where the stereo-matching occurs)

        @param contours: the image contours
        @param len_thresh: a length threshold to remove small length contours.
                                             using np.inf will filter out all but the longest length
                                             contour.
        @param bspline_k: the degree to fit a bspline. If less than 1, bspline will not be fit and
                          will return bspline=None.

        @param outlier_thresh: the outlier thresholding value
        @param num_neighbors: the number of neigbhors parameter needed for the outlier detection

        @param scale: tuple to scale contour points by (Default is (1,1)).

        @return: pts, bspline:
                    pts = [N x 2] numpy array of bspline pts (i, j) image points
                    bspline = None or bspline that was fit to the contours

    """
    # length thresholding
    len_thresh = min( len_thresh, max( [ len( c ) for c in contours ] ) )  # don't go over max length
    contours_filt = [ c for c in contours if len( c ) >= len_thresh ]

    # numpy-tize points
    pts = np.unique( np.vstack( contours_filt ).squeeze(), axis=0 ) * np.array( scale ).reshape( 1, 2 )

    # outlier detection
    if (outlier_thresh >= 0) and (num_neighbors > 0):
        clf = LocalOutlierFactor( n_neighbors=num_neighbors, contamination='auto' )
        clf.fit_predict( pts )
        inliers = np.abs( -1 - clf.negative_outlier_factor_ ) < outlier_thresh

        pts = pts[ inliers ]

    # if: outlier detection

    # fit bspline
    bspline = None
    if bspline_k > 0:
        #         set_trace()
        idx = np.argsort( pts[ :, 1 ] )
        bspline = BSpline1D( pts[ idx, 1 ], pts[ idx, 0 ], k=bspline_k )

        # grab all of the bspline points
        s = np.linspace( 0, 1, 200 )
        pts = np.stack( (bspline.unscale( s ), bspline( bspline.unscale( s ) )) ).T[ :, [ 1, 0 ] ]

    # if: bspline

    return pts / np.array( scale ).reshape( 1, 2 ), bspline


# centerline_from_contours


def color_segmentation( left_img, right_img, color ):
    """ get the pixels of a specific color"""
    # parse color segmentaiton
    if color.lower() in [ 'red', 'r' ]:
        lb = COLOR_HSVRANGE_RED[ 0 ]
        ub = COLOR_HSVRANGE_RED[ 1 ]

        # if

    elif color.lower() in [ 'yellow', 'y' ]:
        lb = COLOR_HSVRANGE_YELLOW[ 0 ]
        ub = COLOR_HSVRANGE_YELLOW[ 1 ]

    # elif

    elif color.lower() in [ 'green', 'g' ]:
        lb = COLOR_HSVRANGE_GREEN[ 0 ]
        ub = COLOR_HSVRANGE_GREEN[ 1 ]

    # elif

    elif color.lower() in [ 'blue', 'b' ]:
        lb = COLOR_HSVRANGE_BLUE[ 0 ]
        ub = COLOR_HSVRANGE_BLUE[ 1 ]

    # elif

    else:
        raise NotImplementedError( f"{color} is not an implemented color HSV range" )

    # else

    # convert into HSV color space
    left_hsv = cv.cvtColor( left_img, cv.COLOR_BGR2HSV )
    right_hsv = cv.cvtColor( right_img, cv.COLOR_BGR2HSV )

    # determine which colors are within the bounds
    left_mask = cv.inRange( left_hsv, lb, ub )
    right_mask = cv.inRange( right_hsv, lb, ub )

    # masked images
    left_color = cv.bitwise_and( left_img, left_img, mask=left_mask )
    right_color = cv.bitwise_and( right_img, right_img, mask=right_mask )

    return left_mask, right_mask, left_color, right_color


# color_segmentation


def connected_component_filtering( bin_img, N_keep: int = 1 ):
    """ Keep the largest 'N_keep' connected components

        @param bin_img: binary image for connected component analysis
        @param N_keep: the number of components to keep

        @return: segmented out connected components
    """

    # run connected components (0 is bg)
    num_labels, labels = cv.connectedComponents( bin_img )

    # determine number of instances per label
    num_instances_labels = [ np.count_nonzero( labels == lbl ) for lbl in range( num_labels ) ]

    # N largest label that is not the background
    labels_sorted = np.argsort( num_instances_labels[ 1: ] ) + 1
    labels_Nkeep = labels_sorted[ :-N_keep - 1:-1 ]

    bin_img_cc = np.isin( labels, labels_Nkeep )

    return bin_img_cc


# connected_component_filtering


def contours( left_skel, right_skel ):
    # unique_l, counts_l = np.unique(left_skel, return_counts=True)
    # unique_r, counts_r = np.unique(right_skel, return_counts=True)
    # print(10*'*',dict(zip(unique_l,counts_l)))
    # print(10*'*',dict(zip(unique_r,counts_r)))
    conts_l, *_ = cv.findContours( left_skel.astype( np.uint8 ), cv.RETR_LIST, cv.CHAIN_APPROX_NONE )
    conts_r, *_ = cv.findContours( right_skel.astype( np.uint8 ), cv.RETR_LIST, cv.CHAIN_APPROX_NONE )
    # unique_l, counts_l = np.unique(conts_l, return_counts=True)
    # unique_r, counts_r = np.unique(conts_r, return_counts=True)
    # print(10*'*',dict(zip(unique_l,counts_l)))
    # print(10*'*',dict(zip(unique_r,counts_r)))
    conts_l = sorted( conts_l, key=len, reverse=True )
    conts_r = sorted( conts_r, key=len, reverse=True )

    return conts_l, conts_r


# contours


def fit_parabola_img( img, pts, window: int, ransac_num_samples: int = 15, ransac_num_trials: int = 2000 ):
    """ Function to fit a parabola to an image and return its minimum coordinate"""
    assert (pts.shape[ 1 ] == 2)  # 2 columns allowed only

    if img.ndim > 2:
        img = cv.cvtColor( img, cv.COLOR_BGR2GRAY )

    # if

    X_min = [ ]
    Pol_ransac = [ ]
    for x, y in pts:
        win_img = img[ y, x - window // 2:x + window // 2 ]
        X = np.arange( x - window // 2, x + window // 2 )

        # fit parabola ransac
        pol_ransac = fit_parabola_ransac( X, win_img, ransac_num_samples, ransac_num_trials )

        # calculate minimum point
        x_min = -pol_ransac[ 1 ] / (2 * pol_ransac[ 0 ])

        # append to list
        X_min.append( x_min )
        Pol_ransac.append( pol_ransac )

    # for

    pts_min = np.vstack( (X_min, pts[ :, 1 ]) ).T

    return pts_min, Pol_ransac


# fit_parabola_img


def fit_parabola_ransac( X, Y, num_samples: int, num_trials: int = 2000 ):
    """ Function to fit a 1D parabola using RANSAC based on steepness (coefficient on x**2)"""
    # argument checking
    assert (len( X ) == len( Y ))

    # initializations
    most_inliers = -1
    steepness = -np.inf
    pol = None
    for i in range( num_trials ):
        # grab random number of samples
        choices = np.random.choice( len( X ), num_samples, replace=False )

        # grab the points
        X_choice = X[ choices ]
        Y_choice = Y[ choices ]

        # get the polynomial fit
        pol_i = np.polyfit( X_choice, Y_choice, 2 )
        steepness_i = pol_i[ 0 ]

        # update if steeper
        if (steepness_i > steepness) or (pol is None):
            pol = pol_i
            steepness = steepness_i
        # if

    # for

    return pol


# fit_parabola_ransac


def gauss_blur( left_img, right_img, ksize, sigma: tuple = (0, 0) ):
    """ gaussian blur """
    left_blur = cv.GaussianBlur( left_img, ksize, sigmaX=sigma[ 0 ], sigmaY=sigma[ 0 ] )
    right_blur = cv.GaussianBlur( right_img, ksize, sigmaX=sigma[ 0 ], sigmaY=sigma[ 0 ] )

    return left_blur, right_blur


# gauss_blur


def _gridproc_stereo( left_img, right_img,
                      bor_l=None, bor_r=None,
                      proc_show: bool = False ):
    """ DEPRECATED wrapper function to segment the grid out of a stereo pair """
    # convert to grayscale if not already
    if bor_l is None:
        bor_l = [ ]
    if bor_r is None:
        bor_r = [ ]
    if left_img.ndim > 2:
        left_img = cv.cvtColor( left_img, cv.COLOR_BGR2GRAY )
        right_img = cv.cvtColor( right_img, cv.COLOR_BGR2GRAY )

    # if

    # start the image qprocessing
    left_blur, right_blur = gauss_blur( left_img, right_img, (5, 5) )
    left_thresh, right_thresh = thresh( 2.5 * left_blur, 2.5 * right_blur )

    left_thresh_bo = blackout_regions( left_thresh, bor_l )
    right_thresh_bo = blackout_regions( right_thresh, bor_r )

    left_med, right_med = median_blur( left_thresh_bo, right_thresh_bo, 5 )

    left_close, right_close = bin_close( left_med, right_med, ksize=(5, 5) )
    left_open, right_open = bin_open( left_close, right_close, ksize=(3, 3) )
    left_close2, right_close2 = bin_close( left_open, right_open, ksize=(7, 7) )
    left_skel, right_skel = skeleton( left_close2, right_close2 )

    # hough line transform
    hough_thresh = 200
    left_lines = np.squeeze( cv.HoughLines( left_skel.astype( np.uint8 ), 2, np.pi / 180, hough_thresh ) )
    right_lines = np.squeeze( cv.HoughLines( right_skel.astype( np.uint8 ), 2, np.pi / 180, hough_thresh ) )

    print( '# left lines:', len( left_lines ) )
    print( '# right lines:', len( right_lines ) )

    # # draw the hough lines
    left_im_lines = cv.cvtColor( left_img, cv.COLOR_GRAY2RGB )
    for rho, theta in left_lines:
        a = np.cos( theta )
        b = np.sin( theta )
        x0 = a * rho
        y0 = b * rho
        x1 = int( x0 + 1000 * (-b) )
        y1 = int( y0 + 1000 * a )
        x2 = int( x0 - 1000 * (-b) )
        y2 = int( y0 - 1000 * a )

        cv.line( left_im_lines, (x1, y1), (x2, y2), (255, 0, 0), 2 )

    # for

    right_im_lines = cv.cvtColor( right_img, cv.COLOR_GRAY2RGB )
    for rho, theta in right_lines:
        a = np.cos( theta )
        b = np.sin( theta )
        x0 = a * rho
        y0 = b * rho
        x1 = int( x0 + 1000 * (-b) )
        y1 = int( y0 + 1000 * a )
        x2 = int( x0 - 1000 * (-b) )
        y2 = int( y0 - 1000 * a )

        cv.line( right_im_lines, (x1, y1), (x2, y2), (255, 0, 0), 2 )

    # for

    # harris corner detection
    #     left_centroid = cv.cornerHarris( left_open, 2, 5, 0.04 )
    #     left_centroid = cv.dilate( left_centroid, None )
    #     _, left_corners = cv.threshold( left_centroid, 0.2 * left_centroid.max(),
    #                                 255, 0 )
    #     left_corners = np.int0( left_corners )
    #
    #     right_centroid = cv.cornerHarris( right_open, 2, 5, 0.04 )
    #     right_centroid = cv.dilate( right_centroid, None )
    #     _, right_corners = cv.threshold( right_centroid, 0.2 * right_centroid.max(),
    #                                 255, 0 )
    #     right_corners = np.int0( right_corners )
    #
    #     left_crnr = cv.cvtColor( left_img, cv.COLOR_GRAY2RGB )
    #     right_crnr = cv.cvtColor( right_img, cv.COLOR_GRAY2RGB )
    #     left_crnr[left_corners] = [255, 0, 0]
    #     right_crnr[right_corners] = [255, 0, 0]

    # plotting
    if proc_show:
        plt.ion()

        plt.figure()
        plt.imshow( imconcat( left_blur, right_blur ), cmap='gray' )
        plt.title( "gaussian blurring" )

        plt.figure()
        plt.imshow( imconcat( left_thresh, right_thresh, 150 ), cmap='gray' )
        plt.title( 'adaptive thresholding: after blurring' )

        plt.figure()
        plt.imshow( imconcat( left_thresh_bo, right_thresh_bo, 150 ), cmap='gray' )
        plt.title( 'region suppression: after thresholding' )

        plt.figure()
        plt.imshow( imconcat( left_med, right_med, 150 ), cmap='gray' )
        plt.title( 'median: after region suppression' )

        plt.figure()
        plt.imshow( imconcat( left_close, right_close, 150 ), cmap='gray' )
        plt.title( 'closing: after median' )

        plt.figure()
        plt.imshow( imconcat( left_open, right_open, 150 ), cmap='gray' )
        plt.title( 'opening: after closing' )

        plt.figure()
        plt.imshow( imconcat( left_close2, right_close2, 150 ), cmap='gray' )
        plt.title( 'closing 2: after opening' )

        plt.figure()
        plt.imshow( imconcat( left_skel, right_skel, 150 ), cmap='gray' )
        plt.title( 'skeletonize: after closing 2' )

        plt.figure()
        plt.imshow( imconcat( left_im_lines, right_im_lines ) )
        plt.title( 'hough lines transform' )

    # if


# _gridproc_stereo


def gridproc_stereo( left_img, right_img,
                     bor_l=None, bor_r=None,
                     proc_show: bool = False ):
    """ wrapper function to segment the grid out of a stereo pair """
    # convert to grayscale if not already
    if bor_l is None:
        bor_l = [ ]
    if bor_r is None:
        bor_r = [ ]
    if left_img.ndim > 2:
        left_img = cv.cvtColor( left_img, cv.COLOR_BGR2GRAY )
        right_img = cv.cvtColor( right_img, cv.COLOR_BGR2GRAY )

    # if

    # start the image processing
    left_canny, right_canny = canny( left_img, right_img, 20, 60 )
    left_bo = blackout_regions( left_canny, bor_l )
    right_bo = blackout_regions( right_canny, bor_r )

    # ====================== STANDARD HOUGH TRANSFORM  ==============================
    #     # hough line transform
    #     hough_thresh = 450
    #     left_lines = np.squeeze( cv.HoughLines( left_bo, 2, np.pi / 180, hough_thresh ) )
    #     right_lines = np.squeeze( cv.HoughLines( right_bo, 2, np.pi / 180, hough_thresh ) )
    #
    #     print( 'Hough Transform' )
    #     print( '# left lines:', len( left_lines ) )
    #     print( '# right lines:', len( right_lines ) )
    #     print()
    #
    #     # # draw the hough lines
    #     left_im_lines = cv.cvtColor( left_img, cv.COLOR_GRAY2RGB )
    #     for rho, theta in left_lines:
    #         a = np.cos( theta )
    #         b = np.sin( theta )
    #         x0 = a * rho
    #         y0 = b * rho
    #         x1 = int( x0 + 1000 * ( -b ) )
    #         y1 = int( y0 + 1000 * ( a ) )
    #         x2 = int( x0 - 1000 * ( -b ) )
    #         y2 = int( y0 - 1000 * ( a ) )
    #
    #         cv.line( left_im_lines, ( x1, y1 ), ( x2, y2 ), ( 255, 0, 0 ), 2 )
    #
    #     # for
    #
    #     right_im_lines = cv.cvtColor( right_img, cv.COLOR_GRAY2RGB )
    #     for rho, theta in right_lines:
    #         a = np.cos( theta )
    #         b = np.sin( theta )
    #         x0 = a * rho
    #         y0 = b * rho
    #         x1 = int( x0 + 1000 * ( -b ) )
    #         y1 = int( y0 + 1000 * ( a ) )
    #         x2 = int( x0 - 1000 * ( -b ) )
    #         y2 = int( y0 - 1000 * ( a ) )
    #
    #         cv.line( right_im_lines, ( x1, y1 ), ( x2, y2 ), ( 255, 0, 0 ), 2 )
    #
    #     # for
    # ===============================================================================

    # prob. hough line transform
    minlinelength = int( 0.8 * left_img.shape[ 1 ] )
    maxlinegap = 20
    hough_thresh = 100
    left_linesp = np.squeeze( cv.HoughLinesP( left_bo, 1, np.pi / 180, hough_thresh, minlinelength, maxlinegap ) )
    right_linesp = np.squeeze( cv.HoughLinesP( right_bo, 1, np.pi / 180, hough_thresh, minlinelength, maxlinegap ) )

    print( 'Probabilisitic Hough Transform' )
    print( "min. line length, max line gap: ", minlinelength, maxlinegap )
    print( '# left lines:', left_linesp.shape )
    print( '# right lines:', right_linesp.shape )
    print()

    # # Draw probabilistic hough lines
    left_im_linesp = cv.cvtColor( left_img, cv.COLOR_GRAY2RGB )
    left_houghp = np.zeros( left_im_linesp.shape[ 0:2 ], dtype=np.uint8 )
    for x1, y1, x2, y2 in left_linesp:
        cv.line( left_im_linesp, (x1, y1), (x2, y2), (255, 0, 0), 2 )
        cv.line( left_houghp, (x1, y1), (x2, y2), (255, 255, 255), 1 )

    # for

    right_im_linesp = cv.cvtColor( right_img, cv.COLOR_GRAY2RGB )
    right_houghp = np.zeros( right_im_linesp.shape[ 0:2 ], dtype=np.uint8 )
    for x1, y1, x2, y2 in right_linesp:
        cv.line( right_im_linesp, (x1, y1), (x2, y2), (255, 0, 0), 2 )
        cv.line( right_houghp, (x1, y1), (x2, y2), (255, 255, 255), 1 )

    # for

    # hough lines on prob. hough lines image
    # hough line transform
    hough_thresh = 100
    left_lines2 = np.squeeze( cv.HoughLines( left_houghp, 1, np.pi / 180, hough_thresh ) )
    right_lines2 = np.squeeze( cv.HoughLines( right_houghp, 1, np.pi / 180, hough_thresh ) )

    print( 'Hough Transform (2)' )
    print( '# left lines:', len( left_lines2 ) )
    print( '# right lines:', len( right_lines2 ) )
    print()

    # # draw the hough lines
    left_im_lines2 = cv.cvtColor( left_img, cv.COLOR_GRAY2RGB )
    for rho, theta in left_lines2:
        a = np.cos( theta )
        b = np.sin( theta )
        x0 = a * rho
        y0 = b * rho
        x1 = int( x0 + 1000 * (-b) )
        y1 = int( y0 + 1000 * a )
        x2 = int( x0 - 1000 * (-b) )
        y2 = int( y0 - 1000 * a )

        cv.line( left_im_lines2, (x1, y1), (x2, y2), (255, 0, 0), 2 )

    # for

    right_im_lines2 = cv.cvtColor( right_img, cv.COLOR_GRAY2RGB )
    for rho, theta in right_lines2:
        a = np.cos( theta )
        b = np.sin( theta )
        x0 = a * rho
        y0 = b * rho
        x1 = int( x0 + 1000 * (-b) )
        y1 = int( y0 + 1000 * a )
        x2 = int( x0 - 1000 * (-b) )
        y2 = int( y0 - 1000 * a )

        cv.line( right_im_lines2, (x1, y1), (x2, y2), (255, 0, 0), 2 )

    # for

    # plotting
    if proc_show:
        plt.ion()

        plt.figure()
        plt.imshow( imconcat( left_canny, right_canny, 150 ), cmap='gray' )
        plt.title( 'canny' )

        plt.figure()
        plt.imshow( imconcat( left_bo, right_bo, 150 ), cmap='gray' )
        plt.title( 'region suppression: after thresholding' )

        plt.figure()
        plt.imshow( imconcat( left_im_linesp, right_im_linesp ) )
        plt.title( 'probabilistic hough lines transform' )

        plt.figure()
        plt.imshow( imconcat( left_houghp, right_houghp, 150 ), cmap='gray' )
        plt.title( 'prob. hough lines transform (2)' )

        plt.figure()
        plt.imshow( imconcat( left_im_lines2, right_im_lines2 ) )
        plt.title( 'hough lines transform: after prob. hough. lines (2)' )

    # if


# gridproc_stereo


def houghlines( left_img, right_img ):
    """ function for performing hough lines transform on a stereo pair """

    # TODO


# houghlines


def hough_quadratic( img ):
    """ hough transform to fit a quadratic

        Want to fit a function y = a x**2 + b x + c

        This will pickout the SINGLE argmax

    """
    raise NotImplementedError( 'hough_quadratic is not yet implemented.' )


# hough_quadratic


def load_stereoparams_matlab( param_file: str ):
    """ Loads the matlab stereo parameter file created from a struct """

    mat = sio.loadmat( param_file )

    stereo_params = { }

    keys = [ 'cameraMatrix1', 'cameraMatrix2', 'distCoeffs1',
             'distCoeffs2', 'R1', 'tvecs1', 'R2', 'tvecs2',
             'R', 't', 'F', 'E', 'units' ]

    # load stereo parameters
    for key in keys:
        if key == 'units':
            stereo_params[ key ] = mat[ key ][ 0 ]

        elif (key == 'R1') or (key == 'R2'):
            stereo_params[ key + '_ext' ] = mat[ key ]

        else:
            stereo_params[ key ] = mat[ key ]

    # for

    # projection matrices
    R1, R2, P1, P2, Q, *_ = cv.stereoRectify( stereo_params[ 'cameraMatrix1' ], stereo_params[ 'distCoeffs1' ],
                                              stereo_params[ 'cameraMatrix2' ], stereo_params[ 'distCoeffs2' ],
                                              (768, 1024), stereo_params[ 'R' ], stereo_params[ 't' ] )
    R = stereo_params[ 'R' ]
    t = stereo_params[ 't' ]
    H = np.vstack( (np.hstack( (R, t.reshape( 3, 1 )) ), [ 0, 0, 0, 1 ]) )

    stereo_params[ 'R1' ] = R1
    stereo_params[ 'R2' ] = R2
    stereo_params[ 'P1' ] = P1  # stereo_params['cameraMatrix1'] @ np.eye( 3, 4 )
    stereo_params[ 'P2' ] = P2  # stereo_params['cameraMatrix2'] @ H[:-1]
    stereo_params[ 'Q' ] = Q

    return stereo_params


# load_stereoparams_matlab


def imconcat( left_im, right_im, pad_val: Union[ int, list ] = 0, pad_size: int = 20 ):
    """ wrapper for concatenating images"""

    if left_im.ndim == 2:
        padding = np.array( pad_val ) * np.ones( (left_im.shape[ 0 ], pad_size), dtype=left_im.dtype )

    # if

    elif left_im.ndim == 3:
        padding = np.array( pad_val ) * np.ones( (left_im.shape[ 0 ], pad_size, left_im.shape[ 2 ]),
                                                 dtype=left_im.dtype )

    # elif

    else:
        raise IndexError( "Left image must be a 2 or 3-d array" )

    # else

    return np.concatenate( (left_im, padding, right_im), axis=1 )


# imconcat

def imsplit( left_right_im, img_size: tuple ):
    N_rows, N_cols, *_ = img_size

    left_im = left_right_im[ :N_rows, :N_cols ]
    right_im = left_right_im[ :N_rows, -N_cols: ]

    return left_im, right_im


# imsplit


def imgproc_jig( left_img, right_img, bor_l=None, bor_r=None,
                 roi_l: tuple = (), roi_r: tuple = (),
                 proc_show: bool = False ):
    """ wrapper function to process the left and right image pair for needle
        centerline identification

     """

    # convert to grayscale if not already
    if bor_l is None:
        bor_l = [ ]
    if bor_r is None:
        bor_r = [ ]
    if left_img.ndim > 2:
        left_img = cv.cvtColor( left_img, cv.COLOR_BGR2GRAY )

    # if
    if right_img.ndim > 2:
        right_img = cv.cvtColor( right_img, cv.COLOR_BGR2GRAY )

    # if

    # start the image processing
    left_thresh, right_thresh = thresh( left_img, right_img, thresh=75 )

    left_roi = roi( left_thresh, roi_l, full=True )
    right_roi = roi( right_thresh, roi_r, full=True )

    left_thresh_bo = blackout_regions( left_roi, bor_l )
    right_thresh_bo = blackout_regions( right_roi, bor_r )

    left_tmed, right_tmed = median_blur( left_thresh_bo, right_thresh_bo, ksize=7 )

    left_close, right_close = bin_close( left_tmed, right_tmed, ksize=(7, 7) )

    left_open, right_open = bin_open( left_close, right_close, ksize=(3, 3) )

    left_dil, right_dil = bin_erode( left_close, right_close, ksize=(3, 3) )

    left_skel, right_skel = skeleton( left_dil, right_dil )

    # get the contours ( sorted by length)
    conts_l, conts_r = contours( left_skel, right_skel )

    # ===========================================================================
    #
    # # grab b-spline points
    # pts_l = np.unique( np.vstack( conts_l ).squeeze(), axis = 0 )
    # pts_r = np.unique( np.vstack( conts_r ).squeeze(), axis = 0 )
    #
    # # remove outliers
    # clf = LocalOutlierFactor( n_neighbors = 20, contamination = 'auto' )
    # clf.fit_predict( pts_l )
    # inliers_l = np.abs( -1 - clf.negative_outlier_factor_ ) < 0.5
    #
    # clf.fit_predict( pts_r )
    # inliers_r = np.abs( -1 - clf.negative_outlier_factor_ ) < 0.5
    #
    # pts_l_in = pts_l[inliers_l]
    # pts_r_in = pts_r[inliers_r]
    #
    # bspline_l = BSpline1D( pts_l_in[:, 0], pts_l_in[:, 1], k = 3 )
    # bspline_r = BSpline1D( pts_r_in[:, 0], pts_r_in[:, 1], k = 3 )
    #
    # # grab all of the bspline points
    # s = np.linspace( 0, 1, 200 )
    # bspline_pts_l = np.vstack( ( bspline_l.unscale( s ), bspline_l( bspline_l.unscale( s ) ) ) ).T
    # bspline_pts_r = np.vstack( ( bspline_r.unscale( s ), bspline_r( bspline_r.unscale( s ) ) ) ).T
    # ===========================================================================

    if proc_show:
        plt.ion()

        plt.figure()
        plt.imshow( imconcat( left_thresh, right_thresh, 150 ), cmap='gray' )
        plt.title( 'adaptive thresholding' )

        plt.figure()
        plt.imshow( imconcat( left_roi, right_roi, 150 ), cmap='gray' )
        plt.title( 'roi: after thresholding' )

        plt.figure()
        plt.imshow( imconcat( left_thresh_bo, right_thresh_bo, 150 ), cmap='gray' )
        plt.title( 'region suppression: roi' )

        plt.figure()
        plt.imshow( imconcat( left_tmed, right_tmed, 150 ), cmap='gray' )
        plt.title( 'median filtering: after region suppression' )

        plt.figure()
        plt.imshow( imconcat( left_close, right_close, 150 ), cmap='gray' )
        plt.title( 'closing: after median' )

        plt.figure()
        plt.imshow( imconcat( left_open, right_open, 150 ), cmap='gray' )
        plt.title( 'opening: after closing' )

        plt.figure()
        plt.imshow( imconcat( left_dil, right_dil, 150 ), cmap='gray' )
        plt.title( 'dilation: after opening' )

        plt.figure()
        plt.imshow( imconcat( left_skel, right_skel, 150 ), cmap='gray' )
        plt.title( 'skeletization: after dilation' )

        cont_left = left_img.copy().astype( np.uint8 )
        cont_right = right_img.copy().astype( np.uint8 )

        cont_left = cv.cvtColor( cont_left, cv.COLOR_GRAY2RGB )
        cont_right = cv.cvtColor( cont_right, cv.COLOR_GRAY2RGB )

        cv.drawContours( cont_left, conts_l, -1, (255, 0, 0), 3 )
        cv.drawContours( cont_right, conts_r, -1, (255, 0, 0), 3 )
        plt.figure()
        plt.imshow( imconcat( cont_left, cont_right, [ 0, 0, 255 ] ) )
        plt.title( 'contours' )

        # =======================================================================
        # cont_l_filt = [np.vstack( ( pts_l_in, np.flip( pts_l_in, axis = 0 ) ) ).astype( int )]
        # cont_r_filt = [np.vstack( ( pts_r_in, np.flip( pts_r_in, axis = 0 ) ) ).astype( int )]
        #
        # cv.drawContours( cont_left, cont_l_filt, -1, ( 255, 0, 0 ), 6 )
        # cv.drawContours( cont_right, cont_r_filt, -1, ( 255, 0, 0 ), 6 )
        #
        # cv.drawContours( cont_left, conts_l, 0, ( 0, 255, 0 ), 3 )
        # cv.drawContours( cont_right, conts_r, 0, ( 0, 255, 0 ), 3 )
        #
        # plt.figure()
        # plt.imshow( imconcat( cont_left, cont_right, 150 ), cmap = 'gray' )
        # plt.title( 'contours' )
        #
        # impad = 20
        # bspl_left_img = cv.cvtColor( left_img.copy().astype( np.uint8 ), cv.COLOR_GRAY2RGB )
        # bspl_right_img = cv.cvtColor( right_img.copy().astype( np.uint8 ), cv.COLOR_GRAY2RGB )
        #
        # plt.figure()
        # plt.imshow( imconcat( bspl_left_img, bspl_right_img, [0, 0, 255], pad_size = impad ) )
        # plt.plot( bspline_pts_l[:, 0], bspline_pts_l[:, 1], 'r-' )
        # plt.plot( bspline_pts_r[:, 0] + left_img.shape[1] + impad, bspline_pts_r[:, 1], 'r-' )
        # plt.title( 'bspline fits' )
        # =======================================================================

        # close on enter
        print( 'Press any key on the last figure to close all windows.' )
        plt.show()
        while True:
            try:
                if plt.waitforbuttonpress( 0 ):
                    break

                # if
            # try

            except:
                break

            # except
        # while

        print( 'Closing all windows...' )
        plt.close( 'all' )
        print( 'Plotting finished.', end='\n\n' + 80 * '=' + '\n\n' )

    # if

    return left_skel, right_skel, conts_l, conts_r


# imgproc_jig


def median_blur( left_thresh, right_thresh, ksize=11 ):
    left_med = cv.medianBlur( left_thresh, ksize )
    right_med = cv.medianBlur( right_thresh, ksize )

    return left_med, right_med


# median_blur


def meanshift( left_bin, right_bin, q=0.3, n_samps: int = 200, plot_lbls: bool = False ):
    # get non-zero coordinates
    yl, xl = np.nonzero( left_bin )
    pts_l = np.vstack( (xl, yl) ).T

    yr, xr = np.nonzero( right_bin )
    pts_r = np.vstack( (xr, yr) ).T

    # estimate meanshift bandwidth
    bandwidth_l = estimate_bandwidth( pts_l, quantile=q, n_samples=n_samps )
    bandwidth_r = estimate_bandwidth( pts_r, quantile=q, n_samples=n_samps )

    # meanshift fit
    ms_l = MeanShift( bandwidth=bandwidth_l, bin_seeding=True )
    ms_r = MeanShift( bandwidth=bandwidth_r, bin_seeding=True )

    ms_l.fit( pts_l )
    ms_r.fit( pts_r )

    # get the labels
    left_lbls = ms_l.labels_
    right_lbls = ms_r.labels_

    # plot the clusters
    cols = list( pltcolors.CSS4_COLORS.values() )
    if plot_lbls:
        # plot left
        nlbls = len( np.unique( left_lbls ) )

        plt.figure()
        for i, c in zip( range( nlbls ), cols[ :nlbls ] ):
            members = (left_lbls == i)

            plt.plot( xl[ members ], yl[ members ], 'o', color=c, markersize=12 )

        # for
        plt.gca().invert_yaxis()

        # plot right
        nlbls = len( np.unique( right_lbls ) )

        plt.figure()
        for i, c in zip( range( nlbls ), cols[ :nlbls ] ):
            members = (right_lbls == i)

            plt.plot( xr[ members ], yr[ members ], 'o', color=c, markersize=12 )

        # for
        plt.gca().invert_yaxis()

        plt.show()

    # if

    return (pts_l, left_lbls), (pts_r, right_lbls)


# meanshift


def needle_jig_reconstruction( img_left, img_right, stereo_params,
                               bor_l: list = None, bor_r: list = None,
                               roi_l=(), roi_r=(),
                               alpha: float = 0.5, recalc_stereo: bool = False,
                               proc_show: bool = False ):
    """ This is function for performing a 3-D needle reconstruction from a raw stereo pair
        from the jig experiment.
    """
    # argument checking
    if bor_l is None:
        bor_l = [ ]
    if bor_r is None:
        bor_r = [ ]

    # prepare images
    # # roi the images
    left_roi = roi( img_left, roi_l, full=True )
    right_roi = roi( img_right, roi_r, full=True )
    ret_images = { 'roi': imconcat( left_roi, right_roi, [ 0, 0, 255 ] ) }  # return image pairs

    # # black-out regions
    left_roibo = blackout_regions( left_roi, bor_l )
    right_roibo = blackout_regions( right_roi, bor_r )
    ret_images[ 'roi-bo' ] = imconcat( left_roibo, right_roibo, [ 0, 0, 255 ] )

    # stereo rectify the images
    left_rect, right_rect, _, map_l, map_r = stereo_rectify( left_roibo, right_roibo, stereo_params,
                                                             interp_method=cv.INTER_LINEAR, alpha=alpha,
                                                             recalc_stereo=recalc_stereo )
    ret_images[ 'rect' ] = imconcat( left_rect, right_rect, [ 0, 0, 255 ] )

    # # map the rois
    boroi_l_img = roi_mask( roi_l, img_left.shape ) & blackout_image( bor_l, img_left.shape )
    boroi_r_img = roi_mask( roi_r, img_right.shape ) & blackout_image( bor_r, img_right.shape )
    ret_images[ 'roi-bo-bool' ] = imconcat( 255 * boroi_l_img, 255 * boroi_r_img, 125 )

    boroi_l_mapped = cv.remap( boroi_l_img.astype( np.uint8 ), map_l[ 0 ], map_l[ 1 ], cv.INTER_NEAREST )
    boroi_r_mapped = cv.remap( boroi_r_img.astype( np.uint8 ), map_r[ 0 ], map_r[ 1 ], cv.INTER_NEAREST )
    ret_images[ 'roi-bo-bool-mapped' ] = imconcat( 255 * boroi_l_mapped, 255 * boroi_r_mapped, 125 )

    # perform image processing on rectified images
    left_gray = cv.cvtColor( left_rect, cv.COLOR_BGR2GRAY )
    right_gray = cv.cvtColor( right_rect, cv.COLOR_BGR2GRAY )
    left_thresh, right_thresh = thresh( left_gray, right_gray, thresh=50 )

    # # remove extra borders threshed out
    left_thresh *= boroi_l_mapped.astype( np.uint8 )
    right_thresh *= boroi_r_mapped.astype( np.uint8 )
    ret_images[ 'thresh-rect' ] = imconcat( left_thresh, right_thresh, 125 )

    # get the contours and filter out outliers
    left_skel, right_skel = skeleton( left_thresh, right_thresh )
    ret_images[ 'skel-rect' ] = imconcat( 255 * left_skel.astype( np.uint8 ), 255 * right_skel.astype( np.uint8 ), 125 )
    conts_l, conts_r = contours( left_skel, right_skel )

    # # outlier options
    len_thresh = 5
    bspl_k = 2
    out_thresh = 1.25
    n_neigh = 60

    pts_l, bspline_l = centerline_from_contours( conts_l,
                                                 len_thresh=len_thresh,
                                                 bspline_k=bspl_k,
                                                 outlier_thresh=out_thresh,
                                                 num_neighbors=n_neigh )

    pts_r, bspline_r = centerline_from_contours( conts_r,
                                                 len_thresh=len_thresh,
                                                 bspline_k=bspl_k,
                                                 outlier_thresh=out_thresh,
                                                 num_neighbors=n_neigh )

    left_rect_draw = cv.polylines( left_rect.copy(), [ pts_l.reshape( -1, 1, 2 ).astype( np.int32 ) ], False,
                                   (255, 0, 0), 5 )
    right_rect_draw = cv.polylines( right_rect.copy(), [ pts_r.reshape( -1, 1, 2 ).astype( np.int32 ) ], False,
                                    (255, 0, 0), 5 )
    ret_images[ 'contours' ] = imconcat( left_rect_draw, right_rect_draw, [ 0, 0, 255 ] )

    # stereo matching
    pts_l_match, pts_r_match = stereomatch_needle( pts_l, pts_r, method='disparity',
                                                   bspline_l=bspline_l, bspline_r=bspline_r )

    # stereo triangulation
    pts_3d = cv.triangulatePoints( stereo_params[ 'P1' ], stereo_params[ 'P2' ], pts_l_match.T, pts_r_match.T )
    pts_3d /= pts_3d[ -1, : ]
    pts_3d = pts_3d[ :-1, : ].T

    figures = { }
    # show processing
    if proc_show:
        figures[ 'rect' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imconcat( left_rect, right_rect, [ 255, 0, 0 ] ) )
        plt.title( 'Rectified Images' )

        figures[ 'thresh-rect' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imconcat( left_thresh, right_thresh, 125 ), cmap='gray' )
        plt.title( 'Thresholded rectified images' )

        figures[ 'skel' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imconcat( left_skel, right_skel, 125 ), cmap='gray' )
        plt.title( 'Skeletonized threshold' )

        plt.figure( figsize=(12, 8) )
        plt.imshow( imconcat( left_rect_draw, right_rect_draw, [ 255, 0, 0 ] ) )
        plt.title( 'centerline points' )

        fig = plt.figure( figsize=(8, 8) )
        plt.plot( pts_l_match[ :, 0 ] - pts_r_match[ :, 0 ] )
        plt.title( 'disparity' )

        fig = plt.figure( figsize=(8, 8) )
        ax = plt.gca( projection='3d' )
        ax.plot( pts_3d[ :, 0 ], pts_3d[ :, 1 ], pts_3d[ :, 2 ] )
        plt.title( '3-D reconstruction: axes NOT scaled' )

        # plt.show()

    # if

    return pts_3d, pts_l, pts_r, bspline_l, bspline_r, ret_images, figures


# needle_jig_reconstruction


def needle_jig_reconstruction_refined( img_left, img_right, stereo_params,
                                       bor_l=None, bor_r=None,
                                       roi_l=(), roi_r=(),
                                       alpha: float = 0.5, recalc_stereo: bool = False,
                                       zoom: float = 1.0, winsize: tuple = (31, 21),
                                       proc_show: bool = False ):
    """ This is function for performing a 3-D needle reconstruction from a raw stereo pair"""
    # prepare images
    # # roi the images
    if bor_l is None:
        bor_l = [ ]
    if bor_r is None:
        bor_r = [ ]
    left_roi = roi( img_left, roi_l, full=True )
    right_roi = roi( img_right, roi_r, full=True )

    # # black-out regions
    left_roibo = blackout_regions( left_roi, bor_l )
    right_roibo = blackout_regions( right_roi, bor_r )
    imgs_ret = { 'roibo': imconcat( left_roibo, right_roibo, [ 0, 0, 255 ] ) }

    # stereo rectify the images
    left_rect, right_rect, _, map_l, map_r = stereo_rectify( left_roibo, right_roibo, stereo_params,
                                                             interp_method=cv.INTER_LINEAR, alpha=alpha,
                                                             recalc_stereo=recalc_stereo )
    imgs_ret[ 'rect' ] = imconcat( left_rect, right_rect, [ 0, 0, 255 ] )

    # # map the rois
    boroi_l_img = roi_mask( roi_l, img_left.shape ) & blackout_image( bor_l, img_left.shape )
    boroi_r_img = roi_mask( roi_r, img_right.shape ) & blackout_image( bor_r, img_right.shape )
    imgs_ret[ 'roi-bo-bool' ] = imconcat( 255 * boroi_l_img, 255 * boroi_r_img, 125 )

    boroi_l_mapped = cv.remap( boroi_l_img.astype( np.uint8 ), map_l[ 0 ], map_l[ 1 ], cv.INTER_NEAREST )
    boroi_r_mapped = cv.remap( boroi_r_img.astype( np.uint8 ), map_r[ 0 ], map_r[ 1 ], cv.INTER_NEAREST )
    imgs_ret[ 'boroi-bool-mapped' ] = imconcat( 255 * boroi_l_mapped, 255 * boroi_r_mapped, 125 )

    # perform image processing on rectified images
    left_gray = cv.cvtColor( left_rect, cv.COLOR_BGR2GRAY )
    right_gray = cv.cvtColor( right_rect, cv.COLOR_BGR2GRAY )
    imgs_ret[ 'gray-rect' ] = imconcat( left_gray, right_gray, 0 )
    left_thresh, right_thresh = thresh( left_gray, right_gray, thresh=50 )

    # # remove extra borders threshed out
    left_thresh *= boroi_l_mapped.astype( np.uint8 )
    right_thresh *= boroi_r_mapped.astype( np.uint8 )
    imgs_ret[ 'thresh-rect' ] = imconcat( left_thresh, right_thresh, 125 )

    # get the contours and filter out outliers
    left_skel, right_skel = skeleton( left_thresh, right_thresh )
    imgs_ret[ 'skel' ] = imconcat( 255 * left_skel, 255 * right_skel, 125 ).astype( np.uint8 )
    conts_l, conts_r = contours( left_skel, right_skel )

    # # outlier options
    len_thresh = 5
    bspl_k = 2
    out_thresh = 1.25
    n_neigh = 60

    pts_l, bspline_l = centerline_from_contours( conts_l,
                                                 len_thresh=len_thresh,
                                                 bspline_k=bspl_k,
                                                 outlier_thresh=out_thresh,
                                                 num_neighbors=n_neigh )

    pts_r, bspline_r = centerline_from_contours( conts_r,
                                                 len_thresh=len_thresh,
                                                 bspline_k=bspl_k,
                                                 outlier_thresh=out_thresh,
                                                 num_neighbors=n_neigh )

    left_rect_draw = cv.polylines( left_rect.copy(), [ pts_l.reshape( -1, 1, 2 ).astype( np.int32 ) ], False,
                                   (255, 0, 0), 5 )
    right_rect_draw = cv.polylines( right_rect.copy(), [ pts_r.reshape( -1, 1, 2 ).astype( np.int32 ) ], False,
                                    (255, 0, 0), 5 )
    imgs_ret[ 'contours' ] = imconcat( left_rect_draw, right_rect_draw, [ 0, 0, 255 ] )

    # stereo matching
    left_rect_gray = cv.cvtColor( left_rect, cv.COLOR_BGR2GRAY )
    right_rect_gray = cv.cvtColor( right_rect, cv.COLOR_BGR2GRAY )
    pts_l_match, pts_r_match = stereomatch_normxcorr( pts_l, pts_r,
                                                      left_rect_gray, right_rect_gray,
                                                      winsize=winsize, zoom=zoom )
    idx_l = np.argsort( pts_l_match[ :, 1 ] )
    idx_r = np.argsort( pts_r_match[ :, 1 ] )
    pts_l_match = pts_l_match[ idx_l ]
    pts_r_match = pts_r_match[ idx_r ]

    # bspline fit the matching points
    bspline_l_match = BSpline1D( pts_l_match[ :, 1 ], pts_l_match[ :, 0 ], k=bspl_k )
    bspline_r_match = BSpline1D( pts_r_match[ :, 1 ], pts_r_match[ :, 0 ], k=bspl_k )
    
    pts_l_match = np.vstack( (bspline_l_match.eval_unscale( s ), bspline_l_match.unscale( s )) ).T
    pts_r_match = np.vstack( (bspline_r_match.eval_unscale( s ), bspline_r_match.unscale( s )) ).T

    # = add to images
    left_rect_match_draw = cv.polylines( left_rect.copy(),
                                         [ pts_l_match.reshape( -1, 1, 2 ).astype( np.int32 ) ], False, (255, 0, 0), 5 )
    right_rect_match_draw = cv.polylines( right_rect.copy(),
                                          [ pts_r_match.reshape( -1, 1, 2 ).astype( np.int32 ) ], False, (255, 0, 0),
                                          5 )
    imgs_ret[ 'contours-match' ] = imconcat( left_rect_match_draw, right_rect_match_draw, [ 0, 0, 255 ] )

    # stereo
    pts_3d = cv.triangulatePoints( stereo_params[ 'P1' ], stereo_params[ 'P2' ], pts_l_match.T, pts_r_match.T )
    pts_3d /= pts_3d[ -1, : ]
    pts_3d = pts_3d.T

    # show processing
    figs_ret = { }
    if proc_show:
        figs_ret[ 'rect' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imconcat( left_rect, right_rect, [ 0, 0, 255 ] )[ :, :, ::-1 ] )
        plt.title( 'Rectified Images' )

        figs_ret[ 'thresh-rect' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imconcat( left_thresh, right_thresh, 125 ), cmap='gray' )
        plt.title( 'Thresholded rectified images' )

        figs_ret[ 'skel' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imconcat( left_skel, right_skel, 125 ), cmap='gray' )
        plt.title( 'Skeletonized threshold' )

        figs_ret[ 'centerline' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imconcat( left_rect_draw, right_rect_draw, [ 0, 0, 255 ] )[ :, :, ::-1 ] )
        plt.title( 'centerline points' )

        figs_ret[ 'centerline-matches' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imconcat( left_rect_match_draw, right_rect_match_draw, [ 0, 0, 255 ] )[ :, :, ::-1 ] )
        plt.title( 'matched centerline points' )

        figs_ret[ 'disparity' ] = plt.figure( figsize=(8, 8) )
        plt.plot( pts_l_match[ :, 0 ] - pts_r_match[ :, 0 ] )
        plt.title( 'disparity' )

        figs_ret[ '3d' ] = plt.figure( figsize=(8, 8) )
        _, ax = plot3D_equal( pts_3d[ :, :3 ], figs_ret[ '3d' ], 1 )
        plt.title( '3-D reconstruction' )

        plt.show()

    # if

    return pts_3d, pts_l, pts_r, bspline_l, bspline_r, imgs_ret, figs_ret


# needle_jig_reconstruction_refined


def needle_reconstruction_ref( left_img, left_ref, right_img, right_ref, stereo_params,
                               bor_l=None, bor_r=None,
                               roi_l=(), roi_r=(),
                               alpha: float = 0.5, recalc_stereo: bool = True,
                               zoom: float = 1.0, winsize: tuple = (31, 31),
                               sub_thresh: int = 55, proc_show: bool = False ):
    """ This is a method to perform needle reconstruction on an image
        using a reference image prior to insertion for image subtraction

        @param winsize (tuple): 2-tuple of (y,x) window size (top-bottom, left-right)

    """
    # prepare images
    if bor_l is None:
        bor_l = [ ]
    if bor_r is None:
        bor_r = [ ]
    imgs_ret = { }

    # grayscale the images
    left_gray = cv.cvtColor( left_img, cv.COLOR_BGR2GRAY )
    right_gray = cv.cvtColor( right_img, cv.COLOR_BGR2GRAY )
    left_ref_gray = cv.cvtColor( left_ref, cv.COLOR_BGR2GRAY )
    right_ref_gray = cv.cvtColor( right_ref, cv.COLOR_BGR2GRAY )


    # segment each image
    left_seg_init, left_sub = segment_needle_subtract( left_gray, left_ref_gray, threshold=sub_thresh )
    right_seg_init, right_sub = segment_needle_subtract( right_gray, right_ref_gray, threshold=sub_thresh )
    imgs_ret[ 'sub' ] = imconcat( left_sub, right_sub, 125 )
    imgs_ret[ 'seg-init' ] = imconcat( 255 * left_seg_init, 255 * right_seg_init, 125 )

    

    left_seg_init_boroi = roi( blackout_regions( left_seg_init, bor_l ), roi_l, full=True )
    right_seg_init_boroi = roi( blackout_regions( right_seg_init, bor_r ), roi_r, full=True )


    left_seg_init_boroi, right_seg_init_boroi = bin_close( left_seg_init_boroi, right_seg_init_boroi, ksize=(9, 5) )

    # - perform connected component analysis
    num_cc_keep = 1
    left_seg = connected_component_filtering( left_seg_init_boroi, N_keep=num_cc_keep ).astype( np.uint8 )
    right_seg = connected_component_filtering( right_seg_init_boroi, N_keep=num_cc_keep ).astype( np.uint8 )
    imgs_ret[ 'seg' ] = imconcat( 255 * left_seg, 255 * right_seg, 125 )

    # - roi the images
    left_roi = roi( left_img, roi_l, full=True )
    right_roi = roi( right_img, roi_r, full=True )

    cv.imwrite('check_roi-seg.png', imgs_ret[ 'seg' ])

    # - black-out regions
    left_roibo = blackout_regions( left_roi, bor_l )
    right_roibo = blackout_regions( right_roi, bor_r )
    imgs_ret[ 'roibo' ] = imconcat( left_roibo, right_roibo, 125 )

    # stereo rectify the images
    left_rect, right_rect, _, map_l, map_r = stereo_rectify( left_roibo, right_roibo, stereo_params,
                                                             interp_method=cv.INTER_LINEAR, alpha=alpha,
                                                             recalc_stereo=recalc_stereo )
    imgs_ret[ 'rect-roi' ] = imconcat( left_rect, right_rect, [ 0, 0, 255 ] )

    # = apply stereo rectification map to full image
    left_rect_full = cv.remap( left_img, map_l[ 0 ], map_l[ 1 ], cv.INTER_LINEAR )
    right_rect_full = cv.remap( right_img, map_r[ 0 ], map_r[ 1 ], cv.INTER_LINEAR )
    imgs_ret[ 'rect' ] = imconcat( left_rect_full, right_rect_full, [ 0, 0, 255 ] )

    # - apply stereo rectification map to segmented images
    left_seg_rect = cv.remap( left_seg, map_l[ 0 ], map_l[ 1 ], cv.INTER_NEAREST )
    right_seg_rect = cv.remap( right_seg, map_r[ 0 ], map_r[ 1 ], cv.INTER_NEAREST )
    imgs_ret[ 'rect-seg' ] = imconcat( 255 * left_seg_rect, 255 * right_seg_rect, 125 )

    # - map the rois
    boroi_l_img = roi_mask( roi_l, left_img.shape ) & blackout_image( bor_l, left_img.shape )
    boroi_r_img = roi_mask( roi_r, right_img.shape ) & blackout_image( bor_r, right_img.shape )
    imgs_ret[ 'roi-bo-bool' ] = imconcat( 255 * boroi_l_img, 255 * boroi_r_img, 125 )

    boroi_l_mapped = cv.remap( boroi_l_img.astype( np.uint8 ), map_l[ 0 ], map_l[ 1 ], cv.INTER_NEAREST )
    boroi_r_mapped = cv.remap( boroi_r_img.astype( np.uint8 ), map_r[ 0 ], map_r[ 1 ], cv.INTER_NEAREST )
    imgs_ret[ 'boroi-bool-mapped' ] = imconcat( 255 * boroi_l_mapped, 255 * boroi_r_mapped, 125 )

    # = remove extra borders threshed out
    left_seg_rect *= boroi_l_mapped.astype( np.uint8 )
    right_seg_rect *= boroi_r_mapped.astype( np.uint8 )
    imgs_ret[ 'seg-rect-boroi' ] = imconcat( 255 * left_seg_rect, 255 * right_seg_rect, 125 )
    cv.imwrite('check_se-rect-boroi.png', imgs_ret[ 'seg-rect-boroi' ])

    # get the image contours
    left_skel, right_skel = skeleton( left_seg_rect, right_seg_rect )

    imgs_ret[ 'skel' ] = imconcat( 255 * left_skel, 255 * right_skel, 125 ).astype( np.uint8 )
    conts_l, conts_r = contours( left_skel, right_skel )
    cv.imwrite('check_skel.png', imgs_ret[ 'skel' ])

    # - outlier options
    len_thresh = 5
    bspl_k = 0
    out_thresh = -1  # don't do outlier detection
    n_neigh = 20
    scale = (1, 1)

    pts_l, bspline_l = centerline_from_contours( conts_l,
                                                 len_thresh=len_thresh,
                                                 bspline_k=bspl_k,
                                                 outlier_thresh=out_thresh,
                                                 num_neighbors=n_neigh, scale=scale )

    pts_r, bspline_r = centerline_from_contours( conts_r,
                                                 len_thresh=len_thresh,
                                                 bspline_k=bspl_k,
                                                 outlier_thresh=out_thresh,
                                                 num_neighbors=n_neigh, scale=scale )

    left_rect_draw = cv.polylines( left_rect_full.copy(), [ pts_l.reshape( -1, 1, 2 ).astype( np.int32 ) ], False,
                                   (255, 0, 0), 3 )
    right_rect_draw = cv.polylines( right_rect_full.copy(), [ pts_r.reshape( -1, 1, 2 ).astype( np.int32 ) ], False,
                                    (255, 0, 0), 3 )
    imgs_ret[ 'contours' ] = imconcat( left_rect_draw, right_rect_draw, [ 0, 0, 255 ] )
    cv.imwrite('check_contours.png', imgs_ret[ 'contours' ])
    # stereo matching
    left_rect_gray = cv.cvtColor( left_rect_full, cv.COLOR_BGR2GRAY )
    right_rect_gray = cv.cvtColor( right_rect_full, cv.COLOR_BGR2GRAY )
    al_l = np.linalg.norm( np.diff( pts_l, axis=0 ), axis=1 ).sum()
    al_r = np.linalg.norm( np.diff( pts_r, axis=0 ), axis=1 ).sum()
    if al_l >= al_r:  # take the max arclengths
        pts_l_match, pts_r_match = stereomatch_normxcorr( pts_l, pts_r,
                                                          left_rect_gray, right_rect_gray,
                                                          winsize=winsize, zoom=zoom,
                                                          score_thresh=0.5 )  # left search right
    # if

    else:
        pts_r_match, pts_l_match = stereomatch_normxcorr( pts_r, pts_l,
                                                          right_rect_gray, left_rect_gray,
                                                          winsize=winsize, zoom=zoom,
                                                          score_thresh=0.5 )  # right search left
    # else

    idx_l = np.argsort( pts_l_match[ :, 1 ] )
    pts_l_match = pts_l_match[ idx_l ]
    pts_r_match = pts_r_match[ idx_l ]

    # bspline fit the matching points
    if bspl_k > 0 or True:
        bspline_l_match = BSpline1D( pts_l_match[ :, 1 ], pts_l_match[ :, 0 ], k=2 )
        bspline_r_match = BSpline1D( pts_r_match[ :, 1 ], pts_r_match[ :, 0 ], k=2 )
        s = np.linspace( 0, 1, 200 )
        pts_l_match = np.vstack( (bspline_l_match.eval_unscale( s ), bspline_l_match.unscale( s )) ).T  # (j, i)
        pts_r_match = np.vstack( (bspline_r_match.eval_unscale( s ), bspline_r_match.unscale( s )) ).T  # (j, i)

    # if

    # = add to images
    left_rect_match_draw = cv.polylines( left_rect_full.copy(),
                                         [ pts_l_match.reshape( -1, 1, 2 ).round().astype( np.int32 ) ], False,
                                         (255, 0, 0), 3 )
    right_rect_match_draw = cv.polylines( right_rect_full.copy(),
                                          [ pts_r_match.reshape( -1, 1, 2 ).round().astype( np.int32 ) ], False,
                                          (255, 0, 0), 3 )
    imgs_ret[ 'contours-match' ] = imconcat( left_rect_match_draw, right_rect_match_draw, [ 0, 0, 255 ] )
    cv.imwrite('check_contours-match.png', imgs_ret[ 'contours-match' ])
    # stereo
    pts_3d = cv.triangulatePoints( stereo_params[ 'P1' ], stereo_params[ 'P2' ], pts_l_match.T, pts_r_match.T )
    pts_3d /= pts_3d[ -1, : ]
    pts_3d = pts_3d.T

    # show processing
    figs_ret = { }
    if proc_show:
        figs_ret[ 'sub' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imgs_ret[ 'sub' ], cmap='gray' )
        plt.title( "Image Subtraction" )

        figs_ret[ 'seg-init' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imgs_ret[ 'seg-init' ], cmap='gray' )
        plt.title( 'Segmented needle (Initial)' )

        figs_ret[ 'seg' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imgs_ret[ 'seg' ], cmap='gray' )
        plt.title( 'Segmented needle (Post CC Analysis)' )

        figs_ret[ 'rect' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imgs_ret[ 'rect' ][ :, :, ::-1 ] )
        plt.title( 'Rectified Images' )

        figs_ret[ 'rect-roi' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imconcat( left_rect, right_rect, [ 0, 0, 255 ] )[ :, :, ::-1 ] )
        plt.title( 'Rectified Images: ROI' )

        figs_ret[ 'rect-seg' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imgs_ret[ 'rect-seg' ], cmap='gray' )
        plt.title( 'Segmented rectified images' )

        figs_ret[ 'skel' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imconcat( left_skel, right_skel, 125 ), cmap='gray' )
        plt.title( 'Skeletonized threshold' )

        figs_ret[ 'centerline' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imconcat( left_rect_draw, right_rect_draw, [ 0, 0, 255 ] )[ :, :, ::-1 ] )
        plt.title( 'centerline points' )

        figs_ret[ 'centerline-plot' ] = plt.figure( figsize=(12, 8) )
        plt.plot( pts_l[ :, 0 ], pts_l[ :, 1 ], '.', label='left' )
        plt.plot( pts_r[ :, 0 ], pts_r[ :, 1 ], '.', label='right' )
        plt.gca().invert_yaxis()
        plt.title( 'centerline points' )
        plt.legend()

        figs_ret[ 'centerline-matches' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imconcat( left_rect_match_draw, right_rect_match_draw, [ 0, 0, 255 ] )[ :, :, ::-1 ] )
        plt.title( 'matched centerline points' )

        figs_ret[ 'centerline-matches-plot' ] = plt.figure( figsize=(12, 8) )
        plt.plot( pts_l_match[ :, 0 ], pts_l_match[ :, 1 ], '.', label='left' )
        plt.plot( pts_r_match[ :, 0 ], pts_r_match[ :, 1 ], '.', label='right' )
        plt.gca().invert_yaxis()
        plt.title( 'matched centerline points' )
        plt.legend()

        figs_ret[ 'disparity' ] = plt.figure( figsize=(8, 8) )
        plt.plot( pts_l_match[ :, 0 ] - pts_r_match[ :, 0 ] )
        plt.title( 'disparity' )

        figs_ret[ '3d' ] = plt.figure( figsize=(8, 8) )
        _, ax = plot3D_equal( pts_3d[ :, :3 ], figs_ret[ '3d' ], 1 )
        ax.plot( pts_3d[ -1, 0 ], pts_3d[ -1, 1 ], pts_3d[ -1, 2 ], 'g*' )
        plt.title( '3-D reconstruction' )

        # plt.show()

    # if

    return pts_3d, pts_l, pts_r, bspline_l, bspline_r, imgs_ret, figs_ret


# needle_reconstruction_ref


def needle_tissue_reconstruction_refined( img_left, img_right, stereo_params,
                                          bor_l=None, bor_r=None,
                                          roi_l=(), roi_r=(),
                                          alpha: float = 0.5, recalc_stereo: bool = True,
                                          zoom: float = 1.0, winsize: tuple = (31, 21),
                                          proc_show: bool = False ):
    """ This is function for performing a 3-D needle reconstruction from a raw stereo pair"""
    # prepare images
    if bor_l is None:
        bor_l = [ ]
    if bor_r is None:
        bor_r = [ ]
    imgs_ret = { }

    # = gaussian blur the images
    left_gauss, right_gauss = gauss_blur( img_left, img_right, ksize=(5, 5) )
    imgs_ret[ 'gauss' ] = imconcat( left_gauss, right_gauss, 125 )

    # # roi the images
    left_roi = roi( left_gauss, roi_l, full=True )
    right_roi = roi( right_gauss, roi_r, full=True )

    # # black-out regions
    left_roibo = blackout_regions( left_roi, bor_l )
    right_roibo = blackout_regions( right_roi, bor_r )
    imgs_ret[ 'roibo' ] = imconcat( left_roibo, right_roibo, 125 )

    # stereo rectify the images
    left_rect, right_rect, _, map_l, map_r = stereo_rectify( left_roibo, right_roibo, stereo_params,
                                                             interp_method=cv.INTER_LINEAR, alpha=alpha,
                                                             recalc_stereo=recalc_stereo )
    imgs_ret[ 'rect-roi' ] = imconcat( left_rect, right_rect, [ 0, 0, 255 ] )

    # = apply stereo rectification map to full image
    left_rect_full = cv.remap( img_left, map_l[ 0 ], map_l[ 1 ], cv.INTER_LINEAR )
    right_rect_full = cv.remap( img_right, map_r[ 0 ], map_r[ 1 ], cv.INTER_LINEAR )
    imgs_ret[ 'rect' ] = imconcat( left_rect_full, right_rect_full, [ 0, 0, 255 ] )

    # # map the rois
    boroi_l_img = roi_mask( roi_l, img_left.shape ) & blackout_image( bor_l, img_left.shape )
    boroi_r_img = roi_mask( roi_r, img_right.shape ) & blackout_image( bor_r, img_right.shape )
    imgs_ret[ 'roi-bo-bool' ] = imconcat( 255 * boroi_l_img, 255 * boroi_r_img, 125 )

    boroi_l_mapped = cv.remap( boroi_l_img.astype( np.uint8 ), map_l[ 0 ], map_l[ 1 ], cv.INTER_NEAREST )
    boroi_r_mapped = cv.remap( boroi_r_img.astype( np.uint8 ), map_r[ 0 ], map_r[ 1 ], cv.INTER_NEAREST )
    imgs_ret[ 'boroi-bool-mapped' ] = imconcat( 255 * boroi_l_mapped, 255 * boroi_r_mapped, 125 )

    # perform image processing on rectified images
    left_gray = cv.cvtColor( left_rect, cv.COLOR_BGR2GRAY )
    right_gray = cv.cvtColor( right_rect, cv.COLOR_BGR2GRAY )
    imgs_ret[ 'rect-gray' ] = imconcat( left_gray, right_gray )
    left_thresh, right_thresh = thresh( left_gray, right_gray, thresh=65 )
    left_thresh_low, right_thresh_low = thresh( left_gray, right_gray, thresh=12 )  # remove black colors
    left_thresh *= (left_thresh_low == 0)
    right_thresh *= (right_thresh_low == 0)

    # = remove extra borders threshed out
    left_thresh *= boroi_l_mapped.astype( np.uint8 )
    right_thresh *= boroi_r_mapped.astype( np.uint8 )
    imgs_ret[ 'thresh-rect' ] = imconcat( left_thresh, right_thresh, 125 )

    # median filter
    left_med, right_med = median_blur( left_thresh, right_thresh, ksize=3 )
    imgs_ret[ 'median' ] = imconcat( left_med, right_med, 125 )

    # segment out the red color
    redmask_l, redmask_r, left_red, right_red = color_segmentation( img_left, img_right, 'red' )
    left_thresh_masked = np.logical_not( redmask_l ).astype( np.uint8 ) * left_med
    right_thresh_masked = np.logical_not( redmask_r ).astype( np.uint8 ) * right_med
    imgs_ret[ 'mask-red' ] = imconcat( left_red, right_red, [ 125, 125, 125 ] )
    imgs_ret[ 'thresh-no-red' ] = imconcat( left_thresh_masked, right_thresh_masked, 125 )

    #     # opening operation
    #     left_open, right_open = bin_open( left_thresh_masked, right_thresh_masked, ( 6, 4 ) )
    #     imgs_ret['open'] = imconcat( left_open, right_open, 125 )

    # get the contours and filter out outliers
    left_skel, right_skel = skeleton( left_med, right_med )
    imgs_ret[ 'skel' ] = imconcat( 255 * left_skel, 255 * right_skel, 125 ).astype( np.uint8 )
    conts_l, conts_r = contours( left_skel, right_skel )

    # - outlier options
    len_thresh = 5
    bspl_k = 2
    out_thresh = 0.5
    n_neigh = 50
    scale = (5, 1 / 10)

    pts_l, bspline_l = centerline_from_contours( conts_l,
                                                 len_thresh=len_thresh,
                                                 bspline_k=bspl_k,
                                                 outlier_thresh=out_thresh,
                                                 num_neighbors=n_neigh, scale=scale )

    pts_r, bspline_r = centerline_from_contours( conts_r,
                                                 len_thresh=len_thresh,
                                                 bspline_k=bspl_k,
                                                 outlier_thresh=out_thresh,
                                                 num_neighbors=n_neigh, scale=scale )

    left_rect_draw = cv.polylines( left_rect_full.copy(), [ pts_l.reshape( -1, 1, 2 ).astype( np.int32 ) ], False,
                                   (255, 0, 0), 5 )
    right_rect_draw = cv.polylines( right_rect_full.copy(), [ pts_r.reshape( -1, 1, 2 ).astype( np.int32 ) ], False,
                                    (255, 0, 0), 5 )
    imgs_ret[ 'contours' ] = imconcat( left_rect_draw, right_rect_draw, [ 0, 0, 255 ] )

    # stereo matching
    left_rect_gray = cv.cvtColor( left_rect_full, cv.COLOR_BGR2GRAY )
    right_rect_gray = cv.cvtColor( right_rect_full, cv.COLOR_BGR2GRAY )
    al_l = np.linalg.norm( np.diff( pts_l, axis=0 ), axis=1 ).sum()
    al_r = np.linalg.norm( np.diff( pts_r, axis=0 ), axis=1 ).sum()
    if al_l >= al_r and False:
        pts_l_match, pts_r_match = stereomatch_normxcorr( pts_l, pts_r,
                                                          left_rect_gray.copy(), right_rect_gray.copy(),
                                                          winsize=winsize, zoom=zoom,
                                                          score_thresh=0.5 )  # left search right
    # if

    else:
        pts_r_match, pts_l_match = stereomatch_normxcorr( pts_r, pts_l,
                                                          right_rect_gray.copy(), left_rect_gray.copy(),
                                                          winsize=winsize, zoom=zoom,
                                                          score_thresh=0.5 )  # right search left
    # else

    idx_l = np.argsort( pts_l_match[ :, 1 ] )
    pts_l_match = pts_l_match[ idx_l ]
    pts_r_match = pts_r_match[ idx_l ]

    # bspline fit the matching points
    if bspl_k > 0:
        bspline_l_match = BSpline1D( pts_l_match[ :, 1 ], pts_l_match[ :, 0 ], k=2 )
        bspline_r_match = BSpline1D( pts_r_match[ :, 1 ], pts_r_match[ :, 0 ], k=2 )
        s = np.linspace( 0, 1, 200 )
        pts_l_match = np.vstack( (bspline_l_match.eval_unscale( s ), bspline_l_match.unscale( s )) ).T  # (j, i)
        pts_r_match = np.vstack( (bspline_r_match.eval_unscale( s ), bspline_r_match.unscale( s )) ).T  # (j, i)

    # if

    # = add to images
    left_rect_match_draw = cv.polylines( left_rect_full.copy(),
                                         [ pts_l_match.reshape( -1, 1, 2 ).round().astype( np.int32 ) ], False,
                                         (255, 0, 0), 5 )
    right_rect_match_draw = cv.polylines( right_rect_full.copy(),
                                          [ pts_r_match.reshape( -1, 1, 2 ).round().astype( np.int32 ) ], False,
                                          (255, 0, 0), 5 )
    imgs_ret[ 'contours-match' ] = imconcat( left_rect_match_draw, right_rect_match_draw, [ 0, 0, 255 ] )

    # stereo
    #     pts_l_match = np.flip( pts_l_match, axis = 1 ) # align so (x, y) = (j, i)
    #     pts_r_match = np.flip( pts_r_match, axis = 1 ) # align so (x, y) = (j, i)
    pts_3d = cv.triangulatePoints( stereo_params[ 'P1' ], stereo_params[ 'P2' ], pts_l_match.T, pts_r_match.T )
    pts_3d /= pts_3d[ -1, : ]
    pts_3d = pts_3d.T

    # show processing
    figs_ret = { }
    if proc_show:
        figs_ret[ 'rect' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imgs_ret[ 'rect' ][ :, :, ::-1 ] )
        plt.title( 'Rectified Images' )

        figs_ret[ 'rect-roi' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imconcat( left_rect, right_rect, [ 0, 0, 255 ] )[ :, :, ::-1 ] )
        plt.title( 'Rectified Images: ROI' )

        figs_ret[ 'thresh-rect' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imconcat( left_thresh, right_thresh, 125 ), cmap='gray' )
        plt.title( 'Thresholded rectified images' )

        figs_ret[ 'median' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imconcat( left_med, right_med, 125 ), cmap='gray' )
        plt.title( 'Median-Filtered Images' )

        #         figs_ret['mask-red'] = plt.figure( figsize = ( 12, 8 ) )
        #         plt.imshow( imgs_ret['mask-red'][:,:,::-1] )
        #         plt.title( 'Red mask' )

        #         figs_ret['thresh-no-red'] = plt.figure( figsize = ( 12, 8 ) )
        #         plt.imshow( imgs_ret['thresh-no-red'], cmap = 'gray' )
        #         plt.title( 'Threshold masked out-red' )

        #         figs_ret['open'] = plt.figure( figsize = ( 12, 8 ) )
        #         plt.imshow( imconcat( left_open, right_open, 125 ), cmap = 'gray' )
        #         plt.title( 'Morphological opening' )

        figs_ret[ 'skel' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imconcat( left_skel, right_skel, 125 ), cmap='gray' )
        plt.title( 'Skeletonized threshold' )

        figs_ret[ 'centerline' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imconcat( left_rect_draw, right_rect_draw, [ 0, 0, 255 ] )[ :, :, ::-1 ] )
        plt.title( 'centerline points' )

        figs_ret[ 'centerline-plot' ] = plt.figure( figsize=(12, 8) )
        plt.plot( pts_l[ :, 0 ], pts_l[ :, 1 ], '.', label='left' )
        plt.plot( pts_r[ :, 0 ], pts_r[ :, 1 ], '.', label='right' )
        plt.gca().invert_yaxis()
        plt.title( 'centerline points' )
        plt.legend()

        figs_ret[ 'centerline-matches' ] = plt.figure( figsize=(12, 8) )
        plt.imshow( imconcat( left_rect_match_draw, right_rect_match_draw, [ 0, 0, 255 ] )[ :, :, ::-1 ] )
        plt.title( 'matched centerline points' )

        figs_ret[ 'centerline-matches-plot' ] = plt.figure( figsize=(12, 8) )
        plt.plot( pts_l_match[ :, 0 ], pts_l_match[ :, 1 ], '.', label='left' )
        plt.plot( pts_r_match[ :, 0 ], pts_r_match[ :, 1 ], '.', label='right' )
        plt.gca().invert_yaxis()
        plt.title( 'matched centerline points' )
        plt.legend()

        figs_ret[ 'disparity' ] = plt.figure( figsize=(8, 8) )
        plt.plot( pts_l_match[ :, 0 ] - pts_r_match[ :, 0 ] )
        plt.title( 'disparity' )

        figs_ret[ '3d' ] = plt.figure( figsize=(8, 8) )
        _, ax = plot3D_equal( pts_3d[ :, :3 ], figs_ret[ '3d' ], 1 )
        plt.title( '3-D reconstruction' )

        plt.show()

    # if

    return pts_3d, pts_l, pts_r, bspline_l, bspline_r, imgs_ret, figs_ret


# needle_tissue_reconstruction_refined


def needleproc_stereo( left_img, right_img,
                       bor_l=None, bor_r=None,
                       roi_l: tuple = (), roi_r: tuple = (),
                       proc_show: bool = False ):
    """ wrapper function to process the left and right image pair for needle
        centerline identification

     """

    # convert to grayscale if not already
    if bor_l is None:
        bor_l = [ ]
    if bor_r is None:
        bor_r = [ ]
    if left_img.ndim > 2:
        left_img = cv.cvtColor( left_img, cv.COLOR_BGR2GRAY )
        right_img = cv.cvtColor( right_img, cv.COLOR_BGR2GRAY )

    # if

    # start the image qprocessing
    left_thresh, right_thresh = thresh( left_img, right_img )

    left_roi = roi( left_thresh, roi_l, full=True )
    right_roi = roi( right_thresh, roi_r, full=True )

    left_thresh_bo = blackout_regions( left_roi, bor_l )
    right_thresh_bo = blackout_regions( right_roi, bor_r )

    left_tmed, right_tmed = median_blur( left_thresh_bo, right_thresh_bo, ksize=5 )

    left_open, right_open = bin_open( left_tmed, right_tmed, ksize=(5, 5) )

    left_close, right_close = bin_close( left_open, right_open, ksize=(7, 7) )

    left_dil, right_dil = bin_dilate( left_close, right_close, ksize=(0, 0) )

    left_skel, right_skel = skeleton( left_dil, right_dil )

    # get the contours ( sorted by length)
    conts_l, conts_r = contours( left_skel, right_skel )

    if proc_show:
        plt.ion()

        plt.figure()
        plt.imshow( imconcat( left_thresh, right_thresh, 150 ), cmap='gray' )
        plt.title( 'adaptive thresholding' )

        plt.figure()
        plt.imshow( imconcat( left_roi, right_roi, 150 ), cmap='gray' )
        plt.title( 'roi: after thresholding' )

        plt.figure()
        plt.imshow( imconcat( left_thresh_bo, right_thresh_bo, 150 ), cmap='gray' )
        plt.title( 'region suppression: roi' )

        plt.figure()
        plt.imshow( imconcat( left_tmed, right_tmed, 150 ), cmap='gray' )
        plt.title( 'median filtering: after region suppression' )

        plt.figure()
        plt.imshow( imconcat( left_close, right_close, 150 ), cmap='gray' )
        plt.title( 'opening: after median' )

        plt.figure()
        plt.imshow( imconcat( left_open, right_open, 150 ), cmap='gray' )
        plt.title( 'closing: after opening' )

        plt.figure()
        plt.imshow( imconcat( left_open, right_open, 150 ), cmap='gray' )
        plt.title( 'dilation: after closing' )

        plt.figure()
        plt.imshow( imconcat( left_skel, right_skel, 150 ), cmap='gray' )
        plt.title( 'skeletization: after dilation' )

        cont_left = left_img.copy().astype( np.uint8 )
        cont_right = right_img.copy().astype( np.uint8 )

        cont_left = cv.cvtColor( cont_left, cv.COLOR_GRAY2RGB )
        cont_right = cv.cvtColor( cont_right, cv.COLOR_GRAY2RGB )

        cv.drawContours( cont_left, conts_l, 0, (255, 0, 0), 3 )
        cv.drawContours( cont_right, conts_r, 0, (255, 0, 0), 3 )

        plt.figure()
        plt.imshow( imconcat( cont_left, cont_right, 150 ) )
        plt.title( 'contour: the longest 1' )

        plt.show()
        while True:
            if plt.waitforbuttonpress( 0 ):
                break

        # while
        plt.close( 'all' )

    # if

    return left_skel, right_skel, conts_l, conts_r


# needleproc_stereo


def plot3D_equal( pts, fig=None, axis: int = 0 ):
    if axis == 1:
        pts = pts.T

    # if

    if fig is None:
        fig = plt.figure()

    # if

    ax = fig.add_subplot( projection='3d' )
    X, Y, Z = pts[ :3 ]
    ax.plot( X, Y, Z )
    max_range = np.max( [ X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min() ] )
    Xb = 0.5 * max_range * np.mgrid[ -1:2:2, -1:2:2, -1:2:2 ][ 0 ].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[ -1:2:2, -1:2:2, -1:2:2 ][ 1 ].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[ -1:2:2, -1:2:2, -1:2:2 ][ 2 ].flatten() + 0.5 * (Z.max() + Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip( Xb, Yb, Zb ):
        ax.plot( [ xb ], [ yb ], [ zb ], 'w' )

    # for

    return fig, ax


# plot3D_equal


def roi( img, roi_bnds, full: bool = True ):
    """ return region of interest

        @param img: image to be ROI'd
        @param roi_bnds: [tuple of top-left point, tuple of bottom-right point]
        @param full: boolean of whether to return only ROI or the entire image size

        @return: subimage of the within the roi
    """

    if len( roi_bnds ) == 0:
        return img

    # if

    tl_i, tl_j = roi_bnds[ 0 ]
    br_i, br_j = roi_bnds[ 1 ]

    if full:
        img_roi = img.copy()

        # zero-out value
        zval = 0 if img.ndim == 2 else np.array( [ 0, 0, 0 ] )

        # zero out values
        img_roi[ :tl_i, : ] = zval
        img_roi[ br_i:, : ] = zval

        img_roi[ :, :tl_j ] = zval
        img_roi[ :, br_j: ] = zval

    # if

    else:
        img_roi = img[ tl_i:br_i, tl_j:br_j ].copy()

    # else

    return img_roi


# roi


def roi_mask( reg_of_int, image_size ):
    roi_bool = np.ones( image_size[ :2 ], dtype=bool )

    roi_bool = roi( roi_bool, reg_of_int ).astype( bool )

    return roi_bool


# roi_image


def segment_needle_subtract( img, ref_img, threshold ):
    """ This is a method to segment the needle by subtracting out the background of an image

        (Note: threshold of 55 seems to work OK)

        @param img: gray scale image for the current insertion
        @param ref_img: gray scale image as the reference prior to insertion
        @param threshold: the subtraction cutoff

        @return: segmented needle, the highlighted subtraction image

    """
    diff_img = ref_img.astype( int ) - img.astype(
            int )  # needle is darker therefore, diff > 0 is what we want to highlight

    hl_img = diff_img * (diff_img > 0)  # highlight only positive changes

    bin_img = (hl_img >= threshold).astype( np.uint8 )

    # morphological operations
    bin_img = cv.morphologyEx( bin_img, cv.MORPH_OPEN, np.ones( (3, 3) ) )
    bin_img = cv.morphologyEx( bin_img, cv.MORPH_CLOSE, np.ones( (3, 3) ) )

    return bin_img, hl_img


# segment_needle_subtract


def skeleton( left_bin, right_bin ):
    """ skeletonize the left and right binary images"""

    left_bin = (left_bin > 0).astype( np.uint8 )
    right_bin = (right_bin > 0).astype( np.uint8 )

    left_skel = skeletonize( left_bin )
    right_skel = skeletonize( right_bin )
    #     left_skel = thin( left_bin, max_iter = 2 )
    #     right_skel = thin( right_bin, max_iter = 2 )

    return left_skel, right_skel


# skeleton


def stereo_disparity( left_gray, right_gray, stereo_params: dict ):
    """ stereo distparity mapping """
    # parameters
    win_size = 5

    left_gauss, right_gauss = gauss_blur( left_gray, right_gray, ksize=(1, 1) )
    stereo = cv.StereoSGBM_create( numDisparities=64,
                                   blockSize=win_size,
                                   speckleRange=2, speckleWindowSize=5,
                                   P1=8 * 3 * win_size ** 2,
                                   P2=20 * 3 * win_size ** 2 )
    disparity = stereo.compute( left_gauss, right_gauss )

    return disparity


# stereo_disparity


def stereo_rectify( img_left, img_right, stereo_params, interp_method=cv.INTER_LINEAR, alpha: float = -1,
                    recalc_stereo: bool = True ):
    # gray-scale the image
    left_gray = cv.cvtColor( img_left, cv.COLOR_BGR2GRAY )
    right_gray = cv.cvtColor( img_right, cv.COLOR_BGR2GRAY )

    # perform stereo rectification
    K_l = stereo_params[ 'cameraMatrix1' ].astype( float )
    K_r = stereo_params[ 'cameraMatrix2' ].astype( float )
    dists_l = stereo_params[ 'distCoeffs1' ].astype( float )
    dists_r = stereo_params[ 'distCoeffs2' ].astype( float )
    R = stereo_params[ 'R' ].astype( float )
    t = stereo_params[ 't' ].astype( float )

    if recalc_stereo or any( k not in stereo_params.keys() for k in [ 'R1', 'R2', 'P1', 'P2' ] ):
        R_l, R_r, P_l, P_r, Q, roi_l, roi_r = cv.stereoRectify( K_l, dists_l, K_r, dists_r, left_gray.shape[ ::-1 ],
                                                                R, t, alpha=alpha )
        stereo_params[ 'R1' ] = R_l
        stereo_params[ 'R2' ] = R_r
        stereo_params[ 'P1' ] = P_l
        stereo_params[ 'P2' ] = P_r
        stereo_params[ 'Q' ] = Q
        rois_lr = (roi_l, roi_r)

    # if

    else:
        R_l = stereo_params[ 'R1' ]
        R_r = stereo_params[ 'R2' ]
        P_l = stereo_params[ 'P1' ]
        P_r = stereo_params[ 'P2' ]
        rois_lr = tuple( 2 * [ [ [ 0, 0 ], [ -1, 1 ] ] ] )

    # else

    # compute stereo rectification map
    map1_l, map2_l = cv.initUndistortRectifyMap( K_l, dists_l, R_l, P_l, left_gray.shape[ ::-1 ], cv.CV_32FC1 )
    map1_r, map2_r = cv.initUndistortRectifyMap( K_r, dists_r, R_r, P_r, right_gray.shape[ ::-1 ], cv.CV_32FC1 )

    # apply stereo rectification map
    left_rect = cv.remap( img_left, map1_l, map2_l, interp_method )
    right_rect = cv.remap( img_right, map1_r, map2_r, interp_method )

    return left_rect, right_rect, rois_lr, (map1_l, map2_l), (map1_r, map2_r)


# stereo_rectify


def stereomatch_needle( left_conts, right_conts, method="tip-count", col: int = 1,
                        bspline_l: BSpline1D = None, bspline_r: BSpline1D = None ):
    """ stereo matching needle arclength points for the needle


        Args:
            (left/right)_conts: a nx2 array of pixel coordinates
                                for the contours in the (left/right) image

            method (Default: "tip"): method to use for stereomatching.

            col (int = 1): the column to begin matching by

     """
    # squeeze dimensions just in case
    left_conts = np.squeeze( left_conts )
    right_conts = np.squeeze( right_conts )

    # remove duplicate rows
    pts_l = np.unique( left_conts, axis=0 )
    pts_r = np.unique( right_conts, axis=0 )

    if method == "tip-count":
        if bspline_l is None or bspline_r is None:
            n = min( pts_l.shape[ 0 ], pts_r.shape[ 0 ] )  # min number of points to match
            left_idx = np.argsort( pts_l[ :, col ] )[ -n: ]
            right_idx = np.argsort( pts_r[ :, col ] )[ -n: ]

            left_matches = pts_l[ left_idx ]
            right_matches = pts_r[ right_idx ]

        # if

        else:
            warnings.filterwarnings( 'ignore', category=UserWarning, module='BSpline1D' )
            # determine each of the bspline arclengths
            s = np.linspace( 0, 1, 200 )
            pts_lx = bspline_l.unscale( s )
            pts_ly = bspline_l( s )

            pts_rx = bspline_r.unscale( s )
            pts_ry = bspline_r( s )
            if col == 0:
                pts_l = np.vstack( (pts_lx, pts_ly) ).T
                pts_r = np.vstack( (pts_rx, pts_ry) ).T

            # if

            else:
                pts_l = np.vstack( (pts_lx, pts_ly) ).T
                pts_r = np.vstack( (pts_rx, pts_ry) ).T

            # else

            left_matches = pts_l
            right_matches = pts_r

        # else
    # if

    elif method == 'disparity':
        if bspline_l is None or bspline_r is None:
            # match from tip (bottom most point | largest number)
            uniq_ax = np.unique( np.append( pts_l[ :, col ], pts_r[ :, col ] ) )  # unique values in the columns

            left_matches = np.zeros( (0, 2) )
            right_matches = np.zeros( (0, 2) )
            for val in uniq_ax:
                # find the rows that have this value
                mask_val_l = pts_l[ :, col ] == val
                mask_val_r = pts_l[ :, col ] == val

                # if the both have matches, add the means to the matches
                if np.any( mask_val_l ) and np.any( mask_val_r ):
                    match_l = np.mean( pts_l[ mask_val_l, : ], axis=0, keepdims=True )
                    match_r = np.mean( pts_r[ mask_val_r, : ], axis=0, keepdims=True )

                    left_matches = np.append( left_matches, match_l, axis=0 )
                    right_matches = np.append( right_matches, match_r, axis=0 )

            # for
        # if

        else:
            # determine bounds on y-axis
            y_min = max( bspline_l.qmin, bspline_r.qmin )
            y_max = min( bspline_l.qmax, bspline_r.qmax )

            y = np.linspace( y_min, y_max, 200 )

            # get the x-coordinates of each b-spline
            if col == 1:
                left_matches = np.vstack( (bspline_l( y ), y) ).T
                right_matches = np.vstack( (bspline_r( y ), y) ).T

            # if

            elif col == 0:
                left_matches = np.vstack( (y, bspline_l( y )) ).T
                right_matches = np.vstack( (y, bspline_l( y )) ).T

            # elif

            else:
                left_matches = None
                right_matches = None

        # else
    # elif

    else:
        raise NotImplementedError( f"method = {method} not implemented." )

    # else

    return left_matches, right_matches


# stereomatch_needle


def stereomatch_normxcorr( left_conts, right_conts, img_left, img_right,
                           roi_l_mask=None, roi_r_mask=None, score_thresh=0.75,  # thresh=50,
                           col: int = 1, zoom=1.0, winsize=(5, 5) ):
    """ stereo matching needle arclength points for the needle

        performs this using normxcorr along 'col'-axis

        Args:
            (left/right)_conts: a nx2 array of pixel coordinates
                                for the contours in the (left/right) image

            img_(left/right): the left and right rectified images.

            roi_(left/right): the rectified left/right ROI images.

            col (int = 1): the column to begin matching by

            winsize: 2 tuple to match (rows, cols) OR (y, x)

     """
    # argument checking
    assert (0 < zoom)  # limit the make sure zoom is positive

    assert (len( winsize ) == 2)
    assert (all( length % 2 == 1 for length in winsize ))  # make sure they are all both odd

    # process the contours
    if isinstance( left_conts, np.ndarray ):
        left_conts = np.squeeze( left_conts )  # squeeze dimensions just in case

    # if

    elif isinstance( left_conts, BSpline1D ):
        s = np.linspace( 0, 1, 200 )
        if col == 0:
            left_conts = np.vstack( (left_conts.unscale( s ), left_conts.eval_unscale( s )) ).T

        else:
            left_conts = np.vstack( (left_conts.eval_unscale( s ), left_conts.unscale( s )) ).T

    # elif

    else:
        raise TypeError( 'left_conts is not a BSpline1D or numpy array.' )

    # else

    if isinstance( right_conts, np.ndarray ):
        right_conts = np.squeeze( right_conts )  # squeeze dimensions just in case

    # if

    elif isinstance( right_conts, BSpline1D ):
        s = np.linspace( 0, 1, 200 )
        if col == 0:
            right_conts = np.vstack( (right_conts.unscale( s ), right_conts.eval_unscale( s )) ).T

        else:
            right_conts = np.vstack( (right_conts.eval_unscale( s ), right_conts.unscale( s )) ).T

    # elif

    else:
        raise TypeError( 'right_conts is not a BSpline1D or numpy array.' )

    # else

    # pad the images with winsize and zoom
    left_pad = np.pad( img_left, ((winsize[ 0 ] // 2, winsize[ 0 ] // 2), (winsize[ 1 ] // 2, winsize[ 1 ] // 2)) )
    right_pad = np.pad( img_right, ((winsize[ 0 ] // 2, winsize[ 0 ] // 2), (winsize[ 1 ] // 2, winsize[ 1 ] // 2)) )
    left_zoom = cv.resize( left_pad, None, fx=zoom, fy=zoom, interpolation=cv.INTER_CUBIC )
    right_zoom = cv.resize( right_pad, None, fx=zoom, fy=zoom, interpolation=cv.INTER_CUBIC )

    # remove duplicate rows
    pts_l = np.unique( left_conts, axis=0 )
    pts_r = np.unique( right_conts, axis=0 )

    # ================= helper function ====================================
    def normxcorr1( template, img, row ):
        """ returns the best match in a line along a specific row index"""
        # unpack template shape
        t_0, t_1 = template.shape

        # get the search image
        # # handle edge cases
        img_roi = np.array( [ [ row - t_0 // 2, 0 ], [ row + t_0 // 2 + 1, -1 ] ] )
        img_roi[ img_roi[ :, 0 ] < 0, 0 ] = 0
        img_search = roi( img, img_roi, full=False )

        res = cv.matchTemplate( img_search, template, cv.TM_CCOEFF_NORMED )

        min_val, max_val, min_loc, max_loc = cv.minMaxLoc( res )

        best_idx = (max_loc[ 1 ] + row, max_loc[ 0 ] + t_1 // 2 + 1)

        return best_idx, max_val

    # = def: normxcorr1 ====================================================

    # match points left-to-right
    new_winsize = tuple( [ round( zoom * win ) for win in winsize ] )
    new_winsize = tuple( [ win if win % 2 == 1 else win + 1 for win in new_winsize ] )
    left_matches = np.zeros( (0, 2) )
    right_matches = np.zeros( (0, 2) )
    for i in range( pts_l.shape[ 0 ] ):
        # grab the template
        px_pt = np.flip( zoom * pts_l[ i ] ).round().astype( np.int32 ) + np.array( new_winsize ) // 2

        img_roi = np.vstack( (-np.array( new_winsize ) // 2, np.array( new_winsize ) // 2) ) + px_pt.reshape( 1,
                                                                                                              -1 ).tolist()
        img_roi[ img_roi < 0 ] = 0

        template = roi( left_zoom, img_roi, full=False )

        if template.shape[ 0 ] % 2 == 0:
            template = template[ :-1, : ]

        if template.shape[ 1 ] % 2 == 0:
            template = template[ :, :-1 ]

        match_pt, score = normxcorr1( template, right_zoom, px_pt[ 0 ] )
        if score > score_thresh:
            left_matches = np.append( left_matches, np.reshape( px_pt - np.array( new_winsize ) // 2, (1, 2) ), axis=0 )
            right_matches = np.append( right_matches, np.reshape( match_pt - np.array( new_winsize ) // 2, (1, 2) ),
                                       axis=0 )

            # if
    # for

    return np.flip( left_matches, axis=1 ) / zoom, np.flip( right_matches, axis=1 ) / zoom  # when using px_pt


# stereomatch_normxcorr


def triangulate_points( pts_l, pts_r, stereo_params: dict, distorted: bool = False ):
    """ function to perform 3-D reconstruction of the pts in left and right images.

        @param pts_(l/r): the left/right image points to triangulate of size [Nx2]
        @param stereo_params: dict of the stereo parameters
        @param distorted: (bool, Default=True) whether to undistort the pts in each image

        DO NOT USE 'distorted'! This causes major errors @ the moment.

        @return: [Nx3] world frame points

    """

    # load in stereo parameters, camera matrices and distortion coefficients
    # - camera matrices
    Kl = stereo_params[ 'cameraMatrix1' ]
    distl = stereo_params[ 'distCoeffs1' ]

    Kr = stereo_params[ 'cameraMatrix2' ]
    distr = stereo_params[ 'distCoeffs2' ]

    # - stereo parameters
    R = stereo_params[ 'R' ]
    t = stereo_params[ 't' ]

    # convert to float types
    pts_l = np.float64( pts_l )
    pts_r = np.float64( pts_r )

    # undistort the points if needed
    if distorted:
        pts_l, pts_r = undistort_points( pts_l, pts_r, stereo_params )

        # get undistorted camera params
        Kl = stereo_params[ 'cameraMatrix1_new' ]
        Kr = stereo_params[ 'cameraMatrix2_new' ]

    # if

    # calculate projection matrices
    Pl = Kl @ np.eye( 3, 4 )
    H = np.vstack( (np.hstack( (R, t.reshape( 3, 1 )) ), [ 0, 0, 0, 1 ]) )
    Pr = Kr @ H[ 0:3 ]

    # - transpose to [2 x N]
    pts_l = pts_l.T
    pts_r = pts_r.T

    # perform triangulation of the points
    pts_3d = cv.triangulatePoints( Pl, Pr, pts_l, pts_r )
    pts_3d /= pts_3d[ 3 ]  # normalize the triangulation points

    return pts_3d[ :-1 ]


# triangulate


def thresh( left_img, right_img, thresh: Union[ str, int ] = 'adapt', thresh_max: int = 255 ):
    """ image thresholding"""

    if isinstance( thresh, str ):
        if thresh.lower() == 'adapt':
            left_thresh = cv.adaptiveThreshold( left_img.astype( np.uint8 ), thresh_max, cv.ADAPTIVE_THRESH_MEAN_C,
                                                cv.THRESH_BINARY_INV, 13, 4 )
            right_thresh = cv.adaptiveThreshold( right_img.astype( np.uint8 ), thresh_max, cv.ADAPTIVE_THRESH_MEAN_C,
                                                 cv.THRESH_BINARY_INV, 13, 4 )

        # if
    # if

    elif isinstance( thresh, (float, int) ):
        _, left_thresh = cv.threshold( left_img, thresh, thresh_max, cv.THRESH_BINARY_INV )
        _, right_thresh = cv.threshold( right_img, thresh, thresh_max, cv.THRESH_BINARY_INV )

    # elif

    else:
        raise ValueError( f"thresh: {thresh} is not a valid thresholding." )

    return left_thresh, right_thresh


# thresh


def undistort( left_img, right_img, stereo_params: dict ):
    """ stereo wrapper to undistort """
    # load in camera matrices and distortion coefficients
    Kl = stereo_params[ 'cameraMatrix1' ]
    distl = stereo_params[ 'distCoeffs1' ]

    Kr = stereo_params[ 'cameraMatrix2' ]
    distr = stereo_params[ 'distCoeffs2' ]

    # undistort/recitfy the images
    hgtl, wdtl = left_img.shape[ :2 ]
    Kl_new, roi = cv.getOptimalNewCameraMatrix( Kl, distl, (wdtl, hgtl), 1, (wdtl, hgtl) )
    xl, yl, wl, hl = roi
    left_img_rect = cv.undistort( left_img, Kl, distl, None, Kl_new )[ yl:yl + hl, xl:xl + wl ]

    hgtr, wdtr = right_img.shape[ :2 ]
    Kr_new, roi = cv.getOptimalNewCameraMatrix( Kr, distr, (wdtr, hgtr), 1, (wdtr, hgtr) )
    xr, yr, wr, hr = roi
    right_img_rect = cv.undistort( right_img, Kr, distr, None, Kr_new )[ yr:yr + hr, xr:xr + wr ]

    return left_img_rect, right_img_rect


# undistort


def undistort_points( pts_l, pts_r, stereo_params: dict ):
    """ wrapper for undistorting points

        pts is of shape [N x 2]

    """
    # load in camera matrices and distortion coefficients
    Kl = stereo_params[ 'cameraMatrix1' ]
    distl = stereo_params[ 'distCoeffs1' ]

    Kr = stereo_params[ 'cameraMatrix2' ]
    distr = stereo_params[ 'distCoeffs2' ]

    # calculate optimal camera matrix
    Kl_new, _ = cv.getOptimalNewCameraMatrix( Kl, distl, IMAGE_SIZE, 1, IMAGE_SIZE )
    Kr_new, _ = cv.getOptimalNewCameraMatrix( Kr, distr, IMAGE_SIZE, 1, IMAGE_SIZE )

    stereo_params[ 'cameraMatrix1_new' ] = Kl_new
    stereo_params[ 'cameraMatrix2_new' ] = Kr_new

    # undistort the image points
    pts_l_undist = cv.undistortPoints( np.expand_dims( pts_l, 1 ), Kl, distl,
                                       None, Kl_new ).squeeze()
    pts_r_undist = cv.undistortPoints( np.expand_dims( pts_r, 1 ), Kr, distr,
                                       None, Kr_new ).squeeze()

    return pts_l_undist, pts_r_undist


# undistort_points

def main():
    # set-up
    validation = False  # DO NOT CHANGE, KEEP False
    insertion_expmt = True  # DO NOT CHANGE, KEEP True
    proc_show = False  # Can change, True if you would like to see the processing of the images
    res_show = False  # Can change, True if you would like to see the results of the needle reconstruction
    save_bool = True  # Can change, True if you would like to save the processed data

    # directory settings
    stereo_dir = "../data/"
    # needle_dir = stereo_dir + "needle_examples/"  # needle insertion examples directory DO NOT USE
    # grid_dir = stereo_dir + "grid_only/"  # grid testqing directory DO NOT USE
    valid_dir = stereo_dir + "stereo_validation_jig/"  # validation directory DO NOT USE
    insertion_dir = "../../data/3CH-4AA-0004/08-30-2021_Insertion-Expmt-1/"  # CHANGE, CHANGE HERE FOR DIFFERENT DATASET

    curvature_dir = glob.glob( os.path.join( valid_dir, 'k_*/' ) )  # validation curvature directories
    curvature_dir = sorted( curvature_dir )
    #     curvature_dir = []  # cancel it out | don't want to do this right now

    insertion_dirs = glob.glob( os.path.join( insertion_dir, "Insertion*/" ) )
    insertion_dirs = sorted( [ os.path.normpath( d ) for d in insertion_dirs ] )

    # load matlab stereo calibration parameters
    stereo_param_dir = "../calibration/Stereo_Camera_Calibration_02-08-2021/6x7_5mm/"
    stereo_param_file = os.path.join( stereo_param_dir, "calibrationSession_params-error_opencv-struct.mat" )
    stereo_params = load_stereoparams_matlab( stereo_param_file )

    # regex pattern
    pattern = r".*/?(left|right)-([0-9]{4}).png"  # image regex

    # perform validation over the entire dataset
    warnings.filterwarnings( 'ignore', message='.*ndarray.*' )
    if validation:
        for curv_dir in curvature_dir:
            # gather curvature file numbers
            files = sorted( glob.glob( curv_dir + 'left-*.png' ) )
            file_nums = [ int( re.match( pattern, f ).group( 2 ) ) for f in files if re.match( pattern, f ) ]

            print( 'Working on directory:', curv_dir )

            try:
                if save_bool:
                    main_needleval( file_nums, curv_dir, stereo_params,
                                    save_dir=curv_dir, proc_show=proc_show, res_show=False )
                # if

                else:
                    main_needleval( file_nums, curv_dir, stereo_params,
                                    save_dir=None, proc_show=proc_show, res_show=False )
                # else
            # try

            except Exception as e:
                print( e )
                print( 'Continuing...' )
                raise e

            # except

            print()

        # for

    # if

    # run the insertion experiment
    if insertion_expmt:
        #         main_insertionval( insertion_dirs, stereo_params, save_bool = save_bool,
        #                           proc_show = proc_show, res_show = res_show )
        main_insertion_sub( insertion_dirs, stereo_params, save_bool=save_bool,
                            proc_show=proc_show, res_show=res_show )

    # if

    print( 'Program complete.' )


# main

def main_dbg():
    # directory settings
    stereo_dir = "../Test Images/stereo_needle/"
    needle_dir = stereo_dir + "needle_examples/"
    grid_dir = stereo_dir + "grid_only/"

    # the left and right image to test
    num = 5
    left_fimg = needle_dir + f"left-{num:04d}.png"
    right_fimg = needle_dir + f"right-{num:04d}.png"

    # load matlab stereo calibration parameters
    stereo_param_dir = "../Stereo_Camera_Calibration_10-23-2020"
    stereo_param_file = stereo_param_dir + "/calibrationSession_params-error_opencv-struct.mat"
    stereo_params = load_stereoparams_matlab( stereo_param_file )

    #     # read in the images and convert to grayscale
    #     left_img = cv.imread( left_fimg, cv.IMREAD_COLOR )
    #     right_img = cv.imread( right_fimg, cv.IMREAD_COLOR )
    #     left_gray = cv.cvtColor( left_img, cv.COLOR_BGR2GRAY )
    #     right_gray = cv.cvtColor( right_img, cv.COLOR_BGR2GRAY )

    #     # test undistort function ( GOOD )
    #     left_rect, right_rect = undistort( left_img, right_img, stereo_params )
    #     test_arr = np.zeros( ( 3, 2 ) )
    #     undist_pts = undistort_points( test_arr, test_arr, stereo_params )
    #     print( np.hstack( undist_pts ) )
    #     print()

    # test point triangulation ( GOOD )
    world_points = np.random.randn( 3, 5 )
    world_pointsh = np.vstack( (world_points, np.ones( (1, world_points.shape[ 1 ]) )) )
    Pl = stereo_params[ 'P1' ]
    Pr = stereo_params[ 'P2' ]
    pts_l = Pl @ world_pointsh
    pts_l = (pts_l / pts_l[ -1 ]).T[ :, :-1 ]
    pts_r = Pr @ world_pointsh
    pts_r = (pts_r / pts_r[ -1 ]).T[ :, :-1 ]

    print( 'pts shape (l,r):', pts_l.shape, pts_r.shape )
    tri_pts = triangulate_points( pts_l, pts_r, stereo_params, distorted=False )
    print( 'World points' )
    print( world_points )
    print()
    print( 'triangulated points' )
    print( tri_pts )
    print()

    #     # plotting / showing image results
    #     plt.ion()
    #
    #     plt.figure()
    #     plt.imshow( imconcat( left_img, right_img, [0, 0, 255] ) )
    #     plt.title( 'original image' )
    #
    #     plt.figure()
    #     plt.imshow( imconcat( left_rect, right_rect, [0, 0, 255] ) )
    #     plt.title( 'undistorted image' )
    #
    #     # close on enter
    #     plt.show()
    #     while True:
    #         if plt.waitforbuttonpress( 0 ):
    #             break
    #
    #     # while

    plt.close( 'all' )


# main_dbg


def main_img_gui( file_num, img_dir ):
    """ run a custom GUI for image processing testing """
    left_file = img_dir + f'left-{file_num:04d}.png'
    right_file = img_dir + f'right-{file_num:04d}.png'

    left_img = cv.imread( left_file, cv.IMREAD_ANYCOLOR )
    right_img = cv.imread( right_file, cv.IMREAD_ANYCOLOR )

    f = 2
    dsize = (left_img.shape[ 1 ] // f, right_img.shape[ 0 ] // f)

    left_img = cv.resize( left_img, dsize, fx=f, fy=f )
    right_img = cv.resize( right_img, dsize, fx=f, fy=f )

    lr_img = imconcat( left_img, right_img, [ 255, 0, 0 ] )

    print( 'Working on HSV detection.' )
    hsv_vals = find_hsv_image( lr_img )

    print( 'Gathered HSV values:' )
    for v in hsv_vals:
        print( v )

    print()


# main_img_gui


def main_insertion_sub( insertion_dirs, stereo_params, save_bool: bool = False,
                        proc_show: bool = False, res_show: bool = False ):
    """ main method for running through insertion data using reference imaging

        insertion_dirs are all the insertion directories (will find each of the insertion distances

    """
    # regexp pattens
    pattern = r".*[/,\\]Insertion([0-9]+)[/,\\]([0-9]+)[/,\\]?"

    # iterate through the directories
    time_trials = [ ]
    for ins_dir in insertion_dirs:
        print( 100 * '=' )
        print( f"Processing directory: {ins_dir}" )
        # load in the reference images
        ref_dir = os.path.join( ins_dir, "0/" )
        left_ref = cv.imread( os.path.join( ref_dir, "left.png" ), cv.IMREAD_ANYCOLOR )
        right_ref = cv.imread( os.path.join( ref_dir, "right.png" ), cv.IMREAD_ANYCOLOR )

        # - check for reference images
        if left_ref is None:
            print( "Could not find left reference image: {}".format( os.path.join( ref_dir, "left.png" ) ) )
            continue

        # if

        if right_ref is None:
            print( "Could not find right reference image: {}".format( os.path.join( ref_dir, "right.png" ) ) )
            continue

        # if

        # load other insertion directories
        ins_dist_dirs = sorted( [ os.path.normpath( d ) for d in glob.glob( os.path.join( ins_dir, "*/" ) ) ] )

        for ins_dist_dir in ins_dist_dirs:
            # regular expression matching
            res = re.match( pattern, ins_dist_dir )
            if res is None:
                print( ins_dist_dir, 'is not a valid directory format.' )
                continue

            # if

            else:
                ins_num, ins_dist = res.groups()
                ins_num = int( ins_num )
                ins_dist = float( ins_dist )

            # else

            # make sure for positive insertion distance
            if ins_dist <= 0:
                continue

            # if

            # elif ins_dist != 105:
            #     continue

            # elif ins_num < 8:
            #     continue

            # blackout-regions
            bors_l = [
                    # [[80, 120], [140, -1]],  # staples
                    [ [ -60, 0 ], [ -1, -1 ] ] ]
            bors_r = [
                    # [[70, 140], [120, -1]],  # staples
                    [ [ -60, 0 ], [ -1, -1 ] ] ]

            # =============================== CHANGE HERE FOR ROIs =====================================================
            # load in the pre-determined ROIs
            roi_l = [ [ 40, 200 ],
                      [ -10, -300 ] ]
            roi_r = [ [ roi_l[ 0 ][ 0 ] - 10, roi_l[ 0 ][ 1 ] + 50 ],
                      [ roi_l[ 1 ][ 0 ] - 15, roi_l[ 1 ][ 1 ] + 50 ] ]

            # ==========================================================================================================

            # load the images
            left_file = os.path.join( ins_dist_dir, 'left.png' )
            right_file = os.path.join( ins_dist_dir, 'right.png' )

            left_img = cv.imread( left_file, cv.IMREAD_ANYCOLOR )
            right_img = cv.imread( right_file, cv.IMREAD_ANYCOLOR )

            # perform the reconstruction
            t0 = time.time()
            pts_3d, *_, proc_images, figures = needle_reconstruction_ref( left_img, left_ref,
                                                                          right_img, right_ref,
                                                                          stereo_params,
                                                                          bor_l=bors_l, bor_r=bors_r,
                                                                          roi_l=roi_l, roi_r=roi_r,
                                                                          alpha=0.6, recalc_stereo=True,
                                                                          # zoom = 1.0, winsize = ( 201, 51 ),  # testing
                                                                          zoom=2.5, winsize=(201, 51),  # production
                                                                          sub_thresh=60, proc_show=proc_show )

            arclength = np.linalg.norm( np.diff( pts_3d, axis=0 ), axis=1 ).sum()
            print( f"Arclength of reconstruction: {arclength:.3f} mm" )
            # measure the time per trial
            dt = time.time() - t0
            time_trials.append( dt )
            if save_bool:
                print( 'Saving figures and files...' )
                save_fbase = os.path.join( ins_dist_dir, "left-right_{:s}" )

                # save the 3-D points
                np.savetxt( save_fbase.format( '3d-pts' ) + '.txt', pts_3d )
                print( 'Saved data file:', save_fbase.format( '3d-pts' ) + '.txt' )

                # save the processed images
                for key, imgs in proc_images.items():
                    cv.imwrite( save_fbase.format( key ) + '.png', imgs )
                    print( 'Saved figure:', save_fbase.format( key ) + '.png' )

                    # for: images

                # save the figures
                for key, fig in figures.items():
                    fig.savefig( save_fbase.format( key + '-fig' ) + '.png' )
                    print( 'Saved figure:', save_fbase.format( key + '-fig' ) + '.png' )

                # for: figures

                print( 'Finished saving files and figures.', end='\n\n' + 80 * '=' + '\n\n' )

            # if

            print( f"Completed trial: '{ins_dist_dir}'." )
            print( f'Time for Trial: {round( dt / 60 )} mins. {dt % 60:.2f} s' )
            print( '\n' + 75 * '=', end='\n\n' )
            plt.close( 'all' )

        # for

        # display average time.
        if len( time_trials ) > 0:
            dt_avg = sum( time_trials ) / len( time_trials )
            print( f'Average time for all: {round( dt_avg / 60 )} mins. {dt_avg % 60:.2f} s' )

        # if
    # for


# main_insertion_sub


def main_insertionval( insertion_dirs, stereo_params, save_bool: bool = False,
                       proc_show: bool = False, res_show: bool = False ):
    """ main method for running through insertion data """
    # regexp pattens
    pattern = r".*/Insertion([0-9]+)/([0-9]+)/?"

    # iterate through the directories
    time_trials = [ ]
    for ins_dir in insertion_dirs:
        # regular expression matching
        res = re.match( pattern, ins_dir )
        if res is None:
            print( ins_dir, 'is not a valid directory format.' )
            continue

        # if

        else:
            ins_num, ins_dist = res.groups()
            ins_num = int( ins_num )
            ins_dist = float( ins_dist )

        # else

        # make sure for positive insertion distance
        if ins_dist <= 0:
            continue

        # if

        #         # testings to skip
        #         if ins_dist != 90:
        #             continue
        #
        #         # if

        if ins_num != 3:
            continue

        # blackout-regions
        bors_l = [ ]
        bors_r = [ ]

        # load in the pre-determined ROIs
        h_off = 3
        r_off = -100
        rois_lr = np.load( ins_dir + 'rois_lr.npy' ) + np.array(
                [ [ h_off, 0, h_off, 0 ], [ 0, r_off, 0, r_off ] ] ).reshape( 1, 2, 4 )
        roi_l = rois_lr[ 0, :, 0:2 ]
        roi_r = rois_lr[ 0, :, 2:4 ]
        #         if ins_num < 2:
        #             roi_l = [[62, 300], [-250, 400]]
        #             roi_r = [[55, 325], [-250, 425]]
        #
        #         elif ins_num < 3:
        #             roi_l = [[62, 300], [-250, 450]]
        #             roi_r = [[55, 325], [-250, 475]]
        #
        #         elif ins_num < 5:
        #             roi_l = [[62, 400], [-250, 550]]
        #             roi_r = [[55, 425], [-250, 575]]
        #
        #         elif ins_num < 7:
        #             roi_l = [[62, 450], [-250, 600]]
        #             roi_r = [[55, 475], [-250, 625]]
        #
        #         elif ins_num < 9:
        #             roi_l = [[62, 525], [-250, 675]]
        #             roi_r = [[55, 550], [-250, 700]]
        #
        #         elif ins_num < 10:
        #             roi_l = [[62, 575], [-250, 725]]
        #             roi_r = [[55, 600], [-250, 750]]
        #
        #         else:
        #             roi_l = []
        #             roi_r = []

        # load the images
        left_file = ins_dir + 'left.png'
        right_file = ins_dir + 'right.png'

        left_img = cv.imread( left_file, cv.IMREAD_ANYCOLOR )
        right_img = cv.imread( right_file, cv.IMREAD_ANYCOLOR )

        # perform the reconstruction
        t0 = time.time()
        pts_3d, *_, proc_images, figures = needle_tissue_reconstruction_refined( left_img, right_img, stereo_params,
                                                                                 bor_l=bors_l, bor_r=bors_r,
                                                                                 roi_l=roi_l, roi_r=roi_r,
                                                                                 alpha=0.6, recalc_stereo=True,
                                                                                 proc_show=proc_show, zoom=2.5,
                                                                                 winsize=(51, 51) )

        arclength = np.linalg.norm( np.diff( pts_3d, axis=0 ), axis=1 ).sum()
        print( f"Arclength of reconstruction: {arclength:.3f} mm" )
        # measure the time per trial
        dt = time.time() - t0
        time_trials.append( dt )
        if save_bool:
            print( 'Saving figures and files...' )
            save_fbase = ins_dir + "left-right_{:s}"

            # save the 3-D points
            np.savetxt( save_fbase.format( '3d-pts' ) + '.txt', pts_3d )
            print( 'Saved data file:', save_fbase.format( '3d-pts' ) + '.txt' )

            # save the processed images
            for key, imgs in proc_images.items():
                cv.imwrite( save_fbase.format( key ) + '.png', imgs )
                print( 'Saved figure:', save_fbase.format( key ) + '.png' )

                # for: images

            # save the figures
            for key, fig in figures.items():
                fig.savefig( save_fbase.format( key + '-fig' ) + '.png' )
                print( 'Saved figure:', save_fbase.format( key + '-fig' ) + '.png' )

            # for: figures

            print( 'Finished saving files and figures.', end='\n\n' + 80 * '=' + '\n\n' )

        # if

        print( f"Completed trial: '{ins_dir}'." )
        print( f'Time for Trial: {round( dt / 60 )} mins. {dt % 60:.2f} s' )
        print( '\n' + 75 * '=', end='\n\n' )
        plt.close( 'all' )

    # for

    # display average time.
    if len( time_trials ) > 0:
        dt_avg = sum( time_trials ) / len( time_trials )
        print( f'Average time for all: {round( dt_avg / 60 )} mins. {dt_avg % 60:.2f} s' )

    # if


# main_insertionval


def main_gridproc( num, img_dir, save_dir ):
    """ main method to segment the grid in a stereo pair of images"""
    # the left and right image to test
    left_fimg = img_dir + f"left-{num:04d}.png"
    right_fimg = img_dir + f"right-{num:04d}.png"

    left_img = cv.imread( left_fimg, cv.IMREAD_COLOR )
    right_img = cv.imread( right_fimg, cv.IMREAD_COLOR )
    left_gray = cv.cvtColor( left_img, cv.COLOR_BGR2GRAY )
    right_gray = cv.cvtColor( right_img, cv.COLOR_BGR2GRAY )
    left_img2 = cv.cvtColor( left_img, cv.COLOR_RGB2BGR )
    right_img2 = cv.cvtColor( right_img, cv.COLOR_RGB2BGR )

    # color segmentation ( red for border )
    lmask, rmask, lcolor2, rcolor2 = color_segmentation( left_img2, right_img2, "red" )
    lcolor = cv.cvtColor( lcolor2, cv.COLOR_BGR2RGB )
    rcolor = cv.cvtColor( rcolor2, cv.COLOR_BGR2RGB )

    # plotting
    plt.ion()

    #     plt.figure()
    #     plt.imshow( imconcat( left_img, right_img ) )
    #     plt.title( 'Original images' )

    #     plt.figure()
    #     plt.imshow( imconcat( lcolor, rcolor ) )
    #     plt.title( 'masked red color' )

    # find the grid
    gridproc_stereo( left_gray, right_gray, proc_show=True )

    # close on enter
    plt.show()
    while True:
        if plt.waitforbuttonpress( 0 ):
            break

    # while

    plt.close( 'all' )


# main_gridproc


def main_needleproc( file_num, img_dir, save_dir=None, proc_show=False, res_show=False ):
    """ main method for segmenting the needle centerline in stereo images"""
    # load matlab stereo calibration parameters
    stereo_param_dir = "../Stereo_Camera_Calibration_10-23-2020"
    stereo_param_file = stereo_param_dir + "/calibrationSession_params-error_opencv-struct.mat"
    stereo_params = load_stereoparams_matlab( stereo_param_file )

    # the left and right image to test
    left_fimg = img_dir + f"left-{file_num:04d}.png"
    right_fimg = img_dir + f"right-{file_num:04d}.png"

    left_img = cv.imread( left_fimg, cv.IMREAD_COLOR )
    right_img = cv.imread( right_fimg, cv.IMREAD_COLOR )
    left_gray = cv.cvtColor( left_img, cv.COLOR_BGR2GRAY )
    right_gray = cv.cvtColor( right_img, cv.COLOR_BGR2GRAY )

    print( 'Image shape:', left_gray.shape, end='\n\n' + 80 * '=' + '\n\n' )

    # blackout regions
    bor_l = [ (left_gray.shape[ 0 ] - 100, 0), left_gray.shape ]
    bor_r = bor_l

    # regions of interest
    roi_l = ((70, 80), (500, 915))
    roi_r = ((70, 55), (500, -1))

    # needle image processing
    print( 'Processing stereo pair images...' )
    left_skel, right_skel, conts_l, conts_r = needleproc_stereo( left_img, right_img,
                                                                 bor_l=[ bor_l ], bor_r=[ bor_r ],
                                                                 roi_l=roi_l, roi_r=roi_r,
                                                                 proc_show=proc_show )
    print( 'Stereo pair processed. Contours extracted.', end='\n\n' + 80 * '=' + '\n\n' )

    left_cont = left_img.copy()
    left_cont = cv.drawContours( left_cont, conts_l, 0, (255, 0, 0), 12 )

    right_cont = right_img.copy()
    right_cont = cv.drawContours( right_cont, conts_r, 0, (255, 0, 0), 12 )

    # matching contours
    print( 'Performing stereo triangulation...' )
    cont_l_match, cont_r_match = stereomatch_needle( conts_l[ 0 ], conts_r[ 0 ], start_location='tip', col=1 )

    left_match = left_cont.copy()
    cv.drawContours( left_match, [ np.vstack( (cont_l_match, np.flip( cont_l_match, 0 )) ) ], 0, (0, 255, 0), 4 )

    right_match = right_cont.copy()
    cv.drawContours( right_match, [ np.vstack( (cont_r_match, np.flip( cont_r_match, 0 )) ) ], 0, (0, 255, 0), 4 )

    # draw lines from matching points
    plot_pt_freq = int( 0.1 * len( cont_l_match ) )
    pad_width = 20
    lr_match = imconcat( left_match, right_match, pad_val=[ 0, 0, 255 ], pad_size=pad_width )
    for (x_l, y_l), (x_r, y_r) in zip( cont_l_match[ ::plot_pt_freq ], cont_r_match[ ::plot_pt_freq ] ):
        cv.line( lr_match, (x_l, y_l), (x_r + pad_width + right_match.shape[ 1 ], y_r), [ 255, 0, 255 ], 2 )

    # for

    # perform triangulation on points
    cont_match_3d = triangulate_points( cont_l_match, cont_r_match, stereo_params, distorted=True )

    # - smooth 3-D points
    print( 'Smoothing 3-D stereo points and fitting 3-D NURBS...' )
    win_size = 55
    cont_match_3d_sg = savgol_filter( cont_match_3d, win_size, 1, deriv=0 )

    # - NURBS fitting
    nurbs = fitting.approximate_curve( cont_match_3d_sg.T.tolist(), degree=2, ctrlpts_size=35 )
    nurbs.delta = 0.005
    nurbs.vis = VisMPL.VisCurve3D()
    print( 'Smoothing and NURBS fit.', end='\n\n' + 80 * '=' + '\n\n' )

    # test disparity mapping
    disparity = stereo_disparity( left_img, right_img, stereo_params )

    # show results
    if res_show:
        print( 'Plotting...' )
        plt.ion()

        plt.figure()
        plt.imshow( imconcat( left_cont, right_cont, pad_val=[ 0, 0, 255 ] ) )
        plt.title( 'Contours of needle' )

        plt.figure()
        plt.imshow( lr_match )
        plt.title( 'matching contour points' )

        extras = [
                dict( points=cont_match_3d.T.tolist(),
                      name='triangulation',
                      color='red',
                      size=1 ),
                dict( points=cont_match_3d_sg.T.tolist(),
                      name='savgol_filter',
                      color='green',
                      size=1 )
                ]
        nurbs.render( extras=extras )
        ax = plt.gca()
        axisEqual3D( ax )

        # ==================== OLD 3-D PLOTTING =========================================
        #         f3d = plt.figure()
        #         ax = fig3d.add_subplot(111, projection='3d')
        #         ax.plot( cont_match_3d[0], cont_match_3d[1], cont_match_3d[2], '.' , label = 'triangulation' )
        #         ax.plot( cont_match_3d_sg[0], cont_match_3d_sg[1], cont_match_3d_sg[2], '-' , label = 'savgol_filter' )
        # #         ax.plot( cont_match_3d_mvavg[0], cont_match_3d_mvavg[1], cont_match_3d_mvavg[2], '-' , label='moving average')
        #         plt.legend( [ 'nurbs', 'triangulation', 'savgol_filter'] )
        #         axisEqual3D( ax )
        #         plt.title( '3-D needle reconstruction' )
        # ===============================================================================

        plt.figure()
        plt.imshow( disparity, cmap='gray' )
        plt.title( 'stereo disparity map' )

        # close on enter
        print( 'Press any key on the last figure to close all windows.' )
        plt.show()
        while True:
            try:
                if plt.waitforbuttonpress( 0 ):
                    break

                # if
            # try

            except:
                break

            # except
        # while

        print( 'Closing all windows...' )
        plt.close( 'all' )
        print( 'Plotting finished.', end='\n\n' + 80 * '=' + '\n\n' )

    # if

    # save the processed images
    if save_dir:
        print( 'Saving figures and files...' )
        save_fbase = save_dir + f"left-right-{file_num:04d}" + "_{:s}.png"
        save_fbase_txt = save_dir + f"left-right-{file_num:04d}" + "_{:s}.txt"
        save_fbase_fmt = save_dir + f"left-right-{file_num:04d}" + "_{:s}.{:s}"

        # - skeletons
        plt.imsave( save_fbase.format( 'skel' ), imconcat( left_skel, right_skel, 150 ),
                    cmap='gray' )
        print( 'Saved figure:', save_fbase.format( 'skel' ) )

        # - contours
        plt.imsave( save_fbase.format( 'cont' ), imconcat( left_cont, right_cont, pad_val=[ 0, 0, 255 ] ) )
        print( 'Saved figure:', save_fbase.format( 'cont' ) )

        # - matching contours
        plt.imsave( save_fbase.format( 'cont-match' ), lr_match )
        print( 'Saved Figure:', save_fbase.format( 'cont-match' ) )

        # - 3D reconstruction
        extras = [
                dict( points=cont_match_3d.T.tolist(),
                      name='triangulation',
                      color='red',
                      size=1 ),
                dict( points=cont_match_3d_sg.T.tolist(),
                      name='savgol_filter',
                      color='green',
                      size=1 )
                ]
        nurbs.render( plot=False, filename=save_fbase.format( '3d-reconstruction' ), extras=extras )
        ax = plt.gca()
        axisEqual3D( ax )
        print( 'Saved Figure:', save_fbase.format( '3d-reconstruction' ) )

        np.savetxt( save_fbase_txt.format( 'cont-match' ), np.hstack( (cont_l_match, cont_r_match) ) )
        print( 'Saved file:', save_fbase_txt.format( 'cont-match' ) )

        np.savetxt( save_fbase_txt.format( 'cont-match_3d' ), cont_match_3d )
        print( 'Saved file:', save_fbase_txt.format( 'cont-match_3d' ) )

        nurbs.save( save_fbase_fmt.format( 'nurbs', 'pkl' ) )
        print( 'Saved nurbs:', save_fbase_fmt.format( 'nurbs', 'pkl' ) )

        np.savetxt( save_fbase_txt.format( 'nurbs-pts' ), nurbs.evalpts )
        print( 'Saved file:', save_fbase_txt.format( 'nurbs-pts' ) )

        plt.close()
        print( 'Finished saving files and figures.', end='\n\n' + 80 * '=' + '\n\n' )

    # if

    return nurbs


# main_needleproc


def main_needleval( file_nums, img_dir, stereo_params, save_dir=None,
                    proc_show: bool = False, res_show: bool = False ):
    """
        NEED TO UPDATE NEEDLE PROCESSING FOR SKELETONIZATIONSQ

        main method for needle validation using the jig (known curvature)

        Args:
            file_nums: int or list of integers for the image number stereo pairs
            img_dir: string of the curvature directory
            save_dir (Default None): where to save the data (if None, no saving)
            stereo_params: dict of stereo parameters
            proc_show (bool = False): whether to show image processing
            res_show (bool = False): whether to show the results
    """
    # regex pattern
    re_pattern = r".*k_([0-9].+[0-9]+)/?"
    re_match = re.match( re_pattern, img_dir )

    if not re_match:
        raise ValueError( "'file_dir' does not have a valid curvature format." )

    # if

    k = float( re_match.group( 1 ) )  # curvature (1/m)
    print( 'Processing curvature = ', k, '1/m' )

    # iterate through the stereo pairs
    if isinstance( file_nums, int ):
        file_nums = range( file_nums )

    # if

    # load in the pre-determined ROIs
    rois_rl = np.load( img_dir + 'rois_lr.npy' )
    rois_l = rois_rl[ :, :, 0:2 ].tolist()
    rois_r = rois_rl[ :, :, 2:4 ].tolist()

    time_trials = [ ]
    for img_num in file_nums:
        t0 = time.time()
        # left-right stereo pairs
        left_file = img_dir + f'left-{img_num:04d}.png'
        right_file = img_dir + f'right-{img_num:04d}.png'
        left_img = cv.imread( left_file, cv.IMREAD_ANYCOLOR )
        right_img = cv.imread( right_file, cv.IMREAD_ANYCOLOR )

        # image read check
        if (left_img is None) or (right_img is None):
            print( f'Passing stereo pair number: {img_num:04d}. Stereo pair not found.' )
            print()
            continue

        # if

        # stereo image processing
        # # blackout regions
        if k < 2.0:
            bor_l = [ [ [ 0, left_img.shape[ 1 ] // 2 ], [ -1, -1 ] ] ]
            bor_r = [ [ [ 0, right_img.shape[ 1 ] // 2 ], [ -1, -1 ] ] ]

        # if

        elif k == 2.0:
            bor_l = [ [ [ 0, 0 ], [ -1, right_img.shape[ 1 ] // 3 ] ],
                      [ [ 0, 2 * right_img.shape[ 1 ] // 3 ], [ -1, -1 ] ] ]
            bor_r = [ [ [ 0, 0 ], [ -1, right_img.shape[ 1 ] // 3 ] ],
                      [ [ 0, 2 * right_img.shape[ 1 ] // 3 ], [ -1, -1 ] ] ]

        # elif

        elif k <= 4.5:
            bor_l = [ [ [ 0, 0 ], [ -1, left_img.shape[ 1 ] // 3 ] ] ]
            bor_r = [ [ [ 0, 0 ], [ -1, right_img.shape[ 1 ] // 3 ] ] ]

        # elif

        else:
            bor_l = [ ]
            bor_r = [ ]

        # else

        roi_l = tuple( rois_l[ img_num ] )
        roi_r = tuple( rois_r[ img_num ] )

        # ============================= OLD ====================================
        # pts_3d, *_, proc_images, figures = needle_jig_reconstruction( left_img, right_img, stereo_params,
        #                                                               bor_l = bor_l, bor_r = bor_r,
        #                                                               roi_l = roi_l, roi_r = roi_r,
        #                                                               alpha = 0.5, recalc_stereo = True,
        #                                                               proc_show = proc_show )
        # =======================================================================

        pts_3d, *_, proc_images, figures = needle_jig_reconstruction_refined( left_img, right_img, stereo_params,
                                                                              bor_l=bor_l, bor_r=bor_r,
                                                                              roi_l=roi_l, roi_r=roi_r,
                                                                              alpha=0.5, recalc_stereo=True,
                                                                              proc_show=proc_show, zoom=3,
                                                                              winsize=(31, 31) )

        dt = time.time() - t0
        time_trials.append( dt )
        # save the processed images
        if save_dir:
            print( 'Saving figures and files...' )
            save_fbase = save_dir + f"left-right-{img_num:04d}" + "_{:s}"

            # save the 3-D points
            np.savetxt( save_fbase.format( '3d-pts' ) + '.txt', pts_3d )
            print( 'Saved data file:', save_fbase.format( '3d-pts' ) + '.txt' )

            # save the processed images
            for key, imgs in proc_images.items():
                #                 if imgs[0].ndim == 3:
                #                     cv.imwrite( save_fbase.format( key ) + '.png',
                #                                 imconcat( ( 255 * imgs[0] / imgs[0].max() ).astype( np.uint8 ),
                #                                           ( 255 * imgs[1] / imgs[1].max() ).astype( np.uint8 ), ( 0, 0, 255 ) ) )
                #
                #                 # if
                #
                #                 elif imgs[0].ndim == 2:
                #                     cv.imwrite( save_fbase.format( key ) + '.png',
                #                                 imconcat( ( 255 * imgs[0] / imgs[0].max() ).astype( np.uint8 ),
                #                                           ( 255 * imgs[1] / imgs[1].max() ).astype( np.uint8 ), 125 ) )
                #
                #                 # elif
                cv.imwrite( save_fbase.format( key ) + '.png', imgs )
                print( 'Saved figure:', save_fbase.format( key ) + '.png' )

                # for: images

            # save the figures
            for key, fig in figures.items():
                fig.savefig( save_fbase.format( key + '-fig' ) + '.png' )
                print( 'Saved figure:', save_fbase.format( key + '-fig' ) + '.png' )

            # for: figures

            print( 'Finished saving files and figures.', end='\n\n' + 80 * '=' + '\n\n' )

        # if

        print( f'Completed stereo pair {img_num:04d}.' )
        print( f'Time for Trial: {round( dt / 60 )} mins. {dt % 60:.2f} s' )
        print( '\n' + 75 * '=', end='\n\n' )
        plt.close( 'all' )

    # for

    # display average time.
    if len( time_trials ) > 0:
        dt_avg = sum( time_trials ) / len( time_trials )
        print( f'Average time for all {round( dt_avg / 60 )} mins. {dt_avg % 60:.2f}' )

    # if


# main_needleval


if __name__ == '__main__':
    main()

# if __main__
