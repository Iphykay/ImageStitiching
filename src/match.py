module_name = 'match'

'''
Version: v1.0.0

Description:
    Master module

Authors:
    Iphy Kelvin

Date Created     : 06/20/2025
Date Last Updated: 06/20/2025

Doc:
    <***>

Notes:
    <***>

ToDo:
'''

# CUSTOM IMPORTS


# OTHER IMPORTS
from numpy import sqrt, isinf, abs


class Match:

  '''
  Represents a pairwise match between two images and the inlier points
  that represent a correspondence between the two images

  Notes: 
    Homography is taken as the transform of cam_from onto cam_to
    cam_from --H-> cam_to
    x(to) = H(to)(from) @ x(from)
  '''
  _homography = None 

  def __init__(self, cam_from, cam_to, h, inliers):
    self._cam_from   = cam_from
    self._cam_to     = cam_to
    self._homography = h
    self._inliers    = inliers
  #

  @property
  def cam_to(self):
    return self._cam_to
  #

  @cam_to.setter
  def cam_to(self, value):
    self._cam_to = value
  #

  @property
  def cam_from(self):
    return self._cam_from
  #
  
  @cam_from.setter
  def cam_from(self, value):
    self._cam_from = value
  #

  @property
  def H(self):
    return self._homography
  #

  @H.setter
  def H(self, value):
    self._homography = value
  #

  @property
  def inliers(self):
    return self._inliers
  #

  def cams(self):
    return [self._cam_to, self._cam_from]
  #
  
  def estimate_focal_from_homography(self):
    '''
    Calculates the estimate of the focal length
    using the homography matrix

    Input:
    ------
    None

    Output:
    ------
    Returns the focal length
    '''
    h = self._homography

    f1 = None
    f0 = None

    d1 = (h[2][0] * h[2][1])
    d2 = (h[2][1] - h[2][0]) * (h[2][1] + h[2][0])
    v1 = -(h[0][0] * h[0][1] + h[1][0] * h[1][1]) / d1
    v2 = (h[0][0] * h[0][0] + h[1][0] * h[1][0] - h[0][1] * h[0][1] - h[1][1] * h[1][1]) / d2
    if (v1 < v2): temp = v1; v1 = v2; v2 = temp
    if (v1 > 0 and v2 > 0): f1 = sqrt(v1 if abs(d1) > abs(d2) else v2)
    elif (v1 > 0): f1 = sqrt(v1)
    else: return 0

    d1 = h[0][0] * h[1][0] + h[0][1] * h[1][1]
    d2 = h[0][0] * h[0][0] + h[0][1] * h[0][1] - h[1][0] * h[1][0] - h[1][1] * h[1][1]
    v1 = -h[0][2] * h[1][2] / d1
    v2 = (h[1][2] * h[1][2] - h[0][2] * h[0][2]) / d2
    if (v1 < v2): temp = v1; v1 = v2; v2 = temp
    if (v1 > 0 and v2 > 0): f0 = sqrt(v1 if abs(d1) > abs(d2) else v2)
    elif (v1 > 0): f0 = sqrt(v1)
    else: return 0

    if (isinf(f1) or isinf(f0)):
      return 0

    return sqrt(f1 * f0)
