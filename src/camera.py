module_name = ''

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

# OTHER IMPORTS

# CUSTOM IMPORTS


import numpy as np
import math
from scipy.spatial.transform import Rotation

class Camera:

  focal = 1
  ppx = 0
  ppy = 0
  R = None

  # Constructor
  def __init__(self, image):
    self._image = image

  @property
  def image(self):
    return self._image

  @property
  def K(self):
    I = np.identity(3, dtype=np.float64)
    I[0][0] = self.focal
    I[0][2] = self.ppx
    I[1][1] = self.focal
    I[1][2] = self.ppy
    return I

  def angle_parameterisation(self):
    u,s,v = np.linalg.svd(self.R)
    R_new = u @ (v) 
    if (np.linalg.det(R_new) < 0):
      R_new *= -1
    # if
    
    rx = R_new[2][1] - R_new[1][2]
    ry = R_new[0][2] - R_new[2][0]
    rz = R_new[1][0] - R_new[0][1]

    s = math.sqrt(rx**2 + ry**2 + rz**2)
    if (s < 1e-7):
      rx, ry, rz = 0, 0, 0
    else:
      cos = (R_new[0][0] + R_new[1][1] + R_new[2][2] - 1) * 0.5
      if (cos > 1):
        cos = 1
      elif (cos < -1):
        cos = -1
      # if
      
      theta = np.arccos(cos)
      mul = 1 / s * theta
      rx *= mul
      ry *= mul
      rz *= mul
    # if

    return np.array([rx, ry, rz], dtype=np.float64)


  def rotvec_to_matrix(self, rotvec):
    rotation = Rotation.from_rotvec(rotvec)
    return rotation.as_matrix()
