module_name = 'Utils'

'''
Version: v1.0.0

Description:
    Master module

Authors:
    Iphy Kelvin

Date Created     : 05/20/2025
Date Last Updated: 06/20/2025

Doc:
    <***>

Notes:
    <***>

ToDo:
'''

# OTHER IMPORTS
from numpy import array

# CONSTANTS
PARAMS_PER_POINT_MATCHES = 2
PARAMS_PER_CAMERA        = 6
REGULARISATION_PARAM     = 5
MAX_ITR                  = 50
FOCAL_DERIVATIVE         = array([[1,0,0],
                                  [0,1,0],
                                  [0,0,0]])

PPX_DERIVATIVE           = array([[0,0,1],
                                  [0,0,0],
                                  [0,0,0]])

PPY_DERIVATIVE           = array([[0,0,0],
                                  [0,0,1],
                                  [0,0,0]])

