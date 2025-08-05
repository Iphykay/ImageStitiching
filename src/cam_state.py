module_name = 'Camera State'

'''
Version: v1.0.0

Description:
    Holds the diferent camera states and params 
    for each image

Authors:
    Iphy Kelvin

Date Created     : 06/20/2025
Date Last Updated: 07/20/2025

Doc:
    <***>

Notes:
    <***>

ToDo:
'''


# CUSTOM IMPORTS
from camera      import Camera

# OTHER IMPORTS
from ordered_set import OrderedSet
from numpy       import zeros, float64, copy, empty
from utils       import PARAMS_PER_CAMERA

# USER INTERFACE
idt = 1


class state:
    def __init__(self, params=empty(0, dtype=float64)):
        self._params          = params
        self.original_cameras = OrderedSet()
    #
    
    @property
    def params(self):
        return self._params
    #

    def set_initial_cameras(self, cameras):
        '''
        Get the initial cameras 
        '''
        self.original_cameras = cameras
        self.calculate_params(cameras)
    #

    def updatedState(self, update):
        '''
        Returns a new state object with the update applied
        '''
        updatedParams = copy(self._params)
        for i in range(len(self._params)):
            if (i < idt * 6 + 3 or i >= idt * 6 + 6):
                updatedParams[i] -= update[i]
            # if
        # for
        
        return state(updatedParams)

    def calculate_params(self, cameras):
        self._params = zeros((PARAMS_PER_CAMERA * len(cameras)), dtype=float64)

        for i in range(0, len(cameras) * PARAMS_PER_CAMERA, PARAMS_PER_CAMERA):
            camera                = cameras[i//PARAMS_PER_CAMERA]
            self._params[i]       = camera.focal
            self._params[i+1]     = camera.ppx
            self._params[i+2]     = camera.ppy
            self._params[i+3:i+6] = camera.angle_parameterisation()
        # for
    #

    @property
    def cameras(self):
        cameras = []

        for i in range(0, len(self._params), PARAMS_PER_CAMERA):
            new_camera       = Camera(None)
            new_camera.focal = self._params[i]
            new_camera.ppx   = self._params[i+1]
            new_camera.ppy   = self._params[i+2]
            m                = new_camera.rotvec_to_matrix(self._params[i+3:i+6])
            new_camera.R     = m
            cameras.append(new_camera)
        # for

        return cameras
    #