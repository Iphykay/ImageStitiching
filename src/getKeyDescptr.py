module_name = 'GetKeyPointsAndDescriptors'

'''
Version: v1.0.0

Description:
    Gets the keypoint and descriptors of images

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
import cv2 as cv

# USER INTERFACE
# maxFeatures = 200


class sift_descriptor:

    keypt   = None
    descptr = None

    def __init__(self, img, filename, num_img):
        self._img         = img
        self._filename    = filename
        self.filedescptn  = {f'{self._filename}':num_img}
    #

    @property
    def image(self):
        return self._img
    #

    @property
    def filename(self):
        return self._filename
    #

    def create_keypt_descrptr(self):
        output_img = cv.cvtColor(self._img, cv.COLOR_BGR2GRAY)
        sift       = cv.SIFT_create()

        self.keypt, self.descptr = sift.detectAndCompute(output_img, None)

        print(f"Keypoint and Descriptors for {self._filename}", flush=True)
    #




