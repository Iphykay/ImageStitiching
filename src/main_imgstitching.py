module_name = 'ImageStitching'

'''
Version: v1.0.0

Description:
    Master module

Authors:
    Iphy Kelvin

Date Created     : 05/20/2025
Date Last Updated: 08/07/2025

Doc:
    <***>

Notes:
    <***>

ToDo:

'''


# CUSTOM IMPORTS
from camera_estimator import camera_estimator as cam_est
from getKeyDescptr    import sift_descriptor as sift_desc
from matcher          import Matcher
from stitch_image     import Stitch

# OTHER IMPORTS
import os
import cv2 as cv


def read_files_dir(dir):

    # Reads all the images and stores them to a list
    imgs = []
    for img_id, filename in enumerate(sorted(os.listdir(dir))):
        
        # Read the file
        read_img = cv.imread(os.path.join(dir, filename))

        if read_img is not None:
            imgs.append(sift_desc(read_img, filename, img_id))
        # if
    # for
    return imgs


def main():
    
    # Read the images
    see_imgs = read_files_dir("C:/Users/Starboy/OneDrive/RIT/Courses/IPCV/Assignments/HW4/Images")

    # Get the keypoints and descriptors
    for img in see_imgs:
        img.create_keypt_descrptr()
    # for

    # 
    matches          = Matcher(see_imgs)
    getKeypt_matches = matches.run_matcher()

    # Get camera estimates
    cameraEsts = cam_est(getKeypt_matches)
    estimated_cams = cameraEsts._estimation()

    # Stitch the images
    stitch = Stitch(estimated_cams)
    stitch.run()

    cv.imshow('Result', stitch.stitched_img)
    cv.imwrite('./stitched_img.png', stitch.stitched_img) 
    cv.waitKey(0)
#




if __name__ == '__main__':
    main()





