module_name = 'Matcher'

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
from ast         import literal_eval
import cv2       as cv
from camera      import Camera
from homo_ransac import use_ransac
from itertools   import combinations
from numpy       import array, zeros, float32
from match       import Match
from os          import path
from pickle      import dump, load

# USER INTERFACE
percentage = 0.6



class Matcher:
    def __init__(self, imgs):
        self._imgs         = imgs
        self._matches      = None
        self.all_keypts    = []
        self.all_descptrs  = []
        self._cameras      = [Camera(img) for img in self._imgs]
    #

    @property
    def matches(self):
        return self._matches


    def get_matches(self):
        '''
        Gets the pairwise matches between two images
        '''
        bf = cv.BFMatcher()

        # Find good matches
        save_matches = dict()

        # Get the keypoints and descriptors of the images
        for img in self._imgs:
            self.all_keypts.append(img.keypt)
            self.all_descptrs.append(img.descptr)
        # for 

        # Match the descriptors for the image/images in pairs
        for id_img in combinations((self._imgs),2):
            print(f"Matches between {list(id_img[0].filedescptn.keys())[0]} and {list(id_img[1].filedescptn.keys())[0]}", flush=True)

            matches = bf.knnMatch(id_img[0].descptr, 
                                  id_img[1].descptr, 
                                  k=2)

            # Get the good matches
            good_matches = array([[m] for m,n in matches if m.distance < percentage*n.distance])

            # Save the matches
            save_matches[f"({list(id_img[0].filedescptn.values())[0]},{list(id_img[1].filedescptn.values())[0]})"] = good_matches
            # save_matches.append(good_matches)
        # for
        return save_matches
    #
           

    def get_keypoint_matches(self, matches):
        '''
        Uses the KNN to match the keypoints

        Input:
        ------
        matches: matches between images 

        Output:
        -------
        Return good matches
        '''

        # Save the matches
        goodmatches = []

        # Initialize the matcher
        for match_key, match_value in matches.items():
            ptA = zeros((len(match_value),2), dtype=float32)
            ptB = zeros((len(match_value),2), dtype=float32)

            # Get the key numbers for the keyoints
            queryImg_num  = literal_eval(match_key)[0]
            targetImg_num = literal_eval(match_key)[1]

            for i, match_id in enumerate(match_value):
                ptA[i,:] = self.all_keypts[queryImg_num][match_id[0].queryIdx].pt
                ptB[i,:] = self.all_keypts[targetImg_num][match_id[0].trainIdx].pt
            # for

            # Get the homography matrix
            H, bestInliers = use_ransac(ptA, ptB, 1000, 4)

            goodmatches.append(Match(self._cameras[queryImg_num], self._cameras[targetImg_num], H, bestInliers))

            # Save the matches to an attribute
            self._matches = goodmatches

        return goodmatches
    #

    def run_matcher(self):
        if path.isfile("pairwise_matches.pckl"):
            try:
                with open(f"pairwise_matches.pckl", "rb") as read_matches:
                    pariwise_matches = load(read_matches)
                # with
                print("Loaded saved pairwise matches")
            except Exception as e:
                print(f"The pairwise matches has to be saved first: \n\t{e}", flush=True)
            # try
        else:
            matches          = self.get_matches()
            pariwise_matches = self.get_keypoint_matches(matches)

            # # Save the matches
            # with open(f"pairwise_matches.pckl","wb") as f:
            #     dump(pariwise_matches, f)
            # # with
        # try
        return pariwise_matches
    #
            





    