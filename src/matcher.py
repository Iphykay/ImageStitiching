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

# Constants
FLANN_INDEX_KDTREE = 0


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
    #

    def get_matches(self):
        '''
        Gets the pairwise matches between two images
        '''
        index_params  = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann_matcher = cv.FlannBasedMatcher(index_params, search_params)

        # Find good matches
        save_img_pairs = dict()
        pairs          = []

        # Get the keypoints and descriptors of the images
        for img in self._imgs:
            self.all_keypts.append(img.keypt)
            self.all_descptrs.append(img.descptr)
        # for 

        # Match the descriptors for the image/images in pairs
        for id_img in range(0, len(self._imgs)):
            flann_matcher.clear()

            # Get the train and query descriptors
            # The flann_train_descptr holds all the descriptors except the one being queried
            # The flann_query_descptr holds the descriptors of the image being queried
            flann_train_descptr = [x for j,x in enumerate(self.all_descptrs) if j != id_img]
            flann_query_descptr = self.all_descptrs[id_img]
            matching_pairs      = [j for j, x in enumerate(self.all_descptrs) if j != id_img]
            matching_pairs.append(id_img); pairs.append(matching_pairs)

            # Train the descriptos
            flann_matcher.add(flann_train_descptr)
            flann_matcher.train()

            # Get the matches
            matches = flann_matcher.knnMatch(flann_query_descptr, k=4)

            # Savee the potential pairs
            potential_pairs = zeros((len(self._imgs),len(flann_query_descptr)), dtype=int)

            # Get the query idx from the image being queried
            for idx, pt_neigbors in enumerate(matches):
                for pt_idx in reversed(pt_neigbors):
                    get_query_img_idx = pt_idx.imgIdx if pt_idx.imgIdx < id_img else pt_idx.imgIdx + 1
                    potential_pairs[get_query_img_idx][idx] = pt_idx.trainIdx
                # for
            # for
            
            # Save the pairs
            save_img_pairs[tuple(matching_pairs)] = potential_pairs
            # save_matches.append(good_matches)
        # for
        return save_img_pairs
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
        for queryIdx, (pairIdx, potential_pairs) in enumerate(matches.items()):
            for targetIdx in pairIdx[:-1]:

                # Lets track the query image and the target image
                query_to_target_imgs = [(match.cam_from.image.filename, match.cam_to.image.filename) for match in goodmatches]
                query_img            = self._imgs[queryIdx].filename
                target_img           = self._imgs[targetIdx].filename

                if ((query_img, target_img) in query_to_target_imgs or (target_img, query_img) in query_to_target_imgs):
                    continue
                # if

                ptA = take(self.all_keypts[queryIdx], nonzero(potential_pairs[queryIdx][targetIdx])[0]).tolist()
                ptB = take(self.all_keypts[targetIdx], potential_pairs[queryIdx][targetIdx][nonzero(potential_pairs[queryIdx][targetIdx])]).tolist()

                kptA = array([x.pt for x in ptA], dtype=float32)
                kptB = array([x.pt for x in ptB], dtype=float32)

                # Get the homography matrix
                H, bestInliers = use_ransac(kptA, kptB, 500, 4)

                for i in range(len(bestInliers)):
                    bestInliers[i][0], bestInliers[i][1] = bestInliers[i][1], bestInliers[i][0]
                # for
                
                goodmatches.append(Match(self._cameras[queryIdx], self._cameras[targetIdx], H, bestInliers))
            # for
        # for
        return goodmatches
    #

    def run_matcher(self):
        if path.isfile("pair_wise_matches.pckl"):
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
        self._matches = pariwise_matches

        return pariwise_matches
    #
            






    

