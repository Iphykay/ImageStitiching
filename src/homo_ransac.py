module_name = 'Ransac'

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
from dlt import use_dlt

# CUSTOM IMPORTS
from numpy import random, hstack, ones, power, std, argmin, array



def projected_error(pt1, pt2, H):
    """
    Calculates the projection error between the 
    pts and the homography

    Input:
    -----
    pts1, pts2: corresponding points
    H         : homography using DLT

    Output:
    -------
    Return projection error
    """

    pt1 = hstack([pt1, [1]])
    pt2 = hstack([pt2, [1]])

    projected_pt1 = ((H @ pt1).T).T

    sse_projcted = sum(power((pt2 - projected_pt1),2))

    return sse_projcted
#


def use_ransac(pts1, pts2, max_iterations, threshold):
    """
    Finds the best guess for the homography to map 
    pts3 onto the plane of pts1
    """

    bestinliers = []
    best_h      = None
    listH       = []
    stdListH    = []

    # Loop through the number of iterations
    for i_iter in range(max_iterations):
        # random points
        idx_pts = random.choice(len(pts1),4)

        # Get the samples using the random generated pts
        sample_pts1 = pts1[idx_pts]
        sample_pts2 = pts2[idx_pts]

        # Compute H using DLT
        H = use_dlt(sample_pts1, sample_pts2)

        # Get the right points
        inliers = []

        for i_pt in range(len(pts1)):
            # Distance for each correspondence point
            dist = projected_error(pts1[i_pt],pts2[i_pt],H)

            # Add a thrshold
            if (dist < threshold):
                inliers.append([pts1[i_pt],pts2[i_pt]])
            # if
        # for

        if (len(inliers) > len(bestinliers)):
            bestinliers = inliers
            best_h      = H

        # elif (len(inliers) and len(bestinliers)) == []:
        #     continue

        # elif ((len(inliers) and len(bestinliers)) != [] and (len(inliers) == len(bestinliers))):
        #     bestinliers.append(inliers)
        #     listH.append(H) # save the list of H 
        #     std_H       = std(H); stdListH.append(std_H)

        #     # Get the best H with the smallest std
        #     indexLowStd = argmin(array(stdListH))
        #     bestinliers = bestinliers[indexLowStd]
        #     best_h      = listH[indexLowStd]
        # if
    # for

    return best_h, bestinliers


