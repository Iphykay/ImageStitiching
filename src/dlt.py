module_name = 'DLT'

'''
Version: v1.0.0

Description:
    Master module

Authors:
    Iphy Kelvin

Date Created     : 05/20/2025
Date Last Updated: 08/06/2025

Doc:
    <***>

Notes:
    <***>

ToDo:

'''

# OTHER IMPORTS

# CUSTOM IMPORTS
from numpy import array, nan, empty, mean, sum, power, sqrt, vstack, ones, linalg, asarray


def compute_A_matrix(pt1, pt2):
    """
    Computes the 2n X 9 matrix A responsible for 
    setting up the linear equations that allows 
    for the homography to be found.

    Input:
    ------
    pts: Corresonding points

    Output:
    -------
    A: 2n x 9 matrix 
    """

    A = []

    for i in range (0, len(pt1)):
        x, y             = pt1[i]
        x_prime, y_prime = pt2[i]

        A.append([0, 0, 0, -x, -y, -1, y_prime * x, y_prime * y, y_prime])
        A.append([x, y, 1, 0, 0, 0, -x_prime * x, -x_prime * y, -x_prime])
    # for
    
    return asarray(A)
#

def normalize_point(pts):
    """
    Normalizes the points

    Input:
    ------
    pts: 2D points ([(x1,y1),(x2,y2),(x3,y3)...])

    Output:
    ------

    """
    # Get the mean of the points
    x_mean, y_mean = mean(pts, axis=0)

    # sum of distance between the mean and points
    total_dist = sum(power((power((pt[0]-x_mean),2)+power((pt[1]-y_mean),2)),0.5) for pt in pts)

    # average the distance
    avg_dist = total_dist / len(pts)

    # scale factor
    scale_f = sqrt(2) / avg_dist

    # Normalization matrix with translation
    projtve_matrix = array([[scale_f, 0, -scale_f*x_mean],
                           [0, scale_f, -scale_f*y_mean],
                            [0, 0, 1]])
    
    # Pad the pts 
    new_pts = vstack((pts.T, ones((1,pts.shape[0]))))

    # Multiply the new pt witht the projective translation matrix
    pts_2d = projtve_matrix @ new_pts

    # Get the x and y coordinates
    pts_xy = pts_2d[0:2].T

    return pts_xy, projtve_matrix

def use_dlt(pts1, pts2):
    """
    Computes the homography matrix that transforms pts1 onto the 
    plane of pts2 via direct linear transform method.
    """

    # Use the nomralize function
    pt1_norm, pt1_pm = normalize_point(pts1)
    pt2_norm, pt2_pm = normalize_point(pts2)

    # compute the A matrix
    A_matrix = compute_A_matrix(pt1_norm, pt2_norm)

    # Compute the SVD of A matrix
    U,S,Vh = linalg.svd(A_matrix)

    # Get the last column of V and normalize the last value
    H_matrix = Vh[-1,:] / Vh[-1,-1]

    # Reshape the homography matrix
    H = H_matrix.reshape(3,3)

    # Denormalize
    H = linalg.inv(pt2_pm) @ H @ pt1_pm

    return H








