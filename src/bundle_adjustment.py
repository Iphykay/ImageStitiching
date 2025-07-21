module_name = 'Bundle Ajustment'

'''
Version: v1.0.0

Description:
    Helps in getting the refined 3D coorinates described the scene geometry.
    It takes both intrisic and extrinsic parameters to minimize the reprojection
    error.
    https://en.wikipedia.org/wiki/Bundle_adjustment

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
from cam_state   import state

# OTHER IMPORTS
from numpy       import zeros, linalg, hstack, array, subtract, sqrt, mean, float64, identity, cross, multiply, power, copy
from ordered_set import OrderedSet
from random      import normalvariate
from utils       import PARAMS_PER_CAMERA, PARAMS_PER_POINT_MATCHES, REGULARISATION_PARAM, MAX_ITR, FOCAL_DERIVATIVE, PPX_DERIVATIVE, PPY_DERIVATIVE

# USER INTERFACE


class bundle_adjustment:
    def __init__(self):
        self._matches    = []
        self.match_count = []
        self._cameras    = OrderedSet()
    #

    def matches(self):
        return self._matches

    def _skew_matrix(self, v):
        '''
        Skew symmetric matrix also known as 
        cross-product matrix

        Input:
        ------
        v: 3D vector containing [x, y, z]

        Output:
        ------
        Returns skew matrix
        '''
        x, y, z = v
        return array([[0, -z, y], [z, 0, -x], [-y, x, 0]], 
                     dtype=float64)
    #

    def _get_match_H(self, cam_pt_from, cam_pt_to):
        '''
        Get the extrinsic and intrinisic parameters 
        and computes the homography matrix

        Input:
        ------
        cam_pt_from: camera mapping from
        cam_pt_to  : camera mapping to

        Output:
        -------
        Returns Hmomography
        '''
        mat_K = cam_pt_from.K
        mat_R = cam_pt_from.R

        inv_K_from = linalg.pinv(cam_pt_to.K)
        inv_R_to   = cam_pt_to.R.T

        H_match    = (mat_K @ mat_R) @ (inv_R_to @ inv_K_from)

        return H_match
    #

    def _dR_dv(self, v, Rot_mat):
        '''
        Change of the rotation in the 
        x, y and z direction

        Input:
        ------
        v       : 3D vector containing [x, y, z]
        Rot_mat : Rotation matrix
        '''
        x, y, z = v
        
        if linalg.norm(v) < 1e-14:
            return array([self._skew_matrix([1, 0, 0]),
                          self._skew_matrix([0, 1, 0]),
                          self._skew_matrix([0, 0, 1])])
        # if

        dRx = self._skew_matrix(v) * x
        dRy = self._skew_matrix(v) * y
        dRz = self._skew_matrix(v) * z

        dRv = [dRx, dRy, dRz]

        # Identity - Rotation
        I_minus_R = identity(3) - Rot_mat

        for v_id in range(len(v)):
            x_, y_, z_ = cross(array(v),I_minus_R[:,v_id])
            dRv[v_id]  += self._skew_matrix([x_, y_, z_])
            dRv[v_id]  = multiply(dRv[v_id], 1/linalg.norm(v))
            dRv[v_id]  = dRv[v_id] @ Rot_mat
        # for
        return dRv
    #

    def _make_homogenous(self, pt):
        '''
        Converts coordinate points to homogenous points

        Input:
        ------
        pt: cartesian coordinate

        Output:
        -------
        Returns homogenous coordinate
        '''
        return hstack([pt, [1]])
    #

    def _camera_to_image_coordinate(self, pt, H, usedivide=None):
        '''
        Projects a point using the Homography back to 
        cartesian coordinate

        Input:
        ------
        pt: cartesian coordinate
        H : Homography matrix
        '''
        if len(pt) == 2:
            get_hom_coord = self._make_homogenous(pt)
            proj          = H @ get_hom_coord
            if usedivide:
                return array([proj[0]/proj[2], proj[1]/proj[2]])
        else:
            proj = H @ pt
        # if
        return proj
    #

    def _dH_homo_coord(self, dhdv, homo, hz_inv, hz_sqr_inv):
        '''
        Computes the derivatives of the homogenous coordinates 

        Input:
        ------
        dhdv      : partial derivative of the homogenous vector [hx, hy, hz] w.r.t with vector v
        homo      : 2D inhomogenous point
        hz_inv    : 1/hz
        hz_sqr_inv: 1/hz**2
        '''
        return array([-dhdv[0] * hz_inv + dhdv[2] * homo[0] * hz_sqr_inv,
                      -dhdv[1] * hz_inv + dhdv[2] * homo[1] * hz_sqr_inv],
                      dtype=float64)
    #


    def _reprojection_error(self, state):
        '''
        Computes the reprojection error of the extrinisic and 
        intrinsic parameters 

        Input:
        -----
        state

        Output:
        ------
        Returns reprojection error
        '''
        current_camera_state = state.cameras #cameras added

        # Get the number of inlier points from each image
        pairwise_matches = sum(len(match.inliers) for match in self._matches)
        reproj_error     = zeros((pairwise_matches * PARAMS_PER_POINT_MATCHES))

        for match in self._matches:
            # cam pt image --> cam pt image 2
            cam_pt_from = current_camera_state[self._cameras.index(match.cam_from)]
            cam_pt_to   = current_camera_state[self._cameras.index(match.cam_to)]

            # Get the extrinisic and intrinsic paramters
            H_match = self._get_match_H(cam_pt_from, cam_pt_to)

            # Get the points in the inliers
            for num_id, pt in zip(range(0, len(match.inliers), 2), match.inliers):
                from_pt_coord = pt[0]
                to_pt_coord   = pt[1]

                # Get the projected cartesian coordinate
                pixel_coord            = self._camera_to_image_coordinate(to_pt_coord, H_match, usedivide=True)
                reproj_error[num_id]   = subtract(from_pt_coord[0], pixel_coord[0])
                reproj_error[num_id+1] = subtract(from_pt_coord[1], pixel_coord[1])
            # for 

            print(f"Error: {sqrt(mean(reproj_error**2))}, Match from {match.cam_from.image.filename} to {match.cam_to.image.filename}:")
        # for
        return reproj_error
    #

    def _solve_jacobian(self, state):
        '''
        Solves the Jacobian matrix
        '''
        
        # get the parameters and cameras
        params  = state.params
        cameras = state.cameras

        num_cams = len(cameras)
        num_pointwise_matches = sum(len(match.inliers) for match in self._matches)

        J   = zeros((PARAMS_PER_POINT_MATCHES * num_pointwise_matches, PARAMS_PER_CAMERA * num_cams), dtype=float64)
        JtJ = zeros((PARAMS_PER_CAMERA * num_cams, PARAMS_PER_CAMERA * num_cams), dtype=float64)

        all_dRdv = []

        for cam_id in range(len(cameras)):
            num_param = cam_id * PARAMS_PER_CAMERA # Have a feeling this has to change

            # Get the x, y, z coordinates from the camera
            x, y, z = params[num_param+3:num_param+6]

            # Get the change of direction in the x, y , z direction with the rotation
            dRdv = self._dR_dv([x,y,z], cameras[cam_id].R)
            all_dRdv.append(dRdv)
        # for

        for (i, match) in enumerate(self._matches):
            num_match_count_idx = self.match_count[i] * 2

            # Get the camera from and camera to
            cam_from = cameras[self._cameras.index(match.cam_from)]
            cam_to   = cameras[self._cameras.index(match.cam_to)]

            params_from_cam = self._cameras.index(match.cam_from) * PARAMS_PER_CAMERA
            params_to_cam   = self._cameras.index(match.cam_to) * PARAMS_PER_CAMERA

            H_cam     = self._get_match_H(cam_from, cam_to)
            dR_from_v = all_dRdv[self._cameras.index(match.cam_from)]
            dR_to_v   = copy(all_dRdv[self._cameras.index(match.cam_to)])
            dR_to_vT  = [m.T for m in dR_to_v]

            for (pair_id, pair) in enumerate(match.inliers):
                coord = pair[1]
                new_coord = self._camera_to_image_coordinate(coord, H_cam, usedivide=False)
                hz_sqr_inv = 1.0/ power(new_coord[2],2)
                hz_inv     = 1.0 / new_coord[2]

                dFrom = zeros((PARAMS_PER_CAMERA, PARAMS_PER_POINT_MATCHES))
                dTo   = zeros((PARAMS_PER_CAMERA, PARAMS_PER_POINT_MATCHES))

                homo_m = cam_from.R @ cam_to.R.T @ linalg.pinv(cam_to.K)
                dot_u2 = self._camera_to_image_coordinate(coord, homo_m, usedivide=False)

                # Get the array for dFrom
                dFrom[0] = self._dH_homo_coord(self._camera_to_image_coordinate(dot_u2, FOCAL_DERIVATIVE, usedivide=False),
                                              new_coord,hz_inv, hz_sqr_inv)
                dFrom[1] = self._dH_homo_coord(self._camera_to_image_coordinate(dot_u2, PPX_DERIVATIVE, usedivide=False),
                                              new_coord,hz_inv, hz_sqr_inv)
                dFrom[2] = self._dH_homo_coord(self._camera_to_image_coordinate(dot_u2, PPY_DERIVATIVE, usedivide=False), 
                                              new_coord,hz_inv, hz_sqr_inv)
                
                dot_u2 = self._camera_to_image_coordinate(coord, (cam_to.R.T @ linalg.pinv(cam_to.K)), usedivide=False)
                dFrom[3] = self._dH_homo_coord(self._camera_to_image_coordinate(dot_u2,(cam_from.K @ dR_from_v[0]), usedivide=False), 
                                              new_coord,hz_inv, hz_sqr_inv)
                dFrom[4] = self._dH_homo_coord(self._camera_to_image_coordinate(dot_u2,(cam_from.K @ dR_from_v[1]), usedivide=False), 
                                              new_coord,hz_inv, hz_sqr_inv)
                dFrom[5] = self._dH_homo_coord(self._camera_to_image_coordinate(dot_u2,(cam_from.K @ dR_from_v[2]), usedivide=False), 
                                              new_coord,hz_inv, hz_sqr_inv)
                
                # Get the array for dTo
                dot_u2 = -1 * self._camera_to_image_coordinate(coord, linalg.pinv(cam_to.K), usedivide=False)
                dTo[0] = self._dH_homo_coord(self._camera_to_image_coordinate(dot_u2,(H_cam @ FOCAL_DERIVATIVE), usedivide=False),
                                            new_coord, hz_inv, hz_sqr_inv)
                dTo[1] = self._dH_homo_coord(self._camera_to_image_coordinate(dot_u2,(H_cam @ PPX_DERIVATIVE), usedivide=False), 
                                            new_coord, hz_inv, hz_sqr_inv)
                dTo[2] = self._dH_homo_coord(self._camera_to_image_coordinate(dot_u2,(H_cam @ PPY_DERIVATIVE), usedivide=False), 
                                            new_coord, hz_inv, hz_sqr_inv)
                
                hMtrx = cam_from.K @ cam_from.R
                dot_u2 = self._camera_to_image_coordinate(coord, linalg.pinv(cam_to.K))
                dTo[3] = self._dH_homo_coord(self._camera_to_image_coordinate(dot_u2,(hMtrx @ dR_to_vT[0]), usedivide=False), 
                                            new_coord, hz_inv, hz_sqr_inv)
                dTo[4] = self._dH_homo_coord(self._camera_to_image_coordinate(dot_u2,(hMtrx @ dR_to_vT[1]), usedivide=False), 
                                            new_coord, hz_inv, hz_sqr_inv)
                dTo[5] = self._dH_homo_coord(self._camera_to_image_coordinate(dot_u2,(hMtrx @ dR_to_vT[2]), usedivide=False), 
                                            new_coord, hz_inv, hz_sqr_inv)
                
                for param_idx in range(PARAMS_PER_CAMERA):
                    J[num_match_count_idx, params_from_cam + param_idx]   = dFrom[param_idx][0]
                    J[num_match_count_idx, params_to_cam + param_idx]     = dTo[param_idx][0]
                    J[num_match_count_idx+1, params_from_cam + param_idx] = dFrom[param_idx][1]
                    J[num_match_count_idx+1, params_to_cam + param_idx]   = dTo[param_idx][1]
                # for

                for param_idx_i in range(PARAMS_PER_CAMERA):
                    for param_idx_j in range(PARAMS_PER_CAMERA):
                        i1          = params_from_cam + param_idx_i
                        i2          = params_to_cam + param_idx_j
                        val         = dFrom[param_idx_i] @ dTo[param_idx_j]
                        JtJ[i1][i2] += val
                        JtJ[i2][i1] += val
                    # for 
                # for

                for param_idx_i in range(PARAMS_PER_CAMERA):
                    for param_idx_j in range(param_idx_i, PARAMS_PER_CAMERA):
                        i1          = params_from_cam + param_idx_i
                        i2          = params_from_cam + param_idx_j
                        val         = dFrom[param_idx_i] @ dFrom[param_idx_j]
                        JtJ[i1][i2] += val

                        if (param_idx_i != param_idx_j):
                            JtJ[i2][i1] += val
                        # if
                    
                        i1          = params_to_cam + param_idx_i
                        i2          = params_to_cam + param_idx_j
                        val         = dTo[param_idx_i] @ dTo[param_idx_j]
                        JtJ[i1][i2] += val

                        if (param_idx_i != param_idx_j):
                            JtJ[i2][i1] += val
                        # if
                    # for
                # for
                num_match_count_idx += 2
            # for
        return J, JtJ
    #

    def run_ba(self):
        '''
        Runs the bundle adjusment class
        '''
        
        if len(self._matches) < 1:
            raise ValueError('Must have at least one match')
        # if

        print('Running Bundle Adjustment...')

        # Get the initial state of the camera for the images
        initial_state = state()
        initial_state.set_initial_cameras(self._cameras)

        # Get the residual
        init_residual = self._reprojection_error(initial_state)
        init_error    = sqrt(mean(power(init_residual,2)))

        print(f'Initial_error: {init_error}')

        itr_count          = 0
        non_decrease_count = 0
        best_state         = initial_state
        best_residuals     = init_residual
        best_error         = init_error

        while (itr_count < MAX_ITR):

            J, JtJ       = self._solve_jacobian(best_state)
            param_update = self._get_next_update(J, JtJ, best_residuals)
            next_state   = best_state.updatedState(param_update)

            next_residuals = self._reprojection_error(next_state)
            next_error_val = sqrt(mean(next_residuals**2))
            print(f'Next error: {next_error_val}')

            if (next_error_val >= best_error - 1e-3):
                non_decrease_count += 1
            else:
                print('Updating state to new best state')
                non_decrease_count = 0
                best_error = next_error_val

                best_state     = next_state
                best_residuals = next_residuals
            # else
        
            if (non_decrease_count > 5):
                break
            # if
        # while

        print(f'BEST ERROR {best_error}')

        # Update actual camera object params
        new_cameras = best_state.cameras
        for cam_id in range(len(new_cameras)):
            print(f'Final focal: {new_cameras[cam_id].focal}')
            self._cameras[cam_id].focal = new_cameras[cam_id].focal
            self._cameras[cam_id].ppx   = new_cameras[cam_id].ppx
            self._cameras[cam_id].ppy   = new_cameras[cam_id].ppy
            self._cameras[cam_id].R     = new_cameras[cam_id].R
        # for
    #

    def _get_next_update(self, J, JtJ, residuals):

        # # Regularisation
        l = normalvariate(1, 0.1)
        for i in range(len(self._cameras) * PARAMS_PER_CAMERA):
            if (i % PARAMS_PER_CAMERA >= 3):
                # TODO: Improve regularisation params (currently a bit off)
                JtJ[i][i] += (3.14/16) * l #random.normalvariate(10, 20) * 5000000000
            else:
                JtJ[i][i] += (1500 / 10) * l # TODO: Use intial focal estimate #random.normalvariate(10, 20) * 5000000000
            # if
        # for

        b = J.T @ residuals

        updates = linalg.solve(JtJ, b)
        
        return updates
    #

    def add(self, match):
        '''
        Add a match to the bundle adjuster
        '''
        num_pointwise_matches = sum(len(match.inliers) for match in self._matches)
        self.match_count.append(num_pointwise_matches)

        self._matches.append(match)
        for cam in match.cams():
            self._cameras.add(cam)
        # for

        print(f'Added match {match}')
    #

    def added_cameras(self):
        return self._cameras
    #







