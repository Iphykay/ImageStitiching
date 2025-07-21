module_name = 'Camera_estimator'

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

# CUSTOM IMPORTS
from bundle_adjustment import bundle_adjustment

# OTHER IMPORTS
from numpy    import median, identity, linalg
import pickle as pckl

# USER INTERFACE


class camera_estimator:
    def __init__(self, matches):
        self._matches = matches

        # Run the estmation
        self._estimation()
    #

    def _estimation(self):
        # Get the focal length
        self._get_focal_length()
        use_order = self._span_trees()

        # do bundle
        self._use_bundle_adjustment(use_order)

        return self._all_cameras
    #


    def _all_cameras(self):
        '''
        Adds all image frame correspondence 
        '''
        all_cameras_matches = set()

        # Loop for over the matches
        for match in self._matches:
            all_cameras_matches.add(match.cam_from)
            all_cameras_matches.add(match.cam_to)
        # for
        
        return all_cameras_matches
    #

    def _normalize_match_H(self, match):
        match.H = match.H * (1.0/ match.H[2,2])
    #

    def _span_trees(self):
        connected_nodes = set()

        # Get all the cameras of the images
        allCamers    = self._all_cameras()
        sorted_edges = sorted(self._matches, key=lambda m: len(m.inliers),
                              reverse=True)
        
        [print(f"{e.cam_from.image.filename},{e.cam_to.image.filename}: {len(e.inliers)}") for e in sorted_edges]

        # Get the best edge to join the images
        best_edge = sorted_edges.pop(0)

        print(f'Best edge: {best_edge.cam_from.image.filename} - {best_edge.cam_to.image.filename}: {len(best_edge.inliers)}')
        print(f'Best edge H: {best_edge.H}')

        add_order = [best_edge]
        connected_nodes.add(best_edge.cam_from)
        connected_nodes.add(best_edge.cam_to)

        while (len(connected_nodes) < len(allCamers)):
            for (i, match) in enumerate(sorted_edges):
                if (match.cam_from in connected_nodes):
                    # Add node as is
                    edge = sorted_edges.pop(i)

                    # Normalize the homography
                    self._normalize_match_H(edge)

                    add_order.append(edge)
                    connected_nodes.add(edge.cam_from)
                    connected_nodes.add(edge.cam_to)
                    break
                # if
            # for
        # while
            
        return add_order
    #

    def _get_focal_length(self):
        focal_length = []
        for match in self._matches:
            estimate_focal_length = match.estimate_focal_from_homography()

            if estimate_focal_length != 0:
                focal_length.append(estimate_focal_length)
                print(f"Estimate from {match.cam_from.image.filename} to {match.cam_to.image.filename}: {estimate_focal_length}",
                      flush=True)
            # if
        # for
        
        # Get the median focal length of the images
        median_focal_length = median(focal_length)

        print(f'Focal length is {median_focal_length}', flush=True)

        for camera in self._all_cameras():
            camera.focal = median_focal_length
        # for
    #

    def _use_bundle_adjustment(self, add_order):
        '''
        Iteratively add each match to the bundle adjuster
        '''

        ba = bundle_adjustment()

        other_matches = set(self._matches) - set(add_order)

        identity_cam   = add_order[0].cam_from
        identity_cam.R = identity(3)
        identity_cam.ppx, identity_cam.ppy = 0, 0

        print(f'Identity cam: {identity_cam.image.filename}')

        print('Original match params:')
        for match in add_order:
            print(f'{match.cam_from.image.filename} to {match.cam_to.image.filename}:\n {match.cam_to.R}\n')
            print('------------------')
        # for

        for match in add_order:
            print(f'match.cam_from.R: {match.cam_from.R}')
            print(f'match.cam_from.K: {match.cam_from.K}')
            print(f'match.H: {match.H}')
            print(f'match.cam_to.K: {match.cam_to.K}')

            match.cam_to.R = (match.cam_from.R.T @ (linalg.pinv(match.cam_from.K) @ match.H @ match.cam_to.K)).T
            match.cam_to.ppx, match.cam_to.ppy = 0, 0
            print(f'{match.cam_from.image.filename} to {match.cam_to.image.filename}:\n {match.cam_to.R}\n')

            ba.add(match)

            added_cams = ba.added_cameras()
            to_add = set()
            for other_match in other_matches:
                # If both cameras already added, add the match to BA
                if (other_match.cam_from in added_cams and other_match.cam_to in added_cams):
                    to_add.add(other_match)
                # if
            # for

            for match in to_add:
                # self._reverse_match(match)
                ba.add(match)
                other_matches.remove(match)
            # for
        # for
            
        all_cameras = None
        try:
            all_cameras = pckl.load(open(f'all_cameras_{len(self._all_cameras())}.p', 'rb'))

            for match in self._matches:
                for cam in all_cameras:
                    if (match.cam_to.image.filename == cam.image.filename):
                        match.cam_to = cam
                    elif (match.cam_from.image.filename == cam.image.filename):
                        match.cam_from = cam
                    # if
                # for
            # for
        except (OSError, IOError):    
            ba.run_ba()
            all_cameras = self._all_cameras()
            # pckl.dump(all_cameras, open(f'all_cameras_{len(self.all_cameras())}.p', 'wb'))

            print('BA complete.')
        # try
            




    

    