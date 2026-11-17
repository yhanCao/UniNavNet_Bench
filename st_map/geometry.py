"""Upstream voxel indexing helpers, independent of the GPU runtime."""
import numpy as np

def extractVoxelCenterfromVoxelmap(index,shape,minbound,size):
    indices_3=np.array(np.unravel_index(index, shape,order='C')).T
    coordinates = (indices_3 + 0.5) * size + np.array(minbound)
    return coordinates

def extractVoxelIndex3(index,shape):
    indices_3=np.array(np.unravel_index(index, shape,order='C')).T
    return indices_3


def rotation_from_euler(roll=0, pitch=0, yaw=0):
        """
        Get rotation matrix
        Args:
            roll, pitch, yaw:       In radians
        Returns:
            R:          [4, 4]
        """
        si, sj, sk = np.sin(roll), np.sin(pitch), np.sin(yaw)
        ci, cj, ck = np.cos(roll), np.cos(pitch), np.cos(yaw)
        cc, cs = ci * ck, ci * sk
        sc, ss = si * ck, si * sk

        R = np.identity(4)
        R[0, 0] = cj * ck  # cos(pitch) * cos(yaw)
        R[0, 1] = sj * sc - cs  # sin(pitch) * sin(roll) * cos(yaw) - cos(roll) * sin(yaw)
        R[0, 2] = sj * cc + ss
        R[1, 0] = cj * sk
        R[1, 1] = sj * ss + cc
        R[1, 2] = sj * cs - sc
        R[2, 0] = -sj
        R[2, 1] = cj * si
        R[2, 2] = cj * ci
        return R

def X_w_to_X_v(R,position,points):
    """
    Transform points from world coordinates to camera coordinates
    Args:
        angle:     [3, 3]
        roll:      [3, 3]
        position:  [3, 1]
        points:    [N, 3]
    Returns:
        points:    [N, 3]
    """
    points = points-position
    points_camera = np.dot(points,R[:3,:3])
    cam_coords = np.array([points_camera[:, 0], -points_camera[:, 2], points_camera[:, 1]])
    return cam_coords
