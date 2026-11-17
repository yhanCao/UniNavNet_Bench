# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for processing depth images.
"""
from argparse import Namespace
from typing import Optional
import itertools
import numpy as np
import torch
import math
import st_map._vendor.model.rotation_utils as ru
from torchvision.transforms.functional import hflip
from scipy.spatial.transform import Rotation as R
def get_camera_matrix(width, height, fov):
    """Returns a camera matrix from image size and fov."""
    xc = (width - 1.) / 2.
    zc = (height - 1.) / 2.
    f = (width / 2.) / np.tan(np.deg2rad(fov / 2.))
    camera_matrix = {'xc': xc, 'zc': zc, 'f': f}
    camera_matrix = Namespace(**camera_matrix)
    return camera_matrix
def get_intrinsic(width, height, fov):
    fov_rad = np.deg2rad(fov)

    # 计算焦距（focal length）
    focal_length = (width / 2) / np.tan(fov_rad / 2)

    # 构造内参矩阵
    intrinsics_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ])
    return intrinsics_matrix

def get_point_cloud_from_z(Y, camera_matrix, scale=1):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """
    x, z = np.meshgrid(np.arange(Y.shape[-1]),
                       np.arange(Y.shape[-2] - 1, -1, -1))
    for _ in range(Y.ndim - 2):
        x = np.expand_dims(x, axis=0)
        z = np.expand_dims(z, axis=0)
    X = (x[::scale, ::scale] - camera_matrix.xc) * \
        Y[::scale, ::scale] / camera_matrix.f
    Z = (z[::scale, ::scale] - camera_matrix.zc) * \
        Y[::scale, ::scale] / camera_matrix.f
    XYZ = np.concatenate((X[..., np.newaxis],
                          Y[::scale, ::scale][..., np.newaxis],
                          Z[..., np.newaxis]), axis=X.ndim)
    return XYZ


def transform_camera_view(XYZ, sensor_height, camera_elevation_degree):
    """
    Transforms the point cloud into geocentric frame to account for
    camera elevation and angle
    Input:
        XYZ                     : ...x3
        sensor_height           : height of the sensor
        camera_elevation_degree : camera elevation to rectify.
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix(
        [1., 0., 0.], angle=np.deg2rad(camera_elevation_degree))
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
    XYZ[..., 2] = XYZ[..., 2] + sensor_height
    return XYZ


def transform_pose(XYZ, current_pose):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : camera position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix([0., 0., 1.], angle=current_pose[2] - np.pi / 2.)
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
    XYZ[:, :, 0] = XYZ[:, :, 0] + current_pose[0]
    XYZ[:, :, 1] = XYZ[:, :, 1] + current_pose[1]
    return XYZ


def bin_points(XYZ_cms, map_size, z_bins, xy_resolution):
    """Bins points into xy-z bins
    XYZ_cms is ... x H x W x3
    Outputs is ... x map_size x map_size x (len(z_bins)+1)
    """
    sh = XYZ_cms.shape
    XYZ_cms = XYZ_cms.reshape([-1, sh[-3], sh[-2], sh[-1]])
    n_z_bins = len(z_bins) + 1
    counts = []
    for XYZ_cm in XYZ_cms:
        isnotnan = np.logical_not(np.isnan(XYZ_cm[:, :, 0]))
        X_bin = np.round(XYZ_cm[:, :, 0] / xy_resolution).astype(np.int32)
        Y_bin = np.round(XYZ_cm[:, :, 1] / xy_resolution).astype(np.int32)
        Z_bin = np.digitize(XYZ_cm[:, :, 2], bins=z_bins).astype(np.int32)

        isvalid = np.array([X_bin >= 0, X_bin < map_size, Y_bin >= 0,
                            Y_bin < map_size,
                            Z_bin >= 0, Z_bin < n_z_bins, isnotnan])
        isvalid = np.all(isvalid, axis=0)

        ind = (Y_bin * map_size + X_bin) * n_z_bins + Z_bin
        ind[np.logical_not(isvalid)] = 0
        count = np.bincount(ind.ravel(), isvalid.ravel().astype(np.int32),
                            minlength=map_size * map_size * n_z_bins)
        counts = np.reshape(count, [map_size, map_size, n_z_bins])

    counts = counts.reshape(list(sh[:-3]) + [map_size, map_size, n_z_bins])

    return counts

def bin_semantic_points(XYZ_cms, semantic, map_size, semantic_map_len, xy_resolution):
    """Bins points into xy-z bins
    XYZ_cms is ... x H x W x3
    semantic is ... x H x W
    Outputs is ... x map_size x map_size x len
    """
    # print("XYZ_cms: ",XYZ_cms.shape) #128*128*3
    # XYZ_cms = XYZ_cms.reshape([-1, sh[-3], sh[-2], sh[-1]])
    # print("XYZ_cms: ",XYZ_cms)
    # print("XYZ_cms: ",XYZ_cms.shape) #1*128*128*3
    # print("z_bins: ",z_bins) 
    # se = list(set(semantic.ravel()))
    # print("bin_semantic_points: ", se) # []
    # print("XYZ_cm:", XYZ_cm.shape) # 128*128*3
    isnotnan = np.logical_not(np.isnan(XYZ_cms[:, :, 0]))
    X_bin = np.round(XYZ_cms[:, :, 0] / xy_resolution).astype(np.int32)
    Y_bin = np.round(XYZ_cms[:, :, 1] / xy_resolution).astype(np.int32)
    Z_bin = semantic.astype(np.int32)

    # print("X_bin S count: ", Counter(X_bin.ravel()))
    # print("Y_bin S count: ", Counter(Y_bin.ravel()))
    # print("Z_bin S count: ", Counter(Z_bin.ravel()))
    # print("Z_bin: {}".format(X_bin))
    # print("X_bin: {}".format(X_bin.shape)) #128*128
    # print("Z_bin: {}".format(Z_bin.shape)) #128*128

    isvalid = np.array([X_bin >= 0, X_bin < map_size, Y_bin >= 0, Y_bin < map_size,
                        Z_bin > 0, Z_bin <= semantic_map_len, isnotnan])
    isvalid = np.all(isvalid, axis=0)

    # print("isvalid: ", Counter(isvalid.ravel()))
    # print("isvalid: ", isvalid.shape) #128*128

    ind = (Y_bin * map_size + X_bin) * semantic_map_len + Z_bin-1 
    ind[np.logical_not(isvalid)] = 0
    # print("ind: ", Counter(ind.ravel()))
    # print("ind: ", ind.shape) # 128*128
    count = np.bincount(ind.ravel(), isvalid.ravel().astype(np.int32),
                        minlength=map_size * map_size * semantic_map_len)
    # print("counts: ", count.shape) # 4838400

    counts = np.reshape(count, [map_size, map_size, semantic_map_len])
    # print("counts: ", counts.shape) # 480*480*semantic_map_len

    return counts


def get_point_cloud_from_z_t(Y_t, camera_matrix, device, scale=1):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """
    grid_x, grid_z = torch.meshgrid(torch.arange(Y_t.shape[-1]),
                                    torch.arange(Y_t.shape[-2] - 1, -1, -1))
    grid_x = grid_x.transpose(1, 0).to(device)
    grid_z = grid_z.transpose(1, 0).to(device)
    grid_x = grid_x.unsqueeze(0).expand(Y_t.size())
    grid_z = grid_z.unsqueeze(0).expand(Y_t.size())

    X_t = (grid_x[:, ::scale, ::scale] - camera_matrix.xc) * \
        Y_t[:, ::scale, ::scale] / camera_matrix.f
    Z_t = (grid_z[:, ::scale, ::scale] - camera_matrix.zc) * \
        Y_t[:, ::scale, ::scale] / camera_matrix.f

    XYZ = torch.stack(
        (X_t, -Z_t,Y_t[:, ::scale, ::scale]), dim=len(Y_t.size()))

    return XYZ


def transform_camera_view_t(_XYZ, sensor_height, camera_elevation_degree, device, need_transmatrix=False):
    """
    Transforms the point cloud into geocentric frame to account for
    camera elevation and angle
    Input:
        XYZ                     : ...x3
        sensor_height           : height of the sensor
        camera_elevation_degree : camera elevation to rectify.
    Output:
        XYZ : ...x3
    """
    XYZ = torch.ones_like(_XYZ)
    for i in range(_XYZ.shape[0]):
        R = ru.get_r_matrix(
            [1., 0., 0.], angle=np.deg2rad(camera_elevation_degree[i]))
        XYZ[i] = torch.matmul(_XYZ[i],torch.from_numpy(R).float().transpose(1, 0).to(device))
    XYZ[..., 2] = XYZ[..., 2] + sensor_height
    trans_matrix = None
    if need_transmatrix:
        trans_matrix = torch.eye(4)
        trans_matrix[:3, :3] = torch.from_numpy(R).float()
        trans_matrix[2, 3] = sensor_height
    return XYZ, trans_matrix


def transform_pose_t(_XYZ, current_pose, device, need_transmatrix=False):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : camera position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix([0., 0., 1.], angle=current_pose[2])
    XYZ = torch.matmul(_XYZ.reshape(-1, 3),
                       torch.from_numpy(R).float().transpose(1, 0).to(device)
                       ).reshape(_XYZ.shape)
    XYZ[..., 0] += current_pose[0]
    XYZ[..., 1] += current_pose[1]

    trans_matrix = None
    if need_transmatrix:
        trans_matrix = torch.eye(4)
        trans_matrix[:3, :3] = torch.from_numpy(R).float()
        trans_matrix[0, 3] += current_pose[0]
        trans_matrix[1, 3] += current_pose[1]
    return XYZ, trans_matrix

# 将depth和semantic 1+16 dims的特征扔在Voxels上面.
# 只看到vision_range内的这些点,也就是将这些点放进体素,其他的不管
def splat_feat_nd(init_grid, feat, coords):
    """
    Args:
        init_grid: B X nF X W X H X D X ..
        feat: B X nF X nPt
        coords: B X nDims X nPt in [-1, 1]
    Returns:
        grid: B X nF X W X H X D X ..
    """
    # init_grid:  [1, 17, 100, 100, 48]
    wts_dim = []
    pos_dim = []
    grid_dims = init_grid.shape[2:] # [100, 100, 48]  [x,y,z]

    B = init_grid.shape[0]
    F = init_grid.shape[1]

    n_dims = len(grid_dims)

    grid_flat = init_grid.view(B, F, -1)

    for d in range(n_dims):
        pos = coords[:, [d], :] * grid_dims[d] / 2 + grid_dims[d] / 2
        pos_d = []
        wts_d = []

        for ix in [0, 1]:
            pos_ix = torch.floor(pos) + ix # floor是向下取整.
            safe_ix = (pos_ix > 0) & (pos_ix < grid_dims[d])
            safe_ix = safe_ix.type(pos.dtype)

            wts_ix = 1 - torch.abs(pos - pos_ix)

            wts_ix = wts_ix * safe_ix
            pos_ix = pos_ix * safe_ix

            pos_d.append(pos_ix)
            wts_d.append(wts_ix)

        pos_dim.append(pos_d)
        wts_dim.append(wts_d)

    l_ix = [[0, 1] for d in range(n_dims)]
    # itertools.product  笛卡尔积
    for ix_d in itertools.product(*l_ix):
        wts = torch.ones_like(wts_dim[0][0])
        index = torch.zeros_like(wts_dim[0][0])
        for d in range(n_dims):
            index = index * grid_dims[d] + pos_dim[d][ix_d[d]]
            wts = wts * wts_dim[d][ix_d[d]]

        index = index.long()
        grid_flat.scatter_add_(2, index.expand(-1, F, -1), feat * wts)
        grid_flat = torch.round(grid_flat)

    return grid_flat.view(init_grid.shape)

def getSize(controller,grid_size=0.25):
    # 获取可达位置的坐标
    try:
        reachable_positions = controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"]
    except Exception as e:
        print(f"Error: {e}")
        controller.reset() 
    x_coords = [pos["x"] for pos in reachable_positions]
    y_coords = [pos["y"] for pos in reachable_positions]
    z_coords = [pos["z"] for pos in reachable_positions]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    z_min, z_max = min(z_coords), max(z_coords)

    obj_x = []
    obj_y = []
    obj_z = []
    # 获取物体位置的坐标并扩展坐标列表
    objects = controller.last_event.metadata["objects"]
    for obj in objects:
        obj_x += np.array(obj["axisAlignedBoundingBox"]['cornerPoints'])[:,0].tolist()
        obj_y += np.array(obj["axisAlignedBoundingBox"]['cornerPoints'])[:,1].tolist()
        obj_z += np.array(obj["axisAlignedBoundingBox"]['cornerPoints'])[:,2].tolist()
    # for obj in objects :
    #     print(obj["position"])
    # 计算扩展后的边界
    obj_x_min, obj_x_max = np.sign(min(obj_x))*np.ceil(abs(min(obj_x))/grid_size)*grid_size, np.sign(max(obj_x))*np.ceil(abs(max(obj_x))/grid_size)*grid_size
    obj_y_min, obj_y_max = np.sign(min(obj_y))*np.ceil(abs(min(obj_y))/grid_size)*grid_size, np.sign(max(obj_y))*np.ceil(abs(max(obj_y))/grid_size)*grid_size
    obj_z_min, obj_z_max = np.sign(min(obj_z))*np.ceil(abs(min(obj_z))/grid_size)*grid_size, np.sign(max(obj_z))*np.ceil(abs(max(obj_z))/grid_size)*grid_size
    border = np.array([[min(x_min,obj_x_min),max(x_max,obj_x_max)],
                       [min(y_min,obj_y_min),max(y_max,obj_y_max)],
                       [min(z_min,obj_z_min),max(z_max,obj_z_max)]])

    grid = border[:,1]-border[:,0]
    size = max(grid[0],grid[2])
    return [int(size/grid_size)+1,int(size/grid_size)+1,int(grid[1]/grid_size)+1],(border[0,0]-grid_size/2,border[2,0]-grid_size/2,border[1,0]-grid_size/2)
def getSizeProcthor(controller,grid_size=0.25):
    # 获取可达位置的坐标
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    bounds = event.metadata["sceneBounds"]["size"]
    cornerPoints = np.array(event.metadata["sceneBounds"]["cornerPoints"])
    min_bound = np.min(cornerPoints,axis=0)
    # x_coords = [pos["x"] for pos in reachable_positions]
    # y_coords = [pos["y"] for pos in reachable_positions]
    # z_coords = [pos["z"] for pos in reachable_positions]
    # x_min, x_max = min(x_coords), max(x_coords)
    # y_min, y_max = min(y_coords), max(y_coords)
    # z_min, z_max = min(z_coords), max(z_coords)

    # obj_x = []
    # obj_y = []
    # obj_z = []
    # # 获取物体位置的坐标并扩展坐标列表
    # objects = controller.last_event.metadata["objects"]
    # for obj in objects:
    #     obj_x += np.array(obj["axisAlignedBoundingBox"]['cornerPoints'])[:,0].tolist()
    #     obj_y += np.array(obj["axisAlignedBoundingBox"]['cornerPoints'])[:,1].tolist()
    #     obj_z += np.array(obj["axisAlignedBoundingBox"]['cornerPoints'])[:,2].tolist()
    # # for obj in objects :
    # #     print(obj["position"])
    # # 计算扩展后的边界
    # obj_x_min, obj_x_max = np.sign(min(obj_x))*np.ceil(abs(min(obj_x))/grid_size)*grid_size, np.sign(max(obj_x))*np.ceil(abs(max(obj_x))/grid_size)*grid_size
    # obj_y_min, obj_y_max = np.sign(min(obj_y))*np.ceil(abs(min(obj_y))/grid_size)*grid_size, np.sign(max(obj_y))*np.ceil(abs(max(obj_y))/grid_size)*grid_size
    # obj_z_min, obj_z_max = np.sign(min(obj_z))*np.ceil(abs(min(obj_z))/grid_size)*grid_size, np.sign(max(obj_z))*np.ceil(abs(max(obj_z))/grid_size)*grid_size
    size = max(bounds["x"], bounds["z"])
    return [int(size/grid_size)+1,int(size/grid_size)+1,int(bounds["y"]/grid_size)+1],(min_bound[0]-grid_size/2,min_bound[2]-grid_size/2,min_bound[1]-grid_size/2)
def depth_frame_to_camera_space_xyz_thor_grid(
    depth_frame: torch.Tensor, mask: Optional[torch.Tensor], fov: float = 90
) -> torch.Tensor:
    """Transforms a input depth map into a collection of xyz points (i.e. a
    point cloud) in the camera's coordinate frame.

    NOTE (samirg): copied from allenact to reduce # of dependencies, easier debugging,
    to ensure compatibility as allenact might change. This function is designed for
    Unity left-handed coordinate system.

    # Parameters
    depth_frame : A square depth map, i.e. an MxM matrix with entry `depth_frame[i, j]` equaling
        the distance from the camera to nearest surface at pixel (i,j).
    mask : An optional boolean mask of the same size (MxM) as the input depth. Only values
        where this mask are true will be included in the returned matrix of xyz coordinates. If
        `None` then no pixels will be masked out (so the returned matrix of xyz points will have
        dimension 3x(M*M)
    fov: The field of view of the camera.

    # Returns

    A 3xN matrix with entry [:, i] equalling a the xyz coordinates (in the camera's coordinate
    frame) of a point in the point cloud corresponding to the input depth frame.
    """
    assert (
        len(
            depth_frame.shape) == 2 and depth_frame.shape[0] == depth_frame.shape[1]
    ), f"depth has shape {depth_frame.shape}, we only support (N, N) shapes for now."



    depth_frame=hflip(depth_frame)

    shape = depth_frame.shape

    resolution = depth_frame.shape[0]
    if mask is None:
        mask = torch.ones_like(depth_frame, dtype=bool)

    # pixel centers
    camera_space_yx_offsets = (
        torch.stack(torch.where(mask))
        + 0.5  # Offset by 0.5 so that we are in the middle of the pixel
    )

    # Subtract center
    camera_space_yx_offsets -= resolution / 2.0

    # Make "up" in y be positive
    camera_space_yx_offsets[0, :] *= -1

    # Put points on the clipping plane
    camera_space_yx_offsets *= (2.0 / resolution) * \
        math.tan((fov / 2) / 180 * math.pi)

    camera_space_xyz = torch.cat(
        [
            camera_space_yx_offsets[1:, :],  # This is x
            camera_space_yx_offsets[:1, :],  # This is y
            torch.ones_like(camera_space_yx_offsets[:1, :]),
        ],
        axis=0,
    )

    points = camera_space_xyz * depth_frame[mask][None, :]

    points = points.reshape(3, shape[0], shape[1])
    points = hflip(points)

    return points
def getPcd(observations,fov):
        depth = observations["depth"]
        height, width = depth.shape

        intrinsic = get_intrinsic(width,height,fov)
        position = observations["position"]
        r = observations["rotation"]
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        z = depth
        x = (x - cx) * z / fx
        y = (y - cy) * z / fy
        points_3d = np.stack((x, -y, z), axis=-1).reshape(-1, 3)
        position = np.array([position['x'], position['y'], position['z']])
        rotation = R.from_euler('yx', [r["y"], r["x"]], degrees=True)
        rotation_matrix = rotation.as_matrix()
        points_global = points_3d @ rotation_matrix.T + position
        return torch.tensor(points_global.reshape(height,width,3)).to(torch.device("cuda:0" ))
# 转换点云到全局坐标系
def depth_to_3d(depth, fx, fy, cx, cy):
    rows, cols = np.indices((depth.shape[0], depth.shape[1]))
    z = depth
    x = (cols - cx) * z / fx
    y = (rows - cy) * z / fy
    points_3d = np.stack([x, -y, z], axis=-1)  # 将y轴取反以匹配ai2thor的方向
    return points_3d
def deal_depth_new(fov,depth,camera_position,camera_rotation_y,camera_pitch):
    height, width = depth.shape
    depth_intrinsic = {
    "fx": width / (2 * np.tan(fov * np.pi / 360)),
    "fy": height / (2 * np.tan(fov * np.pi / 360)),
    "cx": width / 2,
    "cy": height / 2,
    }
    points_3d = depth_to_3d(depth, **depth_intrinsic)
    rotation_pitch = R.from_euler("x", -camera_pitch, degrees=True).as_matrix()
    rotation_yaw = R.from_euler("y", camera_rotation_y, degrees=True).as_matrix()
    rotation_matrix = rotation_yaw  # 先应用俯仰，再应用水平旋转

    # 转换到全局坐标系
    points_3d_global = (rotation_matrix @ points_3d.reshape(-1, 3).T).T + np.array([camera_position["x"], camera_position["y"], camera_position["z"]])
    # camera_position=np.array([camera_position["x"],camera_position["y"],camera_position["z"]])
    # rotation_pitch = R.from_euler("x", -camera_pitch, degrees=True).as_matrix()
    # rotation_yaw = R.from_euler("y", -camera_rotation_y, degrees=True).as_matrix()
    # rotation_matrix = rotation_yaw # 先应用俯仰，再应用水平旋转

    # # 转换到全局坐标系并交换Y和Z方向
    # points_3d_global = (rotation_matrix @ points_3d.reshape(-1, 3).T).T + camera_position
    # points_3d_global_swapped = np.stack([points_3d_global[:, 0], points_3d_global[:, 2], points_3d_global[:, 1]], axis=-1)
    points_3d_global_swapped=points_3d_global.reshape(height,width,3)
    return points_3d_global_swapped