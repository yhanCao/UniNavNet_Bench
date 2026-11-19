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
import numpy as np

# Projects depth and the 1+16 semantic feature dims into voxels,
# keeping only the points within vision_range.

def getSize(controller,grid_size=0.25):
    # coordinates of the reachable positions
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
    # add the object positions
    objects = controller.last_event.metadata["objects"]
    for obj in objects:
        obj_x += np.array(obj["axisAlignedBoundingBox"]['cornerPoints'])[:,0].tolist()
        obj_y += np.array(obj["axisAlignedBoundingBox"]['cornerPoints'])[:,1].tolist()
        obj_z += np.array(obj["axisAlignedBoundingBox"]['cornerPoints'])[:,2].tolist()
    # bounds of all of them
    obj_x_min, obj_x_max = np.sign(min(obj_x))*np.ceil(abs(min(obj_x))/grid_size)*grid_size, np.sign(max(obj_x))*np.ceil(abs(max(obj_x))/grid_size)*grid_size
    obj_y_min, obj_y_max = np.sign(min(obj_y))*np.ceil(abs(min(obj_y))/grid_size)*grid_size, np.sign(max(obj_y))*np.ceil(abs(max(obj_y))/grid_size)*grid_size
    obj_z_min, obj_z_max = np.sign(min(obj_z))*np.ceil(abs(min(obj_z))/grid_size)*grid_size, np.sign(max(obj_z))*np.ceil(abs(max(obj_z))/grid_size)*grid_size
    border = np.array([[min(x_min,obj_x_min),max(x_max,obj_x_max)],
                       [min(y_min,obj_y_min),max(y_max,obj_y_max)],
                       [min(z_min,obj_z_min),max(z_max,obj_z_max)]])

    grid = border[:,1]-border[:,0]
    size = max(grid[0],grid[2])
    return [int(size/grid_size)+1,int(size/grid_size)+1,int(grid[1]/grid_size)+1],(border[0,0]-grid_size/2,border[2,0]-grid_size/2,border[1,0]-grid_size/2)
# point cloud to world coordinates
