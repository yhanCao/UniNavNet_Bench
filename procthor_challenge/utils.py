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

import os
import copy
import numpy as np
from PIL import Image,ImageDraw
def find_last_less_than(nums, target):
    left, right = 0, len(nums) - 1
    result = 0  # 如果没有找到，返回-1
    
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            result = mid  # 记录候选位置
            left = mid + 1  # 继续在右侧寻找更大的索引
        else:
            right = mid - 1  # 调整右边界
            
    return result

def get_target_answer(event_objects, episode, index, current_time=0):
    objs = []
    answer = None
    if "answer" in episode:    
        if isinstance(episode['answer'][index],str):
            answer = episode['answer'][index]
        elif isinstance(episode['answer'][index], dict):
            times = list(episode['answer'][index].keys())
            times = [int(t) for t in times]
            idx = find_last_less_than(times, current_time)
            answer = episode['answer'][index][str(idx)]
        for obj in event_objects:
            if obj['objectId'] == episode['id'][index]:
                objs.append(obj)
    else :
        if episode['object_type'][index] is not None:
            if isinstance(episode['object_type'][index],str):
                for obj in event_objects:
                    if obj['objectType'] == episode['object_type'][index]:
                        objs.append(obj)
            elif isinstance(episode['object_type'][index],list):
                for obj_type in episode['object_type'][index] :
                    for obj in event_objects:
                        if obj['objectType'] == obj_type:
                            objs.append(obj)
        elif episode['id'][index] is not None:
            for obj in event_objects:
                if obj['objectId'] == episode['id'][index]:
                    objs.append(obj)
    return objs,answer
def get_top_down_frame(controller):
    # Setup the top-down camera
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]
    max_bound = max(bounds["x"], bounds["z"])

    pose["fieldOfView"] = 35
    pose["position"]["y"] += 1.1 * max_bound
    pose["orthographic"] = False
    pose["farClippingPlane"] = 50
    del pose["orthographicSize"]
    if len(event.third_party_camera_frames) == 0 :
        # add the camera to the scene
        event = controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )
    else :
        event = controller.last_event
    top_down_frame = event.third_party_camera_frames[-1]
    return Image.fromarray(top_down_frame),pose
def world_2_top_down_view(world_point, cam_pos, fov, img_size=(512, 512)):
    fov_rad = np.radians(fov)
    aspect_ratio = img_size[0] / img_size[1]
    f = 1.0 / np.tan(fov_rad / 2)  # 焦距

    # 2. 世界坐标 → 相机坐标（假设相机朝向 Z 轴正方向）
    # 相机坐标系：X 向右，Y 向下，Z 向前
    x_cam = world_point[0] - cam_pos[0]
    y_cam = world_point[2] - cam_pos[2]
    z_cam = -cam_pos[1]  # 相机高度（Y 轴向下）

    # 3. 透视投影（标准化设备坐标 NDC）
    # 注意：这里假设相机朝 Z 正方向，且 Y 轴向下
    x_ndc = (x_cam * f) / (z_cam * aspect_ratio)
    y_ndc = (y_cam * f) / z_cam

    # 4. NDC → 像素坐标
    pixel_x = int((x_ndc + 1) * 0.5 * img_size[0])
    pixel_y = int((1 - y_ndc) * 0.5 * img_size[1])  # Y 轴翻转（图像坐标系）

    return [img_size[0]-pixel_x, img_size[1]-pixel_y]
def draw_top_down_frame(controller,save_path,idx,target_objects):
    image,pose = get_top_down_frame(controller)
    no_target_path = os.path.join(save_path,"no_target/")
    if not os.path.exists(no_target_path):
        os.makedirs(no_target_path)
    image.save(os.path.join(no_target_path,str(idx)+".png"))
    width, height = image.size
    draw = ImageDraw.Draw(image)
    
    for obj in target_objects:
        obj_center = [obj['position']['x'], obj['position']['y'], obj['position']['z']]
        p2d = world_2_top_down_view(obj_center, [pose['position']['x'], pose['position']['y'], pose['position']['z']], pose['fieldOfView'], img_size=(width, height))
        if p2d is not None:
            draw.ellipse((p2d[0] - 5, p2d[1] - 5, p2d[0] + 5, p2d[1] + 5), fill='red', outline='red')
    target_path = os.path.join(save_path,"target/")
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    image.save(os.path.join(target_path,str(idx)+".png"))
def get_Tracable_Obj(objects,target_object):
    for obj in objects:
        if obj['id'] != target_object:
            if 'children' in obj :
                for child in obj['children']:
                    if child['id'] == target_object:
                        return obj
        else:
            return obj
        