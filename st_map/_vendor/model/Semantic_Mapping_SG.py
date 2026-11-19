import os

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from PIL import Image
from st_map._vendor.utils.common import *
from st_map._vendor.slam.slam_classes import MapObjectList
from st_map._vendor.vlm.embedding_test import EmbeddingText
from st_map._vendor.vlm.vlm import VLMQwQ
from st_map.config import resource_path
from st_map._vendor.detection.clipmodel import CLIPModel
from omegaconf import OmegaConf
from st_map._vendor.slam.utils import intrinsic_from_fov, rotation_from_euler, X_w_to_X_v
from st_map._vendor.slam.cfslam import cfvoxel
from detectron2.data import MetadataCatalog
from st_map._vendor.detection.openseed import detection,InitDetection
import matplotlib.colors as mcolors
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from scipy.spatial import Voronoi
from collections import Counter
def most_common_element(lst):
    if not lst:  # empty list: return None
        return None
    counter = Counter(lst)
    most_common = counter.most_common(1)  # the most frequent element and its count
    return most_common[0][0]  # the element itself
metadata = MetadataCatalog.get('coco_2017_train_panoptic')
css4_colors = mcolors.CSS4_COLORS
color_proposals = [list(mcolors.hex2color(color)) for color in css4_colors.values()]
def map_points_to_grid_centers(points, 
                               offset=np.array([-0.125, -0.125]), 
                               cell_size=0.25):
    """
    Maps 2D points to the centers of the grid cells (offset, cell_size) they fall in.
    
    Parameters:
    --------
    points : np.ndarray
        float array of shape (n, 2), one (x, y) point per row.
    offset : np.ndarray or iterable of shape (2,)
        grid origin (center of cell 0), e.g. [-0.125, -0.125].
    cell_size : float
        cell size (width / height), e.g. 0.25.
    
    Returns:
    --------
    mapped_points : np.ndarray
        array of shape (n, 2): the (x, y) cell center of each point.
    """
    # numpy array for broadcasting
    offset = np.array(offset, dtype=np.float64)
    
    # 1) shift the points so that offset is the origin
    # 2) divide by cell_size: fractional cell coordinates
    # 3) floor(...): cell index
    # 4) +0.5: center of the cell
    # 5) scale by cell_size and add offset: back to world coordinates
    mapped_points = offset + cell_size * (
        np.floor((points - offset) / cell_size) + 0.5
    )
    
    return mapped_points

def get_grid_coordinates(points, cell_size, origin=(-0.125, -0.125)):
    """
    Grid coordinates of points relative to the grid origin.
    :param points: points of shape (n, 2), one (x, y) per row
    :param cell_size: cell size
    :param origin: grid origin, default (-0.125, -0.125)
    :return: grid coordinates of shape (n, 2)
    """
    # relative to the origin
    relative_points = points - origin
    
    # divide by the cell size and round down
    grid_coords = np.floor(relative_points / cell_size).astype(int)
    
    return grid_coords
def are_points_in_same_grid(group1, group2, cell_size):
    contained_points = []
    not_contained_points = []
    points_contained_dict = {}
    # grid coordinates of Group2

    group1_grid = get_grid_coordinates(group1, cell_size)
    group2_grid = get_grid_coordinates(group2, cell_size)
    
    # Group2 cells as a set for fast lookup
    group2_grid_set = set(map(tuple, group2_grid))  # cells as tuples

    # whether each Group1 point lies in a Group2 cell
    for grid_point,origin_point in zip(group1_grid,group1):
        if tuple(grid_point) not in group2_grid_set:
            not_contained_points.append(origin_point)
            points_contained_dict[tuple(origin_point)] = 0
        else :
            contained_points.append(origin_point)
            points_contained_dict[tuple(origin_point)] = 1
    return np.array(contained_points),np.array(not_contained_points),points_contained_dict
def voronoi_polygons_clipped(points: np.ndarray, boundary_poly: Polygon):
    """
    Voronoi partition of points, clipped to the polygon boundary_poly.

    Parameters:
    ----------
    points : np.ndarray
        array of shape (N, 2), one (x, y) per row.
    boundary_poly : Polygon
        Shapely Polygon (or geometry convertible to one): the boundary.

    Returns:
    ----------
    clipped_polygons : list
        clipped (Multi)Polygon per point; may be empty or have several parts.
    """
    # 1) Voronoi diagram with SciPy
    vor = Voronoi(points)
    vertices=vor.vertices
    index = np.zeros([len(vertices)])
    relation = vor.ridge_vertices
    for i in range(len(vertices)) :
        if boundary_poly.contains(Point(vertices[i][0],vertices[i][1])) :
            index[i] = 1
    index_remain = list(np.where(index)[0])
    relation_new=[]
    for i in range(len(relation)):
        if relation[i][0] in index_remain and relation[i][1] in index_remain and relation[i][0] >=0 and relation[i][1] >=0:
            relation_new.append((index_remain.index(relation[i][0]),index_remain.index(relation[i][1])))
    vor.ridge_vertices = relation_new
    vor.vertices= vertices[index_remain]
    return vor

def union_grid_boundary(center_points: np.ndarray, cell_size: float = 0.25):
    """
    Union of the grid cells centered at the given points, and its exterior boundary coordinates.

    Parameters:
    ----------
    center_points : np.ndarray
        array of shape (n, 2), one cell center (x, y) per row.
    cell_size : float
        cell size (side length), default 0.25.

    Returns:
    ----------
    boundaries : list
        [list_of_coords] if the union is a single Polygon;
        [list_of_coords1, list_of_coords2, ...], one per disconnected region, otherwise.

        where each list_of_coords is:
        [
            (x0, y0),
            (x1, y1),
            ...,
            (x0, y0)    # first and last equal: closed polygon
        ]
    """
    # half side length
    half_side = cell_size / 2

    # 1. one Polygon per cell
    polygons = []
    # x_min, y_min, x_max, y_max of every cell
    x_min = center_points[:, 0] - half_side
    y_min = center_points[:, 1] - half_side
    x_max = center_points[:, 0] + half_side
    y_max = center_points[:, 1] + half_side

    # corners: (n, 4, 2)
    n = center_points.shape[0]
    corners = np.empty((n, 4, 2), dtype=float)

    # the 4 corners
    # corner order: (x_min, y_min) -> (x_min, y_max) -> (x_max, y_max) -> (x_max, y_min)
    corners[:, 0, 0] = x_min
    corners[:, 0, 1] = y_min

    corners[:, 1, 0] = x_min
    corners[:, 1, 1] = y_max

    corners[:, 2, 0] = x_max
    corners[:, 2, 1] = y_max

    corners[:, 3, 0] = x_max
    corners[:, 3, 1] = y_min

    # one Polygon per row of 4 corners
    polygons = [Polygon(corner) for corner in corners]

    # 2. union
    union_poly = unary_union(polygons)

    # 3. exterior coordinates of the union
    #     # single Polygon
    # elif union_poly.geom_type == 'MultiPolygon':
    #     # several Polygons (disconnected regions)
    # else:
    #     # not handled: raise

    return union_poly

from st_map.memory import TemporalMemoryMixin

class Semantic_Mapping_SG(TemporalMemoryMixin, nn.Module):
    def __init__(self,fov,size,min_bound,memory_cfg=None,runtime_cfg=None):
        super(Semantic_Mapping_SG, self).__init__()
        self.device = torch.device("cuda:0" )
        self.screen_h = 480
        self.screen_w = 640
        self.resolution = 0.25
        self.z_resolution = 0.25
        self.num_scene=1
        self.size = size
        self.min_bound=min_bound
        self.fov = fov
        self.du_scale = 1
        self.exp_pred_threshold = 1.0 # 1.0
        self.map_pred_threshold = 1.0 # 1.0
        self.agent_history={}
        self.selected_view=[]
        self.agent_height = 1.5
        self.camera_to_agent = torch.eye(4)
        self.env_camera_matrix = intrinsic_from_fov(self.screen_h, self.screen_w, self.fov) # camera intrinsics
        self.scene_graph={}
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # detection xxx
        self.BG_CLASSES =  ["wall","wall-wood", "floor", "ceiling","rug","Carpet"]
        runtime_cfg = runtime_cfg or {}
        self.cfg_detection = load_config(runtime_cfg.get("mapping_config", resource_path("sg.yaml")))
        self.cfg_detection = process_cfg(OmegaConf.create(self.cfg_detection))
        self.instance_classes_set,_ = InitDetection(runtime_cfg)
        self.textmodel = EmbeddingText(runtime_cfg.get("embedding_model", "mixedbread-ai/mxbai-embed-large-v1"))
        self.clipmodel = CLIPModel(self.device, runtime_cfg.get("clip_model", "ViT-H-14"), runtime_cfg.get("clip_pretrained", "laion2b_s32b_b79k"))
        self.top_down_view_path = None
        self.global_classes = set()
        self.objects = MapObjectList(device=self.device)
        self.bg_objects = {c: None for c in self.BG_CLASSES}
        self.changed=[]
        self.target_similarity_with_objects=[]
        self.history_archive = []          # objects removed from the active graph (state 2)
        self.appearance_sim_thresh = 0.5   # HS-histogram correlation below this counts as appearance mismatch
        self.vlm = VLMQwQ(**runtime_cfg.get("vlm", {}))
        memory_cfg = memory_cfg or {}
        # VLM appearance descriptions for retrieved history objects, off by default
        self.history_caption = memory_cfg.get('history_caption',False)
        # memory bounds (see TemporalMemoryMixin.trimMemory)
        self.views_per_object = memory_cfg.get('views_per_object',3)
        self.class_votes = memory_cfg.get('class_votes',20)
        self.segments_per_object = memory_cfg.get('segments_per_object',8)
        self.archive_limit = memory_cfg.get('archive_limit',32)
        self.debug_artifacts = runtime_cfg.get('debug_artifacts',False)
        self.saved_frames = set()
        # "visible" needs 100 projected points at the upstream 5 mm point spacing: keep the same area at the
        # configured spacing (100 points at 5 mm cover what ~6 points cover at 2 cm)
        self.min_visible_points = max(1, round(100 * (0.005 / self.cfg_detection.downsample_voxel_size) ** 2))
        self.caption_vlm = None
        if self.history_caption:
            vlm_cfg = memory_cfg.get('caption_vlm') or {}
            self.caption_vlm = VLMQwQ(vlm_cfg.get('base_url'),vlm_cfg.get('api_key'),vlm_cfg.get('model','default'))
        self.pcd = o3d.geometry.PointCloud()
    def update_similarity(self):
        self.target_similarity_with_objects=[]
        for i,obj in enumerate(self.objects) :
            self.target_similarity_with_objects.append(self.textmodel.query_multiple(obj['class_name']))
            self.objects[i]['target_judge']=True
    def forward_detection(self, obs, pcd, idx,save_path):
        image_path = os.path.join(save_path, 'image')
        visualization_path = os.path.join(save_path,"visual/")
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        if self.debug_artifacts and not os.path.exists(visualization_path):
            os.makedirs(visualization_path)
        Image.fromarray(obs.astype(np.uint8),'RGB').save(os.path.join(image_path, str(idx)+'.jpg'))
        self.saved_frames.add(idx)
        gobs = detection(Image.fromarray(obs.astype(np.uint8),'RGB'),idx,visualization_path if self.debug_artifacts else None)
        if gobs is not None:
            gobs = self.clipmodel.compute_clip_features(Image.fromarray(obs.astype(np.uint8),'RGB'),gobs)
        if gobs is not None :
            res=[]# [image; pcd; intrinsic; pose]
            res.append(obs)
            res.append(pcd)
            res.append(self.env_camera_matrix)
            self.objects,self.bg_objects,self.target_similarity_with_objects=cfvoxel(self.cfg_detection,res,gobs,self.objects,idx=idx,step=idx,classes=self.instance_classes_set,bg_objects=self.bg_objects,img_name=os.path.join(image_path, '{}.jpg'.format(idx)),pcd_path=None,som_path=None,shape=self.size,min_bound=self.min_bound,similarity = self.target_similarity_with_objects,textmodel = self.textmodel)
    
    def get_point_cloud(self, depth_matrix):

        # Get intrinsic parameters
        height, width = depth_matrix.shape
        K_inv = np.linalg.inv(self.env_camera_matrix)

        # Get pixel coordinates
        pixel_coords = self.pixel_coord_np(width, height)  # [3, npoints]

        # Rotation matrix
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        R_inv = np.linalg.inv(R)

        # Apply back-projection: K_inv @ R_inv @ pixels * depth
        cam_coords = K_inv[:3, :3] @ R_inv @ pixel_coords * depth_matrix.flatten()

        return cam_coords
    
    def pixel_coord_np(self, width, height):
        """
        Pixel in homogenous coordinate
        Returns:
            Pixel coordinate:       [3, width * height]
        """
        x = np.linspace(0, width - 1, width).astype(np.int32)
        y = np.linspace(0, height - 1, height).astype(np.int32)
        [x, y] = np.meshgrid(x, y)
        return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))
    def update_global_points(self, depth_matrix, angle, cam_angle, pos):

        angle = (int(angle) - 90) % 360  # rescale angle according to simulator reference system

        x, y, z = self.get_point_cloud(depth_matrix)

        # flip the y-axis to positive upwards
        cam_coords = np.array((x, -y, z))

        # Filter cam coordinates according to agent view horizon

        # Get camera points
        x, y, z = cam_coords

        # Get agent depth view occupancy points
        occupancy_points = np.column_stack((x, z, y))
        rot_matrix = rotation_from_euler(yaw=np.deg2rad(angle), roll=np.deg2rad(cam_angle))
        # Rotate agent view according to agent orientation
        occupancy_points = np.dot(occupancy_points, rot_matrix.T[:3, :3])

        # Add agent offset position to agent view
        occupancy_points[:, 0] += pos['x']
        occupancy_points[:, 1] += pos['y']
        occupancy_points[:, 2] += pos['z']


        # # Update map model occupancy points
        return occupancy_points.reshape(depth_matrix.shape[0],depth_matrix.shape[1],3)
    
    @torch.no_grad()
    def getGlobalMap(self,map_before,observations,idx,save_path,agent_locations):
        event = observations["event"]
        angle =(360 - event.metadata['agent']['rotation']['y']+90) % 360
        pos = {"x":  event.metadata['agent']['position']['x'], "y": event.metadata['agent']['position']['z'], "z": event.metadata['cameraPosition']['y']}
        self.agent_history[idx]={
            'pos': pos,
            'angle': event.metadata['agent']['rotation']['y'],
            'roll': event.metadata['agent']['cameraHorizon'],
        }
        world_view_t = self.update_global_points(event.depth_frame, angle, -int(event.metadata['agent']['cameraHorizon']), pos)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(world_view_t.reshape(-1,3))
        pcd.normals = o3d.utility.Vector3dVector(np.ones_like(world_view_t.reshape(-1,3)))
        pcd.colors = o3d.utility.Vector3dVector(event.frame.reshape(-1,3)/255)
        self.forward_detection(event.frame, world_view_t, idx ,save_path)
        # 
        #### save results
        # results = {
        #         'objects': self.objects.to_serializable(),
        #         'bg_objects': MapObjectList([_ for _ in self.bg_objects.values() if _ is not None]).to_serializable(),
        #         # 'total_steps':idx,
        #     }
        # # with open(os.path.join(save_path,"sg/")+f"{str(idx)}.json", "wt", encoding="utf-8") as f:
        # results = {
        #         # 'total_steps':idx,
        #         'agent': event.metadata['agent']
        #     }
        #### save results
        self.updateObjectState(idx,save_path,agent_locations,pcd,event.depth_frame,event.frame)
        
        return None,map_before,None,pcd
    def get_mask(self, mask_range):
        size = int(mask_range) * 2
        mask = torch.zeros(size, size)
        for i in range(size):
            for j in range(size):
                if ((i + 0.5) - (size // 2)) ** 2 + ((j + 0.5) - (size // 2)) ** 2 <= mask_range ** 2:
                    mask[i, j] = 1
        return mask
    def reset(self):
        self.global_classes = set()
        self.objects = MapObjectList(device=self.device)
        self.bg_objects = {c: None for c in self.BG_CLASSES}
        self.changed=[]
        self.selected_view=[]
    def find_Maximum_View(self,obj_id,obj):
        ratios = {}
        for idx in obj['image_idx']:
            kernel = np.ones((15, 15), np.uint8)
            depth_buffer = np.zeros((self.screen_h,self.screen_w), dtype=np.float32)
            mask = -np.ones((self.screen_h,self.screen_w), dtype=np.int32)
            R = rotation_from_euler(yaw=np.deg2rad((int((360 - self.agent_history[idx]['angle']+90) % 360)-90)%360), roll=np.deg2rad(-int(self.agent_history[idx]['roll'])))
            for i,obj in enumerate(self.objects) :
                points = np.array(obj['pcd'].points)
                X_w = X_w_to_X_v(R,[self.agent_history[idx]['pos']['x'],self.agent_history[idx]['pos']['y'],self.agent_history[idx]['pos']['z']],points)
                points_camera = self.env_camera_matrix[:3,:3] @ X_w
                points_camera = points_camera.T
                z = points_camera[:,-1]
                # divide only points in front of the near plane (others would give inf / NaN pixel coordinates)
                z_safe = np.where(z >= 0.1, z, 1.0)
                u = (points_camera[:,0] / z_safe).astype(np.int32)
                v = (points_camera[:,1] / z_safe).astype(np.int32)
                index = np.where((u>=0)*(u<self.screen_w)*(v>=0)*(v<self.screen_h)*(z>=0.1))[0]
                index_buffer = np.where((depth_buffer[v[index],u[index]]==0) | (depth_buffer[v[index],u[index]]>z[index]))[0]
                if i == obj_id :
                    ratios[idx] = len(index_buffer)/len(points) if len(points) > 0 else 0
                if index[index_buffer].shape[0] > 0:
                    mask1 = np.zeros((self.screen_h,self.screen_w), dtype=np.uint8)
                    mask1[v[index][index_buffer],u[index][index_buffer]] = 1
                    closed_mask = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
                    depth_buffer[np.where(closed_mask!=0)[0],np.where(closed_mask!=0)[1]] = np.mean(z[index][index_buffer])
                    mask[np.where(closed_mask!=0)[0],np.where(closed_mask!=0)[1]] = i
        sorted_items = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
        sorted_idx = [k for k, v in sorted_items]
        return sorted_idx[:3]
