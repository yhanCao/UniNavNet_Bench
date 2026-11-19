import cv2
import torch
import torch.nn as nn
from PIL import Image
from st_map._vendor.utils.common import *
import st_map._vendor.robothor_challenge.utils as du
# from detection.groundedsam import detection, InitDetection
from st_map._vendor.slam.slam_classes import MapObjectList
from st_map._vendor.slam.mapping import removalOverlap
from st_map._vendor.vlm.embedding_test import EmbeddingText
from st_map._vendor.utils.draw_pcd import draw_voxel_objects,draw_pcd_objects
from st_map._vendor.vlm.vlm import VLMQwQ
import matplotlib.pyplot as plt
import time
import json
import pickle as pkl
import gzip
from st_map.config import resource_path
from st_map._vendor.detection.clipmodel import CLIPModel
from omegaconf import DictConfig, OmegaConf
from st_map._vendor.slam.utils import intrinsic_from_fov,rotation_from_euler,selectPointsFromVoxel,X_w_to_X_v,extractVoxelCenterfromVoxelmap
from st_map._vendor.slam.cfslam import cfvoxel
from detectron2.data import MetadataCatalog
from st_map._vendor.detection.openseed import detection,InitDetection
# from detection.maskclip import detection,InitDetection
from st_map._vendor.scenegraph.prepare import prepareRelationForCOG
# from detection.som import generateSomPrompt,generateSoMImage,mapObjects
import matplotlib.colors as mcolors
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from scipy.spatial import Voronoi
from collections import Counter
def most_common_element(lst):
    if not lst:  # 如果列表为空，返回 None 或抛出异常
        return None
    counter = Counter(lst)
    most_common = counter.most_common(1)  # 返回出现次数最多的元素及其计数
    return most_common[0][0]  # 返回元素本身
metadata = MetadataCatalog.get('coco_2017_train_panoptic')
css4_colors = mcolors.CSS4_COLORS
color_proposals = [list(mcolors.hex2color(color)) for color in css4_colors.values()]
def map_points_to_grid_centers(points, 
                               offset=np.array([-0.125, -0.125]), 
                               cell_size=0.25):
    """
    将一组二维点映射到给定网格 (offset, cell_size) 的中心。
    
    参数：
    --------
    points : np.ndarray
        形状 (n, 2) 的浮点数组，每行表示一个 (x, y) 点。
    offset : np.ndarray 或 形如 (2,) 的可迭代对象
        网格原点（对应索引0时的中心点位置），如 [-0.125, -0.125]。
    cell_size : float
        网格单元大小（宽度/高度），如 0.25。
    
    返回：
    --------
    mapped_points : np.ndarray
        形状 (n, 2) 的数组，对应每个点被映射到网格中心后的 (x, y)。
    """
    # 确保 offset 为 numpy 数组，方便后续广播运算
    offset = np.array(offset, dtype=np.float64)
    
    # 1) 将 points 平移到以 offset 为“坐标原点”
    # 2) 除以 cell_size 得到落在哪个格子（浮点格子坐标）
    # 3) floor(...) 取整得到格子索引
    # 4) +0.5 表示取该格子的中心
    # 5) 乘回 cell_size 并加上 offset，即回到原始坐标系
    mapped_points = offset + cell_size * (
        np.floor((points - offset) / cell_size) + 0.5
    )
    
    return mapped_points

def get_grid_coordinates(points, cell_size, origin=(-0.125, -0.125)):
    """
    计算一组点相对于网格原点的网格坐标。
    :param points: 输入点，形状为 (n, 2)，每一行是 (x, y)
    :param cell_size: 网格单元大小
    :param origin: 网格原点，默认为 (-0.125, -0.125)
    :return: 网格坐标，形状为 (n, 2)
    """
    # 转换坐标，相对于原点
    relative_points = points - origin
    
    # 计算网格坐标，除以网格大小并取整
    grid_coords = np.floor(relative_points / cell_size).astype(int)
    
    return grid_coords
def are_points_in_same_grid(group1, group2, cell_size):
    contained_points = []
    not_contained_points = []
    points_contained_dict = {}
    # 将 Group2 中的点转换为网格坐标

    group1_grid = get_grid_coordinates(group1, cell_size)
    group2_grid = get_grid_coordinates(group2, cell_size)
    
    # 将 Group2 的网格坐标转化为一个集合，便于快速查找
    group2_grid_set = set(map(tuple, group2_grid))  # 将每个网格坐标变为元组并存入集合

    # 判断 Group1 中的每个点是否在 Group2 的网格中
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
    在给定多边形区域 boundary_poly 内，对 points 做 Voronoi 剖分并裁剪到边界。

    参数：
    ----------
    points : np.ndarray
        (N, 2) 形状的数组，每行一个 (x, y)。
    boundary_poly : Polygon
        Shapely Polygon（或可转换为多边形的其它几何对象），表示边界。

    返回：
    ----------
    clipped_polygons : list
        每个元素是裁剪后的 (Multi)Polygon。可能为空，也可能包含多个或单个子多边形。
    """
    # 1) 用 SciPy 生成 Voronoi
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
    根据网格中心点坐标生成所有网格的并集，并返回其外边界坐标。

    参数：
    ----------
    center_points : np.ndarray
        形状 (n, 2) 的数组，每一行表示一个网格中心点 (x, y)。
    cell_size : float
        网格大小(边长)，默认 0.25。

    返回：
    ----------
    boundaries : list
        如果所有网格并集连成一个单一Polygon，则返回 [list_of_coords]；
        如果存在多块不相连区域，则返回多个外边界坐标列表 [list_of_coords1, list_of_coords2, ...]。

        其中每个 list_of_coords 的形式为：
        [
            (x0, y0),
            (x1, y1),
            ...,
            (x0, y0)    # 首尾相同，表示封闭多边形
        ]
    """
    # 半边长
    half_side = cell_size / 2

    # 1. 生成每个网格对应的 Polygon
    polygons = []
    # 预先计算每个中心点的 x_min, y_min, x_max, y_max
    x_min = center_points[:, 0] - half_side
    y_min = center_points[:, 1] - half_side
    x_max = center_points[:, 0] + half_side
    y_max = center_points[:, 1] + half_side

    # corners 数组形状: (n, 4, 2)
    # 第一个维度对应每个网格; 第二个维度是四个角点; 第三个维度是 (x, y)
    n = center_points.shape[0]
    corners = np.empty((n, 4, 2), dtype=float)

    # 依次填入 4 个顶点坐标
    # 角点顺序: (x_min, y_min) -> (x_min, y_max) -> (x_max, y_max) -> (x_max, y_min)
    corners[:, 0, 0] = x_min
    corners[:, 0, 1] = y_min

    corners[:, 1, 0] = x_min
    corners[:, 1, 1] = y_max

    corners[:, 2, 0] = x_max
    corners[:, 2, 1] = y_max

    corners[:, 3, 0] = x_max
    corners[:, 3, 1] = y_min

    # 以列表推导方式将每一行(4个顶点)变为一个 Polygon
    polygons = [Polygon(corner) for corner in corners]

    # 2. 计算并集
    union_poly = unary_union(polygons)

    # 3. 提取并集外边界坐标
    # boundaries = []
    # if union_poly.geom_type == 'Polygon':
    #     # 单块Polygon
    #     boundaries.append(list(union_poly.exterior.coords))
    # elif union_poly.geom_type == 'MultiPolygon':
    #     # 多块Polygon（不相连的区域）
    #     for geom in union_poly.geoms:
    #         boundaries.append(list(geom.exterior.coords))
    # else:
    #     # 根据实际需求，可做相应处理，这里简单抛出异常
    #     raise TypeError(f"Unexpected geometry type: {union_poly.geom_type}")

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
        # self.map_size_cm = args.map_size_cm // args.global_downscaling
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
        self.env_camera_matrix = intrinsic_from_fov(self.screen_h, self.screen_w, self.fov) # 内参
        self.scene_graph={}
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # self.cog_tokenizer,self.cog_model=load_model("/home/caoyihan/Code/THUDM/cogvlm2-llama3-chinese-chat-19B-int4")
        # detection xxx
        self.BG_CLASSES =  ["wall","wall-wood", "floor", "ceiling","rug","Carpet"]
        runtime_cfg = runtime_cfg or {}
        self.cfg_detection = load_config(runtime_cfg.get("mapping_config", resource_path("sg.yaml")))
        self.cfg_detection = process_cfg(OmegaConf.create(self.cfg_detection))
        self.instance_classes_set,_ = InitDetection(runtime_cfg)
        # self.instance_classes_set = None
        # InitDetection(runtime_cfg)
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
        # detection_path = os.path.join(save_path,"detection/")
        # pcd_path = os.path.join(save_path,"pcd/")
        # occupancy_map_path = os.path.join(save_path,"trajetory/")
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        # if not os.path.exists(detection_path):
        #     os.makedirs(detection_path)
        if self.debug_artifacts and not os.path.exists(visualization_path):
            os.makedirs(visualization_path)
        # if not os.path.exists(pcd_path):
        #     os.makedirs(pcd_path)
        # if not os.path.exists(occupancy_map_path):
        #     os.makedirs(occupancy_map_path)
        Image.fromarray(obs.astype(np.uint8),'RGB').save(os.path.join(image_path, str(idx)+'.jpg'))
        self.saved_frames.add(idx)
        gobs = detection(Image.fromarray(obs.astype(np.uint8),'RGB'),idx,visualization_path if self.debug_artifacts else None)
        if gobs is not None:
            gobs = self.clipmodel.compute_clip_features(Image.fromarray(obs.astype(np.uint8),'RGB'),gobs)
        # if detection_path is not None:
        #     with gzip.open(detection_path+str(idx)+".pkl.gz", "wb") as f:
        #         pkl.dump(gobs, f)
        if gobs is not None :
            res=[]# [image; pcd; intrinsic; pose]
            res.append(obs)
            res.append(pcd)
            res.append(self.env_camera_matrix)
            # res.append(pose_curr)
            # self.reses_history[idx]=res
            self.objects,self.bg_objects,self.target_similarity_with_objects=cfvoxel(self.cfg_detection,res,gobs,self.objects,idx=idx,step=idx,classes=self.instance_classes_set,bg_objects=self.bg_objects,img_name=os.path.join(image_path, '{}.jpg'.format(idx)),pcd_path=None,som_path=None,shape=self.size,min_bound=self.min_bound,similarity = self.target_similarity_with_objects,textmodel = self.textmodel)
    def updateSceneGraph(self,save_path,idx,exp_map):
        som_path = os.path.join(save_path,"som/")
        if not os.path.exists(som_path):
            os.makedirs(som_path)
        self.objects,self.bg_objects,self.changed = removalOverlap(self.cfg_detection,self.objects,self.bg_objects,self.changed,self.size,self.min_bound)
        # pcd_path = os.path.join(save_path,"pcd/")
        # if not os.path.exists(pcd_path):
        #     os.makedirs(pcd_path)
        # draw_voxel_objects(self.objects,self.size,self.min_bound,self.resolution)
        som_pth_step = som_path + str(idx) +"/"
        if not os.path.exists(som_pth_step):
            os.makedirs(som_pth_step)
        relation_list,image_list,image_objects_dict,image_pair_dict,check_relation=prepareRelationForCOG(self.cfg_detection,self.objects,self.changed,som_pth_step,idx)
        ####### get surrounding objects#####
        surrounding_objects={}
        for relation in relation_list :
            if relation[0] in surrounding_objects.keys() :
                surrounding_objects[relation[0]].append(relation[1])
                surrounding_objects[relation[0]] = list(set(surrounding_objects[relation[0]]))
            else :
                surrounding_objects[relation[0]] = [relation[1]]
            if relation[1] in surrounding_objects.keys() :
                surrounding_objects[relation[1]].append(relation[0])
                surrounding_objects[relation[1]] = list(set(surrounding_objects[relation[1]]))
            else :
                surrounding_objects[relation[1]] = [relation[0]]
        self.changed=[]
        obj_index=1
        voxels_detection = torch.zeros(
            1,self.size[0],self.size[0],self.size[2]
            ).float().to(self.device)
        wall=None
        ####obj_index =1 indicates wall
        for name,bg in self.bg_objects.items() :
            if (name == "wall" or name == "wall-wood") and bg is not None :
                indices_3=np.array(np.unravel_index(bg['voxel_index'], (self.size[0],self.size[0],self.size[2]),order='C')).T # n*3
                indices_3 = torch.from_numpy(indices_3).to(self.device)
                index =(indices_3 >torch.tensor([0,0,0]).to(self.device)) & (indices_3 < torch.tensor([self.size[0],self.size[0],self.size[2]]).to(self.device))
                index_new = torch.where(index[:,0]&index[:,1]&index[:,2])[0]
                voxels_detection[0,indices_3[index_new,1],indices_3[index_new,0],indices_3[index_new,2]] = obj_index  
                wall = indices_3[index_new,:2].cpu().numpy()
        obj_index +=1
        for obj in self.objects :
            indices_3=np.array(np.unravel_index(obj['voxel_index'],(self.size[0],self.size[0],self.size[2]),order='C')).T # n*3
            indices_3 = torch.from_numpy(indices_3).to(self.device)
            index =(indices_3 >torch.tensor([0,0,0]).to(self.device)) & (indices_3 < torch.tensor([self.size[0],self.size[0],self.size[2]]).to(self.device))
            index_new = torch.where(index[:,0]&index[:,1]&index[:,2])[0]
            voxels_detection[0,indices_3[index_new,1],indices_3[index_new,0],indices_3[index_new,2]] = obj_index  
            obj_index +=1
        min_h =self.min_bound[2]
        min_z = int((0.3-min_h)/self.z_resolution)
        mid_z = int((self.agent_height - min_h) / self.z_resolution )
        agent_height_proj_detection = voxels_detection[..., min_z:mid_z].max(3)[0]
        obstacle_map = agent_height_proj_detection[0]/ self.map_pred_threshold
        outlier_region = torch.where(exp_map==0)
        obstacle_map[outlier_region[0],outlier_region[1]]=-1
        # return self.objects,None,obstacle_map.cpu().numpy()
        # return self.objects,self.scene_graph,obstacle_map
        return self.objects,self.bg_objects,None,obstacle_map,wall
    
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
        # cam_coords = cam_coords[:, np.where(cam_coords[2] <= 20)[0]]  # ???

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

        # filtered_occ_pts = np.array([pt for pt in occupancy_points if pt[2] <= 0 and pt[2] >= -Configuration.CAMERA_HEIGHT])

        # # Update map model occupancy points
        # self.map_model.update_occupancy(filtered_occ_pts, pos, angle, file_name, collision)
        return occupancy_points.reshape(depth_matrix.shape[0],depth_matrix.shape[1],3)
    
    @torch.no_grad()
    def getGlobalMap(self,map_before,observations,idx,save_path,agent_locations):
        # self.getPoint(observations["event"].depth_frame,last_action)
        # sem_pcd_path = os.path.join(save_path,"sem_pcd/")
        # object_path = os.path.join(save_path,"object/")
        # if not os.path.exists(object_path):
        #     os.makedirs(object_path)
        # if not os.path.exists(sem_pcd_path):
        #     os.makedirs(sem_pcd_path)
        event = observations["event"]
        # perceptions = self.perceive(event)
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
        # self.pcd +=pcd
        self.forward_detection(event.frame, world_view_t, idx ,save_path)
        # 
        #### save results
        # o3d.io.write_point_cloud(os.path.join(pcd_path,str(idx)+".ply"), self.pcd)
        # results = {
        #         'objects': self.objects.to_serializable(),
        #         'bg_objects': MapObjectList([_ for _ in self.bg_objects.values() if _ is not None]).to_serializable(),
        #         # 'total_steps':idx,
        #         # 'agent': event.metadata['agent']
        #     }
        # # with open(os.path.join(save_path,"sg/")+f"{str(idx)}.json", "wt", encoding="utf-8") as f:
        # #     json.dump(results, f, indent=4) 
        # open(object_path+str(idx)+".pkl.gz", 'wb').write(gzip.compress(pkl.dumps(results)))
        # results = {
        #         # 'objects': self.objects.to_serializable(),
        #         # 'bg_objects': MapObjectList([_ for _ in self.bg_objects.values() if _ is not None]).to_serializable(),
        #         # 'total_steps':idx,
        #         'agent': event.metadata['agent']
        #     }
        # with open(os.path.join(save_path,"sg/")+f"{str(idx)}.json", "wt", encoding="utf-8") as f:
        #     json.dump(results, f, indent=4) 
        # open(os.path.join(save_path,"sg/")+f"{str(idx)}.pkl.gz", 'wb').write(gzip.compress(pkl.dumps(results)))
        #### save results
        self.updateObjectState(idx,save_path,agent_locations,pcd,event.depth_frame,event.frame)
        
        # draw_pcd_objects(self.objects,self.bg_objects,shape=self.size,min_bound=self.min_bound,idx=idx,pcd_path=sem_pcd_path)
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
    def compute_overlap_voxel(self,pcd1, pcd2, voxel_size=0.05):
        # 体素化点云
        pcd1_voxel = o3d.geometry.PointCloud.voxel_down_sample(pcd1, voxel_size)
        pcd2_voxel = o3d.geometry.PointCloud.voxel_down_sample(pcd2, voxel_size)
        
        # 获取体素中心点
        points1 = np.asarray(pcd1_voxel.points)
        points2 = np.asarray(pcd2_voxel.points)
        
        # 构建KDTree加速搜索
        from scipy.spatial import cKDTree
        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)
        
        # 查找重叠体素
        overlap_indices1 = tree1.query_ball_tree(tree2, r=voxel_size * 0.6)
        overlap_count = sum(len(indices) > 0 for indices in overlap_indices1)
        
        # 计算重叠率
        overlap_ratio = overlap_count / min(len(points1), len(points2))
        
        return overlap_ratio
