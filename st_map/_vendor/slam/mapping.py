import torch
import torch.nn.functional as F
from st_map._vendor.slam.slam_classes import MapObjectList, DetectionList
import numpy as np
from st_map._vendor.utils.ious import (
    compute_iou_batch, 
    compute_giou_batch, 
    compute_3d_iou_accuracte_batch, 
    compute_3d_giou_accurate_batch,
)
from st_map._vendor.utils.general_utils import to_tensor
from st_map._vendor.slam.utils import (
    merge_obj2_into_obj1, 
    merge_obj2_into_obj1_2, 
    compute_overlap_matrix_2set,
    mapPcdToVoxel,
    selectPcdFromVoxel
)
def compute_voxel_similarities(cfg , detection_list: DetectionList,objects: MapObjectList,voxel_map):
    voxel_detect_similarity=torch.zeros([len(detection_list),len(objects)],dtype=torch.float)# [i,j] indicates overlap of (i,j) / i percentage of i 
    voxel_object_similarity=torch.zeros([len(objects),len(detection_list)],dtype=torch.float)# [i,j] indicates overlap of (i,j) / i percentage of i 
    bbox_map = objects.get_stacked_values_torch('bbox')
    bbox_new = detection_list.get_stacked_values_torch('bbox')
    try:
        iou = compute_3d_iou_accuracte_batch(bbox_map, bbox_new) # (m, n)
    except ValueError:
        print("Met `Plane vertices are not coplanar` error, use axis aligned bounding box instead")
        bbox_map = []
        bbox_new = []
        for pcd in objects.get_values('pcd'):
            bbox_map.append(np.asarray(
                pcd.get_axis_aligned_bounding_box().get_box_points()))
        for pcd in detection_list.get_values('pcd'):
            bbox_new.append(np.asarray(
                pcd.get_axis_aligned_bounding_box().get_box_points()))
        bbox_map = torch.from_numpy(np.stack(bbox_map))
        bbox_new = torch.from_numpy(np.stack(bbox_new))
        
        iou = compute_iou_batch(bbox_map, bbox_new) # (m, n)
    for i in range(len(detection_list)):
        for j in range(len(objects)):
            if iou[i,j] < 1e-6:
                continue
            pcd_new = np.array(detection_list[i]['pcd'].points)
            voxel_index_new = mapPcdToVoxel(pcd_new)
            voxel_index_object = np.where(voxel_map==j)[0]
            common_elements = np.intersect1d(voxel_index_new, voxel_index_object)
            
            voxel_detect_similarity[i,j] = len(common_elements) / len(voxel_index_new)
            voxel_object_similarity[j,i] = len(common_elements) / len(voxel_index_object)

    return voxel_detect_similarity , voxel_object_similarity.T
def comput_voxel_overlap(cfg , detection_list: DetectionList,objects: MapObjectList):
    voxel_overlap=torch.zeros([len(detection_list),len(objects)],dtype=torch.float)# [i,j] indicates overlap of (i,j) / i percentage of i 
    bbox_map = objects.get_stacked_values_torch('bbox')
    bbox_new = detection_list.get_stacked_values_torch('bbox')
    try:    
        iou = compute_3d_iou_accuracte_batch(bbox_map, bbox_new) # (m, n)
    except ValueError:
        print("Met `Plane vertices are not coplanar` error, use axis aligned bounding box instead")
        bbox_map = []
        bbox_new = []
        
        for pcd in objects.get_values('pcd'):
            bbox_map.append(np.asarray(
                pcd.get_axis_aligned_bounding_box().get_box_points()))
        
        for pcd in detection_list.get_values('pcd'):
            bbox_new.append(np.asarray(
                pcd.get_axis_aligned_bounding_box().get_box_points()))
        bbox_map = torch.from_numpy(np.stack(bbox_map))
        bbox_new = torch.from_numpy(np.stack(bbox_new))
        iou = compute_iou_batch(bbox_map, bbox_new) # (m, n)


    for i in range(len(detection_list)):
        for j in range(len(objects)):
            if iou[j,i] < 1e-6:
                continue
            voxel_index_new = detection_list[i]['voxel_index']
            voxel_index_object = objects[j]['voxel_index']
            if len(voxel_index_new) != 0 and len(voxel_index_object) != 0 :
                common_elements = np.intersect1d(voxel_index_new, voxel_index_object)
                voxel_overlap[i,j] = len(common_elements) / min(len(voxel_index_new),len(voxel_index_object))

    return voxel_overlap 
def compute_spatial_similarities(cfg, detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''
    Compute the spatial similarities between the detections and the objects
    
    Args:
        detection_list: a list of M detections
        objects: a list of N objects in the map
    Returns:
        A MxN tensor of spatial similarities
    '''
    det_bboxes = detection_list.get_stacked_values_torch('bbox')
    obj_bboxes = objects.get_stacked_values_torch('bbox')

    if cfg.spatial_sim_type == "iou":
        compute_iou_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "giou":
        compute_giou_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "iou_accurate":
        compute_3d_iou_accuracte_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "giou_accurate":
        compute_3d_giou_accurate_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "overlap":
        spatial_sim1,spatial_sim2 = compute_overlap_matrix_2set(cfg, objects, detection_list)
        spatial_sim1 = spatial_sim1.T
        spatial_sim2 = spatial_sim2.T
    else:
        raise ValueError(f"Invalid spatial similarity type: {cfg.spatial_sim_type}")
    
    return spatial_sim1

def compute_visual_similarities(cfg, detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''
    Compute the visual similarities between the detections and the objects
    
    Args:
        detection_list: a list of M detections
        objects: a list of N objects in the map
    Returns:
        A MxN tensor of visual similarities
    '''
    det_fts = detection_list.get_stacked_values_torch('text_ft') # (M, D)
    obj_fts = objects.get_stacked_values_torch('text_ft') # (N, D)

    det_fts = det_fts.unsqueeze(-1) # (M, D, 1)
    obj_fts = obj_fts.T.unsqueeze(0) # (1, D, N)
    
    visual_sim = F.cosine_similarity(det_fts, obj_fts, dim=1) # (M, N)
    
    return visual_sim

def aggregate_similarities(cfg, spatial_sim: torch.Tensor, visual_sim: torch.Tensor) -> torch.Tensor:
    '''
    Aggregate spatial and visual similarities into a single similarity score
    
    Args:
        spatial_sim: a MxN tensor of spatial similarities
        visual_sim: a MxN tensor of visual similarities
    Returns:
        A MxN tensor of aggregated similarities
    '''
    if cfg.match_method == "sim_sum":
        sims = (1 + cfg.phys_bias) * spatial_sim + (1 - cfg.phys_bias) * visual_sim # (M, N)
    else:
        raise ValueError(f"Unknown matching method: {cfg.match_method}")
    
    return sims

# def merge_detections_to_objects(
#     cfg, 
#     detection_list: DetectionList, 
#     objects: MapObjectList, 
#     agg_sim: torch.Tensor,
#     agg_sim2: torch.Tensor,
# ) -> MapObjectList:
#     # Iterate through all detections and merge them into objects
#         # If not matched to any object, add it as a new object
#         # Merge with most similar existing object
#         elif agg_sim2[i].max() == float('-inf'):
#         else :
def merge_detections_to_objects(
    cfg, 
    detection_list: DetectionList, 
    objects: MapObjectList, 
    agg_sim: torch.Tensor
) -> MapObjectList:
    changed=[]
    # Iterate through all detections and merge them into objects
    for i in range(agg_sim.shape[0]):
        # If not matched to any object, add it as a new object
        if agg_sim[i].max() == float('-inf'):
            objects.append(detection_list[i])
            changed.append(i)
        # Merge with most similar existing object
        else:
            j = agg_sim[i].argmax()
            matched_det = detection_list[i]
            matched_obj = objects[j]
            merged_obj = merge_obj2_into_obj1_2(cfg, matched_obj, matched_det, run_dbscan=False)
            objects[j] = merged_obj
            changed.append(j.item())
    return objects,changed
# def updateVoxelDict

def VoxelMergeStrategy(
    cfg, 
    detection_list: DetectionList, 
    objects: MapObjectList, 
    overlap_sim: torch.Tensor,##overlap of detection list
    visual_sim: torch.Tensor,##overlap of object list
    shape,
    minbound,
    similarity,
    textmodel
):
    agg_sim = overlap_sim + visual_sim 
    agg_sim[agg_sim < 1.5] = float('-inf')
    for i in range(agg_sim.shape[0]):
        # If not matched to any object, add it as a new object
        if agg_sim[i].max() == float('-inf'):
            objects.append(detection_list[i])
        # Merge with most similar existing object
        else:
            j = agg_sim[i].argmax()
            matched_det = detection_list[i]
            matched_obj = objects[j]
            merged_obj = merge_obj2_into_obj1(cfg, matched_obj, matched_det,shape,minbound,0.25, run_dbscan=False)
            objects[j] = merged_obj
            
    return objects,similarity
def removalOverlap(cfg,objects,bg_objects,shape,minbound,similarity,textmodel):
    if len(objects) != 0 :
        kept_objects = np.ones(len(objects), dtype=bool)
        for i in range(len(objects)-1):
            for j in range(i+1,len(objects)):
                if objects[i] is not None and objects[j] is not None :
                    common_elements = np.intersect1d(objects[i]['voxel_index'], objects[j]['voxel_index'])
                    if len(common_elements) != 0 :
                        visual_sim = F.cosine_similarity(
                        to_tensor(objects[i]['clip_ft'],device=cfg.device),
                        to_tensor(objects[j]['clip_ft'],device=cfg.device),
                        dim=0
                        )
                        if visual_sim > 0.8 and len(common_elements)/min(len(objects[i]['voxel_index']),len(objects[j]['voxel_index'])) > 0.5:
                            # if visual_sim > cfg.merge_visual_sim_thresh and \
                            #     text_sim > cfg.merge_text_sim_thresh:
                            if kept_objects[i]:
                                # Then merge object i into object j
                                objects[i] = merge_obj2_into_obj1(cfg, objects[j], objects[i],shape,minbound,0.25, run_dbscan=True)
                                objects[j] = None
                                kept_objects[j] = False
                        else:
                            ### use ovlerlap rate first if not use cogvlm2 to judge
                            overlap1 = len(common_elements) / len(objects[i]['voxel_index'])
                            overlap2 = len(common_elements) / len(objects[j]['voxel_index'])
                            if overlap1 >= overlap2 :
                                mask = np.isin(objects[j]['voxel_index'], common_elements, invert=True)
                                objects[j]['voxel_index'] = objects[j]['voxel_index'][mask]
                                objects[j]['pcd'] = selectPcdFromVoxel(objects[j],0.25,shape,minbound)
                            else :
                                mask = np.isin(objects[i]['voxel_index'], common_elements, invert=True)
                                objects[i]['voxel_index'] = objects[i]['voxel_index'][mask]
                                objects[i]['pcd'] = selectPcdFromVoxel(objects[i],0.25,shape,minbound)
                            del overlap1,overlap2,mask
                        del visual_sim
                    del common_elements
        for i in range(len(objects)):
            if kept_objects[i] != False :
                for key,value in bg_objects.items():
                    if kept_objects[i] == False :
                        continue
                    if key != "floor" or key != "ceiling" or key !="rug" or key != "Carpet":
                        continue
                    if value != None: 
                        common_elements = np.intersect1d(objects[i]['voxel_index'], value['voxel_index'])
                        if len(common_elements) != 0 :
                            if len(common_elements)/len(objects[i]['voxel_index']) < 0.2 :
                                mask = np.isin(objects[i]['voxel_index'], common_elements, invert=True)
                                objects[i]['voxel_index'] = objects[i]['voxel_index'][mask]
                            else :
                                kept_objects[i] = False
                                bg_objects[key] = merge_obj2_into_obj1(cfg, objects[i], value,shape,minbound,0.25, run_dbscan=False)
                                objects[i]=None
            if kept_objects[i] != False :
                if len(objects[i]['voxel_index']) < 5 :
                    kept_objects[i] = False
        new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
        objects = MapObjectList(new_objects)
    return objects,bg_objects,similarity
def updateVoxelandObject(voxel_map_dict,objects,voxel_map,changed):
    for id in changed :
        voxel_index = mapPcdToVoxel(np.asarray(objects[id]['pcd'].points))
        max_object = np.array([max(voxel_map_dict[k], key=lambda sub_key: len(voxel_map_dict[k][sub_key])) for k in voxel_index.tolist()])
        ### Objects that appear in the most frames in this voxel
        voxel_map[voxel_index] = max_object
    return voxel_map
