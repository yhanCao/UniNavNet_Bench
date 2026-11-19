import networkx as nx
from collections import Counter
import numpy as np
from st_map.geometry import extractVoxelCenterfromVoxelmap
from scipy.spatial import KDTree
def euclidean_distance(pos, node1, node2):
    x1, y1 = pos[node1]
    x2, y2 = pos[node2]
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
def getObjectName(obj):
    counter = Counter(obj['class_name'])
    most_common_element, count = counter.most_common(1)[0]
    return most_common_element
def calculate_detailed_direction(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1

    if x1 == x2 and y1 == y2:
        return "Close neighbor"
    elif x1 == x2:
        if y2 > y1:
            return "Directly above"
        else:
            return "Directly below"
    elif y1 == y2:
        if x2 > x1:
            return "Directly to the right"
        else:
            return "Directly to the left"
    else:
        if dx > 0 and dy > 0:
            if abs(dx) > abs(dy):
                return "Up-right leaning right"
            else:
                return "Up-right leaning up"
        elif dx > 0 and dy < 0:
            if abs(dx) > abs(dy):
                return "Down-right leaning right"
            else:
                return "Down-right leaning down"
        elif dx < 0 and dy > 0:
            if abs(dx) > abs(dy):
                return "Up-left leaning left"
            else:
                return "Up-left leaning up"
        elif dx < 0 and dy < 0:
            if abs(dx) > abs(dy):
                return "Down-left leaning left"
            else:
                return "Down-left leaning down"
def QueryByLLM(goal_name,agent_node,voronoi_graph,explored_nodes,frontier_nodes,objects_around,history_context=None,current_step=None,keep_nodes=None):
    # one line per node: the prompt stays short and does not grow with the explored area
    pos = nx.get_node_attributes(voronoi_graph, 'pos')
    lines = [f"Goal: {goal_name}", f"Agent at Node {agent_node}"]
    if agent_node in pos:
        dist = nx.single_source_dijkstra_path_length(voronoi_graph,agent_node,weight=lambda u,v,d:euclidean_distance(pos,u,v))
    else:
        dist = {}
    for node in pos.keys():
        if keep_nodes is not None and node not in keep_nodes and node != agent_node:
            continue
        flags = ["you are here" if node == agent_node else "", "frontier" if node in frontier_nodes else "",
                 "explored" if node in explored_nodes else "unexplored"]
        txt = f"Node {node} : " + ", ".join(f for f in flags if f)
        if node != agent_node:
            d = dist.get(node, euclidean_distance(pos,agent_node,node) if agent_node in pos else 0)
            txt += f", {d:.1f}m {calculate_detailed_direction(pos[agent_node],pos[node])}"
        if objects_around is not None and objects_around.get(node):
            txt += " | " + ", ".join(objects_around[node])
        lines.append(txt)
    candidate = list(pos.keys()) if keep_nodes is None else [n for n in pos.keys() if n in keep_nodes]
    if agent_node in candidate:
        candidate.remove(agent_node)
    USER = "\n".join(lines) + "\n" + describe_history_context(history_context,pos,current_step) + f"Candidates: {candidate}"
    return USER,candidate
def describe_history_context(history_context,pos,current_step):
    # temporal context of instruction-related objects removed from the active scene graph
    if history_context is None or len(history_context) == 0 or len(pos) == 0:
        return ""
    keys = list(pos.keys())
    values = np.array(list(pos.values()))
    txt = f"Moved away (current timestep {current_step}):\n"
    for item in history_context:
        nearest = keys[int(np.argmin(np.linalg.norm(values-item['center'],axis=1)))]
        txt += f"  {item['name']} was near Node {nearest}, gone at timestep {item['archived_time']}. " + " ".join(item['captions']) + "\n"
    return txt
def select_subgraph_nodes(voronoi_graph,agent_node,frontier_nodes,objects_around_idx,obj_sims,k_nodes=8,k_frontiers=4):
    # nodes whose surrounding objects are most related to the instruction, plus the nearest frontiers
    pos = nx.get_node_attributes(voronoi_graph, 'pos')
    scores = {}
    if objects_around_idx is not None:
        for node,idxs in objects_around_idx.items():
            if node in pos and node != agent_node and len(idxs) > 0:
                scores[node] = max(obj_sims[i] for i in idxs)
    relevant = sorted(scores,key=lambda n:-scores[n])[:k_nodes]
    dist = nx.single_source_dijkstra_path_length(voronoi_graph,agent_node,weight=lambda u,v,d:euclidean_distance(pos,u,v)) if agent_node in pos else {}
    frontiers = sorted([n for n in frontier_nodes if n in pos and n != agent_node],key=lambda n:dist.get(n,float('inf')))[:k_frontiers]
    keep = set(relevant) | set(frontiers)
    if len(keep) == 0:
        # nothing related and no frontier: fall back to the nearest nodes
        keep = set(sorted([n for n in pos if n != agent_node],key=lambda n:dist.get(n,float('inf')))[:k_nodes])
    return keep
def format_nearby_objects(objects_around_idx,objects,obj_sims,max_per_node=None,sim_thresh=0.5):
    # object names per node, most instruction-related first; related objects carry their last visible timestep
    if objects_around_idx is None:
        return None
    result = {}
    for node,idxs in objects_around_idx.items():
        idxs = sorted(idxs,key=lambda i:-obj_sims[i])
        if max_per_node:
            idxs = idxs[:max_per_node]
        l = []
        for i in idxs:
            txt = getObjectName(objects[i])
            seen = [t for t,state in objects[i]['state_dict'].items() if state == 1]
            if obj_sims[i] > sim_thresh and len(seen) > 0:
                txt += f" (last seen at timestep {max(seen)})"
            l.append(txt)
        result[node] = l
    return result

def find_nearby_objects_2d(grid_points, objects, radius,shape,minbound,size,return_index=False):
    result = {}
    object_points_2d = []
    object_indices = []
    for obj_idx,obj in enumerate(objects):
        obj_points = extractVoxelCenterfromVoxelmap(obj['voxel_index'],shape,minbound,size)
        object_points_2d.append(obj_points[:, :2])  # (x, y)
        object_indices.extend([obj_idx] * len(obj_points))  # object index of each point
    # 1. (x, y) only
    object_points_2d = np.vstack(object_points_2d)  # (points, 2)
    object_indices = np.array(object_indices)  # (points,)
    
    # 3. KD-tree
    kdtree = KDTree(object_points_2d)

    # 4. query every grid point
    
    for key,point in grid_points.items():
        l=[]
        indices = kdtree.query_ball_point(point, radius)  # 2D radius query
        nearby_objects = set(object_indices[indices])  # object indices
        for obj_idx in nearby_objects :
            l.append(int(obj_idx) if return_index else getObjectName(objects[obj_idx]))
        result[key] = l
    return result
def find_nearest_landmark(object,grid_points,shape,minbound,size):
    obj_points = extractVoxelCenterfromVoxelmap(object['voxel_index'],shape,minbound,size)
    object_points_2d = np.mean(obj_points[:, :2],axis=0).reshape(1,-1)  # (x, y)
    keys = np.array(list(grid_points.keys()))
    values = np.array(list(grid_points.values()))
    distances = np.linalg.norm(values[:, np.newaxis, :] - object_points_2d[np.newaxis, :, :], axis=2)
    min_distances = np.min(distances, axis=1)
    nearest_index = np.argmin(min_distances)
    return keys[nearest_index],values[nearest_index]
def find_nearest_node(object,node,points,shape,minbound,size):
    obj_points = extractVoxelCenterfromVoxelmap(object['voxel_index'],shape,minbound,size)
    object_points_2d = np.mean(obj_points[:, :2],axis=0).reshape(1,-1)  # (x, y)
    keys = np.array(node)
    values = np.array(points)
    distances = np.linalg.norm(values[:, np.newaxis, :] - object_points_2d[np.newaxis, :, :], axis=2)
    min_distances = np.min(distances, axis=1)
    nearest_index = np.argmin(min_distances)
    return keys[nearest_index],values[nearest_index]
