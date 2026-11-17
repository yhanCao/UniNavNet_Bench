import networkx as nx
from collections import Counter
import numpy as np
from st_map.geometry import extractVoxelIndex3,extractVoxelCenterfromVoxelmap
from scipy.spatial import KDTree
import time
import string
from networkx.exception import NetworkXNoPath
def euclidean_distance(pos, node1, node2):
    x1, y1 = pos[node1]
    x2, y2 = pos[node2]
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
def QueryNextSubGoal(goal_name,llm,objects,scene_graph,graph,leaf_nodes,agent_location,distance_threshold=30):
    pos = nx.get_node_attributes(graph, 'pos')
    objects_around={}
    for node,value in leaf_nodes.items() :
        if node in pos.keys():
            for i,obj in enumerate(objects) :
                coordinates = extractVoxelIndex3(obj['voxel_index'])
                min_distance=np.min(np.linalg.norm(coordinates[:,:2]-np.array([pos[node][1],pos[node][0]]),axis=-1))
                if min_distance < distance_threshold :
                    if node in objects_around.keys() :
                        objects_around[node].append(i)
                    else :
                        objects_around[node]=[i]
    USER=f"Goal: {goal_name}\n"
    USER = USER + f"Agent Location: ({str(agent_location[0])}, {str(agent_location[1])}).\n"
    for node,value in leaf_nodes.items() :
        if node in pos.keys():
            txt=f"-Description {str(node)}: "
            if value == 0 or value == 1 :
                break
            if value == -1 :
                txt += f"This node is directed to the unexplored space, located at ({str(graph.nodes()[node]['pos'][1])}, {str(graph.nodes()[node]['pos'][0])})."
            else :
                counter = Counter(objects[int(value-2)]['class_name'])
                most_common_element, count = counter.most_common(1)[0]
                txt =txt + f"This node is directed to a {most_common_element}, located at ({str(graph.nodes()[node]['pos'][1])}, {str(graph.nodes()[node]['pos'][0])})."
            if node in objects_around.keys() :
                txt += " The area contains "
                objs_around=objects_around[node]
                for obj_index in objs_around :
                    counter = Counter(objects[obj_index]['class_name'])
                    most_common_element, count = counter.most_common(1)[0]
                    txt = txt + f"a {most_common_element}"

                    # if obj_index in scene_graph.keys() :
                    #     txt += ", which is "
                    #     for obj2,relation in scene_graph[obj_index].items() :
                    #         counter = Counter(relation)
                    #         most_common_element, count = counter.most_common(1)[0]
                    #         if most_common_element == 'on':
                    #             most_common_element = 'under'
                    #         elif most_common_element == 'under':
                    #             most_common_element = 'on'
                    #         cc = Counter(objects[obj2]['class_name'])
                    #         mm, _ = cc.most_common(1)[0]
                    #         txt = txt + most_common_element + " a " + mm + ", "
                    #     txt = txt[:-2] + "; "
                    # else :
                    txt +="; "
            if txt[-1] =='.':
                txt+=f"\n"
            else :
                txt = f"{txt[:-2]}.\n"
            USER += txt
    print(f"=========> Query:\n{USER}")
    
    while True:
        try:
            answer, reply = llm.choose_frontier(USER)
            break
        except Exception as ex: # rate limit
            print(f"[ERROR] in LLM inference =====> {ex}, sleep for 20s...")
            time.sleep(20)
            continue
    print(f"=========> LLM output:\n{reply}")
    selection = np.array([int(graph.nodes()[answer]['pos'][0]),int(graph.nodes()[answer]['pos'][1])])
    return selection
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
def Query_For_State_Transition(goal_name,objects,node_rooms):
    USER=f"Goal: {goal_name}\n"
    USER+=f"Objects contained in the explored area : "

    USER+=f"Rooms contained in the explored area : Node "
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
        flags = ["frontier" if node in frontier_nodes else "", "explored" if node in explored_nodes else "unexplored"]
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
def QueryByLLMFrontier(goal_name,agent_node,voronoi_graph,explored_nodes,frontier_nodes,objects_around):
    USER=f"Goal: {goal_name}\n"
    USER+=f"Agent Now : Node {agent_node}\n"
    ###describe nodes ###
    pos = nx.get_node_attributes(voronoi_graph, 'pos')
    
    Node_descp=f"Node description : \n"
    for node in pos.keys():
        if node in frontier_nodes :
            Node_descp+=f"Node {node} : \n"
        # if node in leaf_nodes.keys() :
        #     if leaf_nodes[node] == -1 :
        #         Node_descp+=f"  Frontier node: True\n"
        #     else :
        #         Node_descp+=f"  Frontier node: False\n"
        # Node_descp+=f"  Frontier node: False\n"
        
            Node_descp+=f"  Frontier node: True\n"
            ##########
            if node != agent_node :
                Node_descp+=f"  Location:\n"
                direction = calculate_detailed_direction(pos[agent_node],pos[node])
                try:
                    path = nx.shortest_path(voronoi_graph, source=agent_node, target=node)
                    distance= euclidean_distance(pos,agent_node,path[0])
                    distance += euclidean_distance(pos,path[-1],node)
                    for nidx in range(len(path)-1):
                        distance+=euclidean_distance(pos,path[nidx],path[nidx+1])
                    Node_descp+=f"    Direction: {direction} relative to {agent_node}\n"
                    Node_descp+=f"    Path from {agent_node} to {node}: {agent_node} ->"
                    for n in path[1:-1] :
                        Node_descp+=f"{n} ->"
                    Node_descp+=f"{node}\n"
                    Node_descp+=f"    Distance from {agent_node} to {node}: {distance}\n"
                except NetworkXNoPath:
                    Node_descp+=f"    Direction: {direction} relative to {agent_node}\n"
            if objects_around is not None and node in objects_around.keys():
                Node_descp+=f"  Surrounding objects:\n"
                objs_around=objects_around[node]
                Node_descp+=f"There are: "
                for obj_message in objs_around :
                    Node_descp+=f"{obj_message}; "
                Node_descp=Node_descp[:-2]+f" around this node.\n"
            # if node_rooms[node] != "None" and node_rooms[node] is not None :
            #     Node_descp+=f"  Room: {node_rooms[node]}.\n"
            # else :
            #     Node_descp+=f"  Room: Unknown.\n"
            if node in explored_nodes :
                Node_descp+=f"  Explored: True\n"
            else :
                Node_descp+=f"  Explored: False\n"
            Node_descp+=f"\n"
    candidate = list(pos.keys())
    if agent_node in candidate:
        candidate.remove(agent_node)
    nodes=f"The nodes candidate you select: {candidate}"
    USER = USER +Node_descp+nodes
    return USER,candidate
def getObjectNearby(objects,bg_objects,graph,obstacle_map,distance_threshold=20):
    pos = nx.get_node_attributes(graph, 'pos')
    objects_around={}
    position=np.array(list(pos.values()))
    pos_id = list(pos.keys())
    edge_nodes=[]
    # 打印所有边及其属性
    obstacle_map = (obstacle_map.cpu().numpy() * 255).astype(np.uint8)
    all_edges_with_data = graph.edges(data=False)
    import cv2
    color_image1 = cv2.cvtColor(obstacle_map, cv2.COLOR_GRAY2BGR)
    node = []
    points_2d=[]
    relation=[]
    for i,pose in pos.items():
            points_2d.append([pose[0],pose[1]])
            point_int = (int(pose[0]),int(pose[1]))
            cv2.circle(color_image1, point_int, 1, (0, 0, 255), -1)  # red
            node.append(i)
    
    for edge in all_edges_with_data:
    
        relation.append([node.index(edge[0]),node.index(edge[1])])

    edge_nodes = list(set(edge_nodes))
    for edge in relation:
            pt1 = (int(points_2d[edge[0]][0]),int(points_2d[edge[0]][1]))  # 转换为整数
            pt2 = (int(points_2d[edge[1]][0]),int(points_2d[edge[1]][1]))   # 转换为整数
            cv2.line(color_image1, pt1, pt2, (255, 0, 0), 1)  # blue色线
    for obj in objects:
        obj_name = getObjectName(obj)

        coordinates = extractVoxelIndex3(obj['voxel_index'])
        # coordinates[:,0],coordinates[:,1] = coordinates[:,1],coordinates[:,0]
        center = np.mean(coordinates[:,:2],axis=0)
        point_int = (int(center[0]),int(center[1]))  # 转换为整数
        cv2.circle(color_image1, point_int, 1, (0, 0, 255), -1)  # red
        min_distance=np.min(np.linalg.norm(coordinates[None,:,:2]-position[:,None,:],axis=-1),axis=-1)
        index = np.where(min_distance < distance_threshold)[0]
        select_id = [pos_id[i] for i in index]
        for i,id in zip(index,select_id) :
            direction = calculate_detailed_direction(position[i],center)
            distance = min_distance[i]
            if id in objects_around.keys():
                objects_around[id].append({'obj':obj_name,'direction':direction,'distance':distance})
            else :
                objects_around[id]=[{'obj':obj_name,'direction':direction,'distance':distance}]
    cv2.imwrite('color_image_with_circle_and_line.png', color_image1)
    # import matplotlib.pyplot as plt
    # plt.imshow(cv2.cvtColor(color_image1, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # # 保存图像到文件 
    # plt.show()
    for name,bg in bg_objects.items() :
        if (name == "wall" or name == "wall-wood") and bg is not None :
            coordinates = extractVoxelIndex3(bg['voxel_index'])
            min_id=np.argmin(np.linalg.norm(coordinates[None,:,:2]-position[:,None,:],axis=-1),axis=-1)
            min_distance=np.min(np.linalg.norm(coordinates[None,:,:2]-position[:,None,:],axis=-1),axis=-1)
            index = np.where(min_distance < distance_threshold)[0]
            select_id = [pos_id[i] for i in index]
            for i,id in zip(index,select_id) :
                direction = calculate_detailed_direction(position[i],coordinates[min_id[i],:2])
                distance = min_distance[i]
                if id in objects_around.keys():
                    objects_around[id].append({'obj':'wall','direction':direction,'distance':distance})
                else :
                    objects_around[id]=[{'obj':'wall','direction':direction,'distance':distance}]
    return objects_around
def QueryByGraph(goal_name,agent_node,obstacle_map,centers,objects,bg_objects,scene_graph,voronoi_graph,node_rooms,explored_nodes,leaf_nodes):
    USER=f"Goal: {goal_name}\n"
    USER+=f"Agent Now : Node {agent_node}\n"
    uppercase_letters = list(string.ascii_uppercase)
    # objects_descp=f"Green Holl Square Area description: \n"
    # area_objects={}
    # object_area={}
    # for i,center in enumerate(centers) :
    #     obj_id = int(obstacle_map[int(center[0]),int(center[1])]-2)
    #     obj_desc = f"\t{uppercase_letters[i]} contains objects: [{getObjectName(objects[obj_id])}"
    #     area_objects[i]=[obj_id]
    #     object_area[obj_id] = i
    #     if obj_id in scene_graph.keys() :
    #         surd_objs = scene_graph[obj_id]
    #         for obj_id in surd_objs :
    #             obj_desc += f", {getObjectName(objects[obj_id])}"
    #             area_objects[i].append(obj_id)
    #             object_area[obj_id] = i
    #     obj_desc +="]."
    #     objects_descp+=f"{obj_desc}\n"
    ###describe nodes ###
    objects_around=getObjectNearby(objects,bg_objects,voronoi_graph,obstacle_map)
    
    # objects_around={}
    # for node in pos.keys() :
    #     if node in pos.keys():
    #         for i,obj in enumerate(objects) :
    #             coordinates = extractVoxelIndex3(obj['voxel_index'])
    #             min_distance=np.min(np.linalg.norm(coordinates[:,:2]-np.array([pos[node][1],pos[node][0]]),axis=-1))
    #             if min_distance < distance_threshold :
    #                 if node in objects_around.keys() :
    #                     objects_around[node].append(i)
    #                 else :
    #                     objects_around[node]=[i]
    pos = nx.get_node_attributes(voronoi_graph, 'pos')
    Node_descp=f"Node description : \n"
    for node in pos.keys():
        Node_descp+=f"Node {node} : \n"
        if node in leaf_nodes.keys() :
            if leaf_nodes[node] == -1 :
                # Node_descp+=f" a frontier node that can navigate to an unexplored space.\t"
                Node_descp+=f"  Frontier node: True\n"
            else :
                Node_descp+=f"  Frontier node: False\n"
        Node_descp+=f"  Frontier node: False\n"
        if node in objects_around.keys():
            # Node_descp += " Near this node, there are "
            Node_descp+=f"  Surrounding objects:\n"
            objs_around=objects_around[node]
            for obj_message in objs_around :
                Node_descp+=f"    There is {obj_message['obj']} is in direction: {obj_message['direction']}, distance: {obj_message['distance']} relativate to this node.\n"
                # if obj_id in object_area.keys() :
                #     area_id = object_area[obj_id]
                #     Node_descp+=f"\t\tobject {obj_name} in area {uppercase_letters[area_id]};\n"
                # else :
                #     Node_descp+=f"\t\tobject {obj_name};\n"
        if node_rooms[node] != "None" and node_rooms[node] is not None :
            Node_descp+=f"  Room: {node_rooms[node]}.\n"
        else :
            Node_descp+=f"  Room: Unknown.\n"
        if node in explored_nodes :
            Node_descp+=f"  Explored: True\n"
        else :
            Node_descp+=f"  Explored: False\n"
        Node_descp+=f"\n"
    candidate = list(pos.keys())
    candidate.remove(agent_node)
    nodes=f"The nodes candidate you select: {candidate}"
    USER = USER +Node_descp+nodes
    return USER

def find_nearby_objects_2d(grid_points, objects, radius,shape,minbound,size,return_index=False):
    result = {}
    object_points_2d = []
    object_indices = []
    for obj_idx,obj in enumerate(objects):
        obj_points = extractVoxelCenterfromVoxelmap(obj['voxel_index'],shape,minbound,size)
        object_points_2d.append(obj_points[:, :2])  # 取 (x, y)
        object_indices.extend([obj_idx] * len(obj_points))  # 记录物体索引
    # 1. 只取 (x, y) 坐标
    object_points_2d = np.vstack(object_points_2d)  # (总点数, 2)
    object_indices = np.array(object_indices)  # (总点数,)
    
    # 3. 构建 KDTree
    kdtree = KDTree(object_points_2d)

    # 4. 查询每个网格点
    
    for key,point in grid_points.items():
        l=[]
        indices = kdtree.query_ball_point(point, radius)  # 查询2D范围
        nearby_objects = set(object_indices[indices])  # 获取物体索引
        for obj_idx in nearby_objects :
            l.append(int(obj_idx) if return_index else getObjectName(objects[obj_idx]))
        result[key] = l
    return result
def find_nearest_landmark(object,grid_points,shape,minbound,size):
    obj_points = extractVoxelCenterfromVoxelmap(object['voxel_index'],shape,minbound,size)
    object_points_2d = np.mean(obj_points[:, :2],axis=0).reshape(1,-1)  # 取 (x, y)
    keys = np.array(list(grid_points.keys()))
    values = np.array(list(grid_points.values()))
    distances = np.linalg.norm(values[:, np.newaxis, :] - object_points_2d[np.newaxis, :, :], axis=2)
    min_distances = np.min(distances, axis=1)
    nearest_index = np.argmin(min_distances)
    return keys[nearest_index],values[nearest_index]
def find_nearest_node(object,node,points,shape,minbound,size):
    obj_points = extractVoxelCenterfromVoxelmap(object['voxel_index'],shape,minbound,size)
    object_points_2d = np.mean(obj_points[:, :2],axis=0).reshape(1,-1)  # 取 (x, y)
    keys = np.array(node)
    values = np.array(points)
    distances = np.linalg.norm(values[:, np.newaxis, :] - object_points_2d[np.newaxis, :, :], axis=2)
    min_distances = np.min(distances, axis=1)
    nearest_index = np.argmin(min_distances)
    return keys[nearest_index],values[nearest_index]
