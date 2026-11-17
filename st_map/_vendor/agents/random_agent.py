from uninavbench.agent import Agent
from uninavbench.challenge import ALLOWED_ACTIONS
import random
import re
import numpy as np
from collections import Counter
from st_map._vendor.model.Semantic_Mapping_SG import Semantic_Mapping_SG
from st_map._vendor.slam.utils import extractVoxelCenterfromVoxelmap
from st_map._vendor.slam.slam_classes import MapObjectList
import open3d as o3d
import os
import torch.nn as nn
import st_map._vendor.robothor_challenge.utils as du
import time
import networkx as nx
import matplotlib.pyplot as plt
from st_map._vendor.utils.local_planner import Local_Planner,Select_now_index
import torch
from st_map._vendor.utils.voronoi import projectCurrentAgentLoc
from st_map._vendor.scenegraph.querypath import find_nearby_objects_2d,QueryByLLM,find_nearest_landmark,find_nearest_node,select_subgraph_nodes,format_nearby_objects
from st_map._vendor.utils.constants import name_to_id,categories_mapping_to_openseed
# from scenegraph.llm import LLM,LLMQwQ
from st_map._vendor.vlm.llm import LLMQwQ
from st_map._vendor.utils.voronoi_ai2thor import * 
import heapq
import math
from ai2thor.controller import Controller
from PIL import Image
import pickle as pkl
import gzip
class SimpleRandomAgent(Agent):
    def __init__(self,controller,memory_cfg=None,runtime_cfg=None, **kwargs):
        # 初始化父类的 __init__，以确保原始初始化过程不受影响
        super().__init__(**kwargs)
        # print(du.getSize(controller))
        self.size,self.min_bound = du.getSize(controller)
        # print(self.size,self.min_bound)
        self.reachable_voxel_centers = None
        self.local_planner = None
        self.actions = None 
        self.device = torch.device("cuda:0" )
        self.fov = controller.initialization_parameters.get("fieldOfView")  # 默认90度
        self.rotateStepDegrees = controller.initialization_parameters.get("rotateStepDegrees", 90)  # 默认90度
        self.controller = controller
        # 自定义初始化参数
        self.sem_map_module = Semantic_Mapping_SG(self.fov,self.size,self.min_bound,memory_cfg,runtime_cfg).to(self.device)
        self.memory_cfg = memory_cfg or {}
        self.sem_map_module.eval()
        self.full_map = torch.zeros([1,5,self.size[0],self.size[1]]).cuda()
        # self.llm=LLM()
        self.llm = LLMQwQ(**(runtime_cfg or {}).get("llm", {}))
        self.global_position=None
        self.global_position_history=[]
        self.room_message={}
        self.graph=None
        self.finished = np.zeros(1)
        self.wait_env = np.zeros(1)
        self.planner_pose_inputs = np.zeros((1, 7))
        self.g_sum_rewards = 1 
        self.save_path=None
        self.visualize=False
        self.print_images=True
        self.replan=False
        self.found_goal = None
        self.global_pcd = o3d.geometry.PointCloud()
        self.reachable_2D_pos=None
        self.explored_node= []
        self.unexplored_node = []
        self.checked_target = []
        self.node_dict = {}
        self.repitch = 0
        self.debug = (runtime_cfg or {}).get("debug_artifacts", False)
        self.visited_cells = set()
    def reset(self,reachable_voxel_centers,path,target_obejct):
        self.update(reachable_voxel_centers)
        self.replan = False
        self.found_goal = None
        self.actions = None 
        self.checked_target = []
        self.target_feature = self.sem_map_module.textmodel.compute_target_feat(target_obejct)
        self.sem_map_module.update_similarity()
        for obj in self.sem_map_module.objects:
            obj["target_judge"] = True
        self.save_path = path
        os.makedirs(os.path.join(path,"depth"), exist_ok=True)
        os.makedirs(os.path.join(path,"sg"), exist_ok=True)
        if not os.path.exists(os.path.join(self.save_path,"depth/")):
            os.makedirs(os.path.join(self.save_path,"depth/"))
        if not os.path.exists(os.path.join(self.save_path,"sg/")):
            os.makedirs(os.path.join(self.save_path,"sg/"))
        if self.repitch >0 :
            action = ['LookUp']
            action = action * self.repitch
        elif self.repitch <0:
            action = ['LookDown']
            action = action * (-self.repitch)
        else:
            action = None
        self.repitch = 0
        return action
    def object_views(self,obj,frames):
        """(image path, bounding box) of an object's detections in the given frames."""
        boxes = dict(zip(obj['image_idx'],obj['xyxy']))
        return [(os.path.join(self.save_path,"image/",str(f)+'.jpg'),boxes.get(f)) for f in frames]
    def init(self,reachable_voxel_centers,path,target_obejct):
        self.target_feature = self.sem_map_module.textmodel.compute_target_feat(target_obejct)
        self.reachable_voxel_centers = reachable_voxel_centers
        self.local_planner = Local_Planner(self.rotateStepDegrees)
        self.local_planner.create_graph(self.reachable_voxel_centers)
        self.reachable_2D_pos =  np.array([[item['x'],item['z']] for item in self.reachable_voxel_centers])
        self.unexplored_node = list(range(len(self.reachable_2D_pos)))
        self.replan = False
        self.found_goal = None
        self.save_path = path
        os.makedirs(os.path.join(path,"depth"), exist_ok=True)
        os.makedirs(os.path.join(path,"sg"), exist_ok=True)
        # if not os.path.exists(os.path.join(self.save_path,"depth/")):
        #     os.makedirs(os.path.join(self.save_path,"depth/"))
        if not os.path.exists(os.path.join(self.save_path,"sg/")):
            os.makedirs(os.path.join(self.save_path,"sg/"))
    def update(self,reachable_voxel_centers):
        self.reachable_voxel_centers = reachable_voxel_centers
        self.local_planner = Local_Planner(self.rotateStepDegrees)
        self.local_planner.create_graph(self.reachable_voxel_centers)
        reachable_2D_pos_new =  np.array([[item['x'],item['z']] for item in self.reachable_voxel_centers])
        remain_index,correspond_new_index = Select_now_index(self.reachable_2D_pos,reachable_2D_pos_new)
        self.explored_node = list(correspond_new_index[np.nonzero(np.isin(remain_index, np.intersect1d(np.array(self.explored_node),remain_index)))[0]])
        self.reachable_2D_pos = reachable_2D_pos_new
        self.explored_node = list(set(self.explored_node).intersection(set(range(len(self.reachable_2D_pos)))))
        self.unexplored_node = list(set(range(len(self.reachable_2D_pos))) - set(self.explored_node))
        
    def act(self, observations):
        result={
            'image':observations['event'].frame,
            'depth':observations['event'].depth_frame,
            'step':observations['step'],
            'agent':observations['event'].metadata['agent'],
        }
        if self.debug:
            open(os.path.join(self.save_path,"sg/")+f"{str(observations['step'])}.pkl.gz", 'wb').write(gzip.compress(pkl.dumps(result)))
        event = observations['event']
        # depth = event.depth_frame
        step = observations["step"]
        # plt.imsave(os.path.join(self.save_path,"rgb/")+str(step)+".png",rgb.astype(np.float32)/255)
        # plt.close()
        # plt.imshow(depth,cmap='plasma') 
        # plt.axis('off')
        # plt.savefig(os.path.join(self.save_path,"depth/")+str(step)+".png", bbox_inches="tight", pad_inches=0)
        # plt.close()
        start = time.time()
        self.full_map,_,pos_map,current_pcd=self.sem_map_module.getGlobalMap(self.full_map,observations,observations['step'],self.save_path,self.global_position)
        end = time.time()
        print("getGlobalMap time:",end-start)
        if self.found_goal is not None and not self.found_goal['exists_in_SG']:
            # the confirmed goal has been removed from the scene graph, search again
            print("found goal disappeared:",self.found_goal['class_name'][-1])
            self.found_goal = None
            self.actions = None
        print("instruction:",observations['instruction'])
        start = time.time()
        # print("before: explored:",len(self.explored_node),"unexplored:",len(self.unexplored_node),"all:",len(self.explored_node)+len(self.unexplored_node))
        self.explored_node,self.unexplored_node,self.node_dict = update_explored_sets(current_pcd,self.explored_node,self.unexplored_node,self.node_dict,self.reachable_2D_pos)
        end = time.time()
        # print("after: explored:",len(self.explored_node),"unexplored:",len(self.unexplored_node),"all:",len(self.explored_node)+len(self.unexplored_node))
        start = time.time()
        if self.found_goal is None and len(self.sem_map_module.target_similarity_with_objects) > 0:
            # import pdb
            # pdb.set_trace()
            similarity = np.array(self.sem_map_module.target_similarity_with_objects)[:,0]
            times = np.array(self.sem_map_module.target_similarity_with_objects)[:,1]
            indices = np.where(similarity > 0.7)[0]
            if len(indices) > 0 :
                similarity1 = similarity[indices] 
                times1 = times[indices]
                sorted_indices = sorted(range(len(similarity1)), key=lambda i: (-similarity1.tolist()[i], -times1.tolist()[i]))
                for i in indices[sorted_indices] :
                    if self.sem_map_module.objects[i]['target_judge'] and self.sem_map_module.objects[i]['state_dict'][step] !=2 :
                        #judge only one 
                        # select_idx = self.sem_map_module.find_Maximum_View(i,self.sem_map_module.objects[i])[0]
                        # judge = self.sem_map_module.vlm.judge_obj_in_img(os.path.join(self.save_path,"image/",str(select_idx)+'.jpg'),observations['object_goal'])
                        #judge three images
                        start = time.time()
                        select_idx = self.sem_map_module.find_Maximum_View(i,self.sem_map_module.objects[i])
                        print("find Maximum View time:",time.time()-start)
                        start = time.time()
                        views = self.object_views(self.sem_map_module.objects[i],select_idx)
                        judge = self.sem_map_module.vlm.judge_obj_in_imglist([v[0] for v in views],observations['instruction'],observations["image_task"],[v[1] for v in views])
                        print("judge time:",time.time()-start)
                        print("similarity:",similarity[i],"instruction:",observations['instruction'],"object_name:",self.sem_map_module.objects[i]['class_name'],"select_idx:",select_idx,"judge:",judge)
                        if judge is None:
                            raise RuntimeError("ST-Map VLM target verification failed; check ST_MAP_VLM settings")
                        if ("True" in judge  or "true" in judge) and ('False' not in judge and 'false' not in judge):
                            self.found_goal = self.sem_map_module.objects[i]
                            self.replan = True
                            break
                        else :
                            self.sem_map_module.objects[i]['target_judge'] = False
                            self.replan = True
        end = time.time()
        goal = observations["instruction"]  # str : e.g. "AlarmClock"
        ### given the current position and the target postion, use this local planner
        print(event.metadata["lastActionSuccess"])
        if event.metadata["lastActionSuccess"] in (False, 'False'):
            self.actions = ['Stop']
        if self.actions is None or len(self.actions) == 0 or (self.found_goal is not None and self.replan) :
            ### state transition
            # needs surrounding objects and landmarks
            if self.found_goal is not None  and self.graph is not None:
                #### change the graph or set the coarse and fine graph
                self.graph,history_explored = generate_voronoi(self.explored_node,self.unexplored_node,self.reachable_2D_pos,self.sem_map_module.pcd,self.global_position_history,self.save_path+"plt1/",observations['step'])
                landmark_goal,landmark_location = find_nearest_node(self.found_goal,self.explored_node,self.reachable_2D_pos[self.explored_node],self.size,self.min_bound,0.25)
                # landmark_goal,landmark_location = find_nearest_landmark(self.found_goal,nx.get_node_attributes(self.graph, 'pos'),self.size,self.min_bound,0.25)
                agent_location = np.array([observations['event'].metadata['agent']['position']['x'],observations['event'].metadata['agent']['position']['z']])
                if np.linalg.norm(agent_location-landmark_location) <= 0.5 :
                    ## judge the angle and view
                    obj_points = extractVoxelCenterfromVoxelmap(self.found_goal['voxel_index'],self.size,self.min_bound,0.25)
                    self.actions,self.repitch = self.local_planner.rotation_to_goal(obj_points,int(event.metadata['agent']['rotation']['y']),event.metadata['agent']['position'],event.metadata['cameraPosition']['y'],self.fov,-int(event.metadata['agent']['cameraHorizon']))
                    print("found goal:",goal,"step:",step,self.actions)
                    self.replan = False
                    # self.actions.append(ALLOWED_ACTIONS[0])
                else :
                    # print("judgement:",len(self.actions),event.metadata["lastActionSuccess"],(self.found_goal is not None and self.replan))
                    # start, goal = self.local_planner.get_farthest_node_from_point((event.metadata['agent']['position']['x'],event.metadata['agent']['position']['y'],event.metadata['agent']['position']['z']),np.array(self.graph.nodes[landmark_goal].get('pos',None)))
                    start, goal = self.local_planner.get_farthest_node_from_point((event.metadata['agent']['position']['x'],event.metadata['agent']['position']['y'],event.metadata['agent']['position']['z']),np.array(landmark_location))
                    print("goal:",goal)
                    current_rotation = int(event.metadata['agent']['rotation']['y'])
                    print("current_rotation:",current_rotation)
                    self.actions = self.local_planner.act(start,goal,current_rotation,self.save_path if self.debug else None,observations['step'],self.explored_node,self.unexplored_node)
                    if len(self.actions) == 0 :
                        obj_points = extractVoxelCenterfromVoxelmap(self.found_goal['voxel_index'],self.size,self.min_bound,0.25)
                        self.actions,self.repitch = self.local_planner.rotation_to_goal(obj_points,int(event.metadata['agent']['rotation']['y']),event.metadata['agent']['position'],event.metadata['cameraPosition']['y'],self.fov,-int(event.metadata['agent']['cameraHorizon']))
                    print(f"start:{start}; goal:{goal}; current_rotation:{current_rotation}; actions:{self.actions}.")
                    self.replan = False
            else:
                self.graph,history_explored = generate_voronoi(self.explored_node,self.unexplored_node,self.reachable_2D_pos,self.sem_map_module.pcd,self.global_position_history,self.save_path+"plt1/",observations['step'])
                if self.graph is not None and  len(list(nx.get_node_attributes(self.graph, 'pos').values())) > 0:
                    if len(self.sem_map_module.objects) > 0 : 
                        objects_nearby_idx = find_nearby_objects_2d(nx.get_node_attributes(self.graph, 'pos'),self.sem_map_module.objects,0.5,self.size,self.min_bound,0.25,return_index=True)
                    else :
                        objects_nearby_idx = None
                    similarity = self.sem_map_module.target_similarity_with_objects
                    obj_sims = [s[0] for s in similarity] if len(similarity) == len(self.sem_map_module.objects) else [0.0]*len(self.sem_map_module.objects)
                    objects_nearby_points = format_nearby_objects(objects_nearby_idx,self.sem_map_module.objects,obj_sims,self.memory_cfg.get('subgraph_objects_per_node',6))
                    frontiers = getFrontiers(self.graph,self.reachable_2D_pos[self.unexplored_node])
                    ## project current agent location to graph
                    agent_location = np.array([observations['event'].metadata['agent']['position']['x'],observations['event'].metadata['agent']['position']['z']])
                    # visited places as grid cells: bounded by the house, not by the episode length
                    cell = (round(agent_location[0]/0.25),round(agent_location[1]/0.25))
                    if cell not in self.visited_cells:
                        self.visited_cells.add(cell)
                        self.global_position_history.append(agent_location)
                    self.graph,agent_location_node = projectCurrentAgentLoc(agent_location,self.graph)
                    history_context = self.sem_map_module.retrieveHistoryContext(os.path.join(self.save_path,"image/"))
                    # extract the instruction-related subgraph so the prompt does not grow with the explored area
                    k_nodes = self.memory_cfg.get('subgraph_nodes',8)
                    keep_nodes = select_subgraph_nodes(self.graph,agent_location_node,frontiers,objects_nearby_idx,obj_sims,k_nodes,self.memory_cfg.get('subgraph_frontiers',4)) if k_nodes > 0 else None
                    user,candidate= QueryByLLM(observations['instruction'],agent_location_node,self.graph,history_explored,frontiers,objects_nearby_points,history_context,step,keep_nodes)
                    print("node candidates:",candidate)
                    if len(candidate) >= 1 :
                        start=time.time()
                        note = None
                        for attempt in range(3):
                            try:
                                response,found_elements = self.llm.query_node_txt(user,note)
                                print("llm reply:",response)
                                if found_elements in candidate:
                                    break
                                note = f"Node {found_elements} is not in Candidates. Choose one of {candidate}."
                            except Exception as ex: # rate limit, malformed reply
                                print("llm query failed:",ex)
                                time.sleep(1)
                        else:
                            # no valid answer: the nearest frontier keeps exploring (nearest candidate without one)
                            pool = [n for n in candidate if n in frontiers] or candidate
                            here = np.array(self.graph.nodes[agent_location_node]['pos'])
                            found_elements = min(pool,key=lambda n:np.linalg.norm(np.array(self.graph.nodes[n]['pos'])-here))
                            response = {'reason':'fallback: no valid LLM choice, nearest frontier'}
                            print("WARNING planner fallback to node",found_elements)
                        end = time.time()
                        print("qwq llm query time:",end-start)
                        print("selection:",found_elements,"reason:",response['reason'])
                        ### random selection ###
                        # if len(frontiers) > 0 :
                        #     found_elements = random.choice(frontiers)
                        # else :
                        #     found_elements = random.choice(candidate)
                        #### random selection ####
                        
                        start, goal = self.local_planner.get_farthest_node_from_point((event.metadata['agent']['position']['x'],event.metadata['agent']['position']['y'],event.metadata['agent']['position']['z']),np.array(self.graph.nodes[found_elements].get('pos',None)))
                        # print("goal:",goal)
                        current_rotation = int(event.metadata['agent']['rotation']['y'])
                        # print("current_rotation:",current_rotation)
                        self.actions = self.local_planner.act(start,goal,current_rotation,self.save_path if self.debug else None,observations['step'],self.explored_node,self.unexplored_node)
                        # print(f"start:{start}; goal:{goal}; current_rotation:{current_rotation}; actions:{self.actions}.")
                    else :
                        self.actions = ['RotateRight','RotateRight']
                else :
                    self.actions = ['RotateRight','RotateRight']
            # results = {
            #     'objects': self.sem_map_module.objects.to_serializable(),
            #     'bg_objects': MapObjectList([_ for _ in self.sem_map_module.bg_objects.values() if _ is not None]).to_serializable(),
            # }
            # open(os.path.join(self.save_path,"pcd/")+str(step)+".pkl.gz", 'wb').write(gzip.compress(pkl.dumps(results)))
        if len(self.actions) != 0 :
            action = self.actions.pop(0)
        else:
            action = "MoveAhead"
        return action,None
        
def build():
    agent_class = SimpleRandomAgent
    agent_kwargs = {}
    # resembles SimpleRandomAgent(**{})
    render_depth = True
    return agent_class, agent_kwargs, render_depth