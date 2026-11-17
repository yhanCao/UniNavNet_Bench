from ai2thor.controller import Controller
import networkx as nx
from math import sqrt
import math
import os
import matplotlib.pyplot as plt
import heapq
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
def Select_now_index(m_coords,n_coords):
    # 容差距离（欧几里得距离）
    tolerance = 1e-5

    # 建立 m_coords 的 KDTree
    tree = cKDTree(m_coords)

    # 查询 n_coords 中每个点在 m_coords 中的近邻（容差范围内）
    matched_pairs = []
    for j,point in enumerate(n_coords):
        indices = tree.query_ball_point(point, r=tolerance)
        for i in indices:
            matched_pairs.append((i, j)) 

    # 排序（按 m_index 排序，保持一致性）
    matched_pairs.sort()

    # 拆开为两个数组（如果你需要）
    m_indices = np.array([i for i, _ in matched_pairs])
    n_indices = np.array([j for _, j in matched_pairs])

    
    return m_indices,n_indices
def euclidean(u, v):
    return math.hypot(u[0] - v[0], u[1] - v[1])

def line_of_sight(G, u, v, obstacles=None):
    """
    Basic Bresenham line-of-sight for grid graphs.
    """
    x0, y0 = u
    x1, y1 = v

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1

    err = dx - dy

    while True:
        if (x0, y0) not in G:
            return False
        if obstacles and (x0, y0) in obstacles:
            return False
        if (x0, y0) == (x1, y1):
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return True

def theta_star(G, start, goal, heuristic=euclidean):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {start: start}
    g_score = {start: 0}

    obstacles = set(G.nodes) - set(G.nodes())

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            break

        for neighbor in G.neighbors(current):
            if line_of_sight(G, came_from[current], neighbor, obstacles):
                parent = came_from[current]
                cost = g_score[parent] + heuristic(parent, neighbor)
            else:
                parent = current
                cost = g_score[current] + heuristic(current, neighbor)

            if neighbor not in g_score or cost < g_score[neighbor]:
                came_from[neighbor] = parent
                g_score[neighbor] = cost
                f = cost + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f, neighbor))

    # Reconstruct path
    path = []
    node = goal
    if node not in came_from:
        return []  # No path
    while node != start:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    return path
def euclidean_heuristic(u, v, pos_dict):
    """欧几里得距离启发式函数"""
    pos1 = pos_dict[u]
    pos2 = pos_dict[v]
    return sqrt((pos1[0]-pos2[0])**2 + (pos1[2]-pos2[2])**2)
# 计算两个位置之间的欧几里得距离
def euclidean_distance(pos1, pos2):
    if isinstance(pos1, tuple) :
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[2] - pos2[2])**2)
    elif isinstance(pos1, dict):
        return math.sqrt((pos1['x'] - pos2['x'])**2 + (pos1['z'] - pos2['z'])**2)
class Local_Planner:
    def __init__(self, rotateStepDegrees):
        self.reachable_positions = None
        self.rotateStepDegrees = rotateStepDegrees
        self.graph = None
        self.pos_dict = None
        self.current_direction = None

    def heuristic(self, u, v):
        """Euclidean heuristic only — turn penalty moves to weight."""
        return euclidean_heuristic(u, v, self.pos_dict)

    def goal_to_paths(self, start, goal):
        self.pos_dict = nx.get_node_attributes(self.graph, 'pos')

        def modified_weight(u, v, d):
            """Penalize turns based on previous direction."""
            base_weight = d.get('weight', 1)

            # Current move direction
            dx = self.pos_dict[v][0] - self.pos_dict[u][0]
            dz = self.pos_dict[v][2] - self.pos_dict[u][2]
            if abs(dx) > abs(dz):
                move_dir = 90 if dx > 0 else 270
            else:
                move_dir = 0 if dz > 0 else 180

            # Get last direction
            if self.current_direction is not None:
                dir_diff = abs(move_dir - self.current_direction) % 360
                dir_diff = min(dir_diff, 360 - dir_diff)
                penalty_factor = 1 + (dir_diff / 90) ** 2  # Stronger penalty
                return base_weight * penalty_factor
            return base_weight

        try:
            path = nx.astar_path(
                self.graph,
                source=start,
                target=goal,
                heuristic=self.heuristic,
                weight=modified_weight
            )

            # Smooth path with line-of-sight
            path = self.smooth_path(path)

            # Update current_direction for next plan
            if len(path) > 1:
                self._update_direction(path[0], path[1])

            return path

        except nx.NetworkXNoPath:
            return None

    def smooth_path(self, path):
        """Remove redundant waypoints using line-of-sight."""
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self.line_of_sight(path[i], path[j]):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed

    def line_of_sight(self, u, v):
        """Simple grid line-of-sight using Bresenham."""
        x0, _, z0 = self.pos_dict[u]
        x1, _, z1 = self.pos_dict[v]

        dx = abs(int(x1 - x0))
        dz = abs(int(z1 - z0))
        sx = 1 if x1 > x0 else -1
        sz = 1 if z1 > z0 else -1
        err = dx - dz

        while True:
            node = (round(x0), 0, round(z0))
            if node not in self.graph.nodes:
                return False
            if node == v:
                break
            e2 = 2 * err
            if e2 > -dz:
                err -= dz
                x0 += sx
            if e2 < dx:
                err += dx
                z0 += sz

        return True

    def _update_direction(self, u, v):
        """Set the planner's last direction."""
        dx = self.pos_dict[v][0] - self.pos_dict[u][0]
        dz = self.pos_dict[v][2] - self.pos_dict[u][2]
        if abs(dx) > abs(dz):
            self.current_direction = 90 if dx > 0 else 270
        else:
            self.current_direction = 0 if dz > 0 else 180

    def _get_direction(self, u, v):
        """Helper for direction calculation."""
        dx = self.pos_dict[v][0] - self.pos_dict[u][0]
        dz = self.pos_dict[v][2] - self.pos_dict[u][2]
        if abs(dx) > abs(dz):
            return 90 if dx > 0 else 270
        else:
            return 0 if dz > 0 else 180

    def path_to_actions(self, path, initial_rotation):
        """Convert path to actions with robust rotations."""
        actions = []
        current_rotation = initial_rotation

        for i in range(1, len(path)):
            current = self.pos_dict[path[i - 1]]
            next_pos = self.pos_dict[path[i]]

            dx = next_pos[0] - current[0]
            dz = next_pos[2] - current[2]

            if abs(dx) < 1e-6 and abs(dz) < 1e-6:
                continue
            if abs(dx) > abs(dz):
                target_rotation = 90 if dx > 0 else 270
            else:
                target_rotation = 0 if dz > 0 else 180

            diff = (target_rotation - current_rotation + 360) % 360
            while abs(diff) > self.rotateStepDegrees / 2:
                if diff <= 180:
                    actions.append("RotateRight")
                    current_rotation = (current_rotation + self.rotateStepDegrees) % 360
                else:
                    actions.append("RotateLeft")
                    current_rotation = (current_rotation - self.rotateStepDegrees) % 360
                diff = (target_rotation - current_rotation + 360) % 360

            actions.append("MoveAhead")

        return actions

    def act(self, start, goal, current_rotation, save_pth, idx, explored_node, unexplored_node):
        paths = self.goal_to_paths(start, goal)
        if save_pth is not None:  # debug plot of every replanned path
            plt_pth = os.path.join(save_pth, "path/")
            os.makedirs(plt_pth, exist_ok=True)
            self.visualization_path(start, goal, paths, plt_pth, idx, explored_node, unexplored_node)
        actions = self.path_to_actions(paths, current_rotation)
        return actions
    def create_graph(self,reachable_positions):
        self.reachable_positions = reachable_positions
        self.graph = nx.Graph()
        for i, pos in enumerate(reachable_positions):
            self.graph.add_node(i,pos = (pos['x'], pos['y'], pos['z']))
        for i, pos1 in enumerate(reachable_positions):
            for j, pos2 in enumerate(reachable_positions):
                if i != j and euclidean_distance(pos1, pos2) <= 0.26:  # 设置相邻阈值
                    self.graph.add_edge(i,j,weight=euclidean_distance(pos1, pos2))

    def get_farthest_node_from_point(self, point,pose):
        # 从图中获取节点坐标属性: node -> (x, y)
        pos_dict = nx.get_node_attributes(self.graph, 'pos')
        goal = (pose[0],0.9,pose[1])
        # 定义欧几里得距离函数
        def euclidean_dist(p1, p2):
            return math.dist(p1, p2)
        
        # 1. 找到与给定坐标最近的节点
        #    假设图中所有节点都有 pos 属性
        closest_node = min(
            pos_dict.keys(),
            key=lambda node: euclidean_dist(pos_dict[node], point)
        )
        
        # 2. 计算从这个最近节点到图中所有节点的最短路径距离
        #    注意要用 weight='weight'（你的图里边的属性名）
        # dist_dict = nx.single_source_dijkstra_path_length(
        #     self.graph, source=closest_node, weight='weight'
        # )
        
        # # 3. 找到最远的节点
        # farthest_node = max(dist_dict, key=dist_dict.get)
        goal_node = min(
            pos_dict.keys(),
            key=lambda node: euclidean_dist(pos_dict[node], goal)
        )
        return closest_node,goal_node
    def visualization_path(self,current_position,goal_position,path,plt_pth,idx,explored_node,unexplored_node):
        erx = [p["x"] for i,p in enumerate(self.reachable_positions) if i in explored_node]
        erz = [p["z"] for i,p in enumerate(self.reachable_positions) if i in explored_node]
        urx = [p["x"] for i,p in enumerate(self.reachable_positions) if i in unexplored_node]
        urz = [p["z"] for i,p in enumerate(self.reachable_positions) if i in unexplored_node]
        rx1 = [self.pos_dict[p][0] for p in path]
        rz1 = [self.pos_dict[p][2] for p in path]
        # 转化为动作序列
        plt.figure(figsize=(20,20))
        plt.scatter(erx, erz, c='blue', marker='s', s=30, label='Reachable')
        plt.scatter(urx, urz, c='orange', marker='s', s=30, label='Unreachable')
        plt.scatter(rx1,rz1,c='purple', marker='s', s=30, label='Path')
        plt.scatter([self.pos_dict[current_position][0]],[self.pos_dict[current_position][2]],c='green', marker='s', s=30, label='Start Position')
        plt.scatter([self.pos_dict[goal_position][0]],[self.pos_dict[goal_position][2]],c='red', marker='s', s=30, label='Goal Position')
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.title("Paths in RoboThor")
        plt.legend(loc='upper left')
        plt.axis("equal")  # 保持等比例
        plt.savefig(plt_pth+str(idx)+".png")
        plt.close()
    def rotation_to_goal(self,point_cloud,current_rotation,current_position,agent_height,fov,current_pitch):
        repitch = 0
        actions = []
        # 计算目标物体的中心点
        center = np.mean(point_cloud, axis=0)
        min_y = np.min(point_cloud,axis=0)[2]
        max_y = np.max(point_cloud,axis=0)[2]
        # 计算目标物体相对于机器人的水平方向角度
        dx = center[0] - current_position['x']
        dz = center[1] - current_position['z']
        angle_rad = math.atan2(dx, dz)  # 注意x,z的顺序决定了角度的定义
        target_rotation = (math.degrees(angle_rad))
        while current_rotation < target_rotation-self.rotateStepDegrees/2 or current_rotation > target_rotation+self.rotateStepDegrees/2:
            # print("current_rotation:",current_rotation,"target_rotation:",target_rotation)
            # 计算旋转差值 (顺时针 or 逆时针)
            diff = (target_rotation - current_rotation + 360) % 360

            # 如果差值为 0，说明已经对准朝向
            if diff > -self.rotateStepDegrees and diff < self.rotateStepDegrees :
                break

            # 如果差值 <= 180，则顺时针转(右转)比较近，否则逆时针(左转)比较近
            if diff <= 180:
                # 顺时针 30°
                actions.append("RotateRight")
                current_rotation = (current_rotation + self.rotateStepDegrees) % 360
            else:
                # 逆时针 30°
                actions.append("RotateLeft")
                current_rotation = (current_rotation - self.rotateStepDegrees) % 360
        if min_y < agent_height :
            dy = min_y - agent_height # y 是高度
        elif max_y > agent_height :
            dy = max_y - agent_height
        else :
            dy = center[2] - agent_height
        distance_2d = math.sqrt(dx**2 + dz**2)  # 水平距离
        pitch_rad = math.atan2(dy, distance_2d)
        target_pitch = math.degrees(pitch_rad)

        # 检查是否需要调整俯仰角（上下看）
        pitch_threshold = 30  # 视野范围的一半
        if target_pitch <  -pitch_threshold:
            while target_pitch < -pitch_threshold:
                actions.append("LookDown")  # 物体在视野上方，向上看
                target_pitch += 30
                repitch +=1
        elif target_pitch >  + pitch_threshold:
            while target_pitch >  + pitch_threshold:
                actions.append("LookUp")  # 物体在视野下方，向下看
                target_pitch -= 30
                repitch -= 1
        actions.append("Stop")
        return actions,repitch
