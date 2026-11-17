from scipy.spatial import  Voronoi,voronoi_plot_2d
import numpy as np
import networkx as nx
from st_map._vendor.scenegraph.visSceneGraph import drawScenegraph,drawScenegraph2,draw_Voronoi
import matplotlib.colors as mcolors
import cv2
from scipy.ndimage import binary_erosion
from skimage.draw import line
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from scipy.spatial import Voronoi, voronoi_plot_2d
import math
from st_map._vendor.utils.voronoi import simplify_graph,merge_nodes
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
    if len(points) == 0:
        print("错误: 点集不能为空")
        return None
    
    try:
        # 尝试计算Voronoi图
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
    except Exception as e:
        return None
    

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
def update_explored_sets(pcd,explored_node,unexplored_node,node_dict,poses,cell_size=0.25):
    pcd_project = np.array(pcd.points)[:,:2]
    unexplored_np = poses[unexplored_node]
    group1_grid = get_grid_coordinates(unexplored_np, cell_size)
    group2_grid = get_grid_coordinates(pcd_project, cell_size)
    group2_grid = np.unique(group2_grid, axis=0)
    group2_grid_set = set(map(tuple, group2_grid))  # 将每个网格坐标变为元组并存入集合
    result = np.array([tuple(row) in group2_grid_set for row in group1_grid])
    changed = np.where(result)[0]
    node_new = np.array(unexplored_node)[changed]
    unexplored_node = list(set(unexplored_node)-set(node_new))
    explored_node = list(set(explored_node).union(set(node_new)))
    return explored_node,unexplored_node,node_dict
def euclidean_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
def add_new_nodes_with_condition(G, new_nodes):
    # 遍历新节点
    pos = nx.get_node_attributes(G, 'pos') 
    explored_nodes=[]
    n=len(pos)
    for new_coord in new_nodes:
        # 计算新节点与图中所有现有节点的最小距离
        min_distance = float('inf')
        closest_node = None
        
        for existing_node in G.nodes(data=True):
            existing_coord = existing_node[1]['pos']
            distance = euclidean_distance(new_coord, existing_coord)
            
            if distance < min_distance:
                min_distance = distance
                closest_node = existing_node[0]
        
        # # 如果最小距离大于10，添加新节点并连接
        # if min_distance > 20:
        #     G.add_node(n, pos=new_coord)  # 添加新节点到图中
        #     if G.degree(closest_node) == 1 :
        #         closest_node = list(G.neighbors(closest_node))[0]
        #     G.add_edge(n, closest_node)  # 连接到距离最小的节点
        #     explored_nodes.append(n)
        #     n=n+1
        # else :
        explored_nodes.append(closest_node)
    return G,list(set(explored_nodes))
def generate_voronoi(explored_nodes,unexplored_nodes,poses,pcd,history_nodes,plt_path,idx):
    contained_points = poses[explored_nodes]
    all_points = []
    union_poly = union_grid_boundary(contained_points, cell_size=0.25)
    if union_poly.geom_type == 'Polygon':
        coords = list(union_poly.exterior.coords)
        # 如果首尾重复(通常Polygon外边界首尾是同一点)，去除最后一个以免影响Voronoi
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        all_points.extend(coords)
        x, y = union_poly.exterior.xy
        # plt.plot(x, y, color='purple', label='Boundary')
        # 若想填充，也可调用plt.fill
        # plt.fill(x, y, facecolor='lightblue', alpha=0.4)
    elif union_poly.geom_type == 'MultiPolygon':
        for geom in union_poly.geoms:
            x, y = geom.exterior.xy
            # plt.plot(x, y, color='purple')
            coords = list(geom.exterior.coords)
            if len(coords) > 1 and coords[0] == coords[-1]:
                coords = coords[:-1]
            all_points.extend(coords)
    points = np.array(all_points)

    vor = voronoi_polygons_clipped(points,union_poly)
    if vor is not None :
        G = nx.Graph()
        for i, point in enumerate(vor.vertices):
            G.add_node(i, pos=(point[0],point[1]))
        G.add_edges_from(vor.ridge_vertices)

        pos_graph = nx.get_node_attributes(G, 'pos')
        poses_graph = np.array(list(pos_graph.values()))
        poses_graph = map_points_to_grid_centers(poses_graph)
        changed_poses={}
        for (node, _), coord in zip(pos_graph.items(), poses_graph):
            changed_poses[node] = (coord[0], coord[1])
        nx.set_node_attributes(G, changed_poses, 'pos')
        G_simplified = simplify_graph(G)
        G_simplified = merge_nodes(G_simplified)
        pos_graph = nx.get_node_attributes(G_simplified, 'pos')
        poses_graph = np.array(list(pos_graph.values()))
        G_simplified,history_explored = add_new_nodes_with_condition(G_simplified,history_nodes)
        return G_simplified,history_explored
    else :
        return None,None
def get_adjacent_grid_points(coordinates, size=0.25):
    """
    使用 numpy 获取 n 个坐标点的八个方位的邻接网格坐标。
    
    参数：
    coordinates : np.ndarray
        n x 2 的数组，包含 n 个坐标点。
    size : float
        网格的大小（默认为 0.25）。
        
    返回：
    -----------
    adjacent_points : np.ndarray
        n x 8 x 2 的数组，包含 n 个坐标点的八个邻接网格坐标。
    """
    # 八个方位的变化量
    directions = np.array([
        [0, size],        # north
        [0, -size],       # south
        [-size, 0],       # west
        [size, 0],        # east
        [-size, size],    # northwest
        [size, size],     # northeast
        [-size, -size],   # southwest
        [size, -size]     # southeast
    ])
    
    # 每个坐标点的邻接网格坐标
    adjacent_points = coordinates[:, np.newaxis, :] + directions
    
    return adjacent_points

def getFrontiers(graph,uncontained_points):
    pos_graph = nx.get_node_attributes(graph, 'pos')
    poses_graph = np.array(list(pos_graph.values()))
    keys = np.array(list(pos_graph.keys()))
    adjacent_grid_points = get_adjacent_grid_points(poses_graph)
    if len(uncontained_points) > 0 :
        distance = np.linalg.norm(adjacent_grid_points.reshape(-1,1,2)-uncontained_points[np.newaxis,:,:], axis=-1).reshape(-1,8,uncontained_points.shape[0])
        frontiers = np.unique(np.where(distance < 0.25)[0])
        print("*********total graph nodes:",len(poses_graph),"frontier graph nodes:",len(frontiers),"**********")
        return keys[frontiers].tolist()
    else :
        return []

    
# def getNearObjects(graph,objects,bg_objects):