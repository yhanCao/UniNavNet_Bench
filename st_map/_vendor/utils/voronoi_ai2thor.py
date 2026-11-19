from scipy.spatial import Voronoi
import numpy as np
import networkx as nx
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from scipy.spatial import Voronoi
import math
from st_map._vendor.utils.voronoi import simplify_graph,merge_nodes
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
    if len(points) == 0:
        print("Error: the point set is empty")
        return None
    
    try:
        # Voronoi diagram
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
    except Exception:
        return None
    

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
def update_explored_sets(pcd,explored_node,unexplored_node,node_dict,poses,cell_size=0.25):
    pcd_project = np.array(pcd.points)[:,:2]
    unexplored_np = poses[unexplored_node]
    group1_grid = get_grid_coordinates(unexplored_np, cell_size)
    group2_grid = get_grid_coordinates(pcd_project, cell_size)
    group2_grid = np.unique(group2_grid, axis=0)
    group2_grid_set = set(map(tuple, group2_grid))  # cells as tuples
    result = np.array([tuple(row) in group2_grid_set for row in group1_grid])
    changed = np.where(result)[0]
    node_new = np.array(unexplored_node)[changed]
    unexplored_node = list(set(unexplored_node)-set(node_new))
    explored_node = list(set(explored_node).union(set(node_new)))
    return explored_node,unexplored_node,node_dict
def euclidean_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
def add_new_nodes_with_condition(G, new_nodes):
    # for every new node
    pos = nx.get_node_attributes(G, 'pos') 
    explored_nodes=[]
    len(pos)
    for new_coord in new_nodes:
        # minimum distance to the existing nodes
        min_distance = float('inf')
        closest_node = None
        
        for existing_node in G.nodes(data=True):
            existing_coord = existing_node[1]['pos']
            distance = euclidean_distance(new_coord, existing_coord)
            
            if distance < min_distance:
                min_distance = distance
                closest_node = existing_node[0]
        
        # # minimum distance above the threshold: add the node and connect it
        # else :
        explored_nodes.append(closest_node)
    return G,list(set(explored_nodes))
def generate_voronoi(explored_nodes,unexplored_nodes,poses,pcd,history_nodes,plt_path,idx):
    contained_points = poses[explored_nodes]
    all_points = []
    union_poly = union_grid_boundary(contained_points, cell_size=0.25)
    if union_poly.geom_type == 'Polygon':
        coords = list(union_poly.exterior.coords)
        # drop a repeated closing point (polygon exteriors repeat the first point), which would disturb the Voronoi diagram
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        all_points.extend(coords)
        x, y = union_poly.exterior.xy
        # plt.fill would fill it
    elif union_poly.geom_type == 'MultiPolygon':
        for geom in union_poly.geoms:
            x, y = geom.exterior.xy
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
    The 8 neighboring grid cells of each of n points.
    
    Parameters:
    coordinates : np.ndarray
        array of shape (n, 2) with the points.
    size : float
        cell size (default 0.25).
        
    Returns:
    -----------
    adjacent_points : np.ndarray
        array of shape (n, 8, 2): the 8 neighbors of every point.
    """
    # offsets of the 8 directions
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
    
    # neighbors of every point
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

    
