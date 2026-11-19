import numpy as np
import networkx as nx
import matplotlib.colors as mcolors
import math
css4_colors = mcolors.TABLEAU_COLORS
color_proposals = [list(mcolors.hex2color(color)) for color in css4_colors.values()]
def remove_isolated_nodes(G):
    isolated_nodes = [node for node, degree in G.degree() if degree == 0]
    G.remove_nodes_from(isolated_nodes)
    return G
def simplify_graph(G):
    # copy, so the graph is not modified while iterating
    H = G.copy()
    # all intermediate nodes of degree 2
    nodes_to_remove = [node for node in H.nodes() if H.degree(node) == 2]
    
    for node in nodes_to_remove:
       while H.degree(node) == 2:  # the node still has degree 2
            neighbors = list(H.neighbors(node))
            if len(neighbors) == 2:
                # connect the node's two neighbors
                H.add_edge(neighbors[0], neighbors[1])
                # remove the node's edges
                H.remove_edge(node, neighbors[0])
                H.remove_edge(node, neighbors[1])
        # remove the intermediate node
    
    # # remove redundant edges (may occur)
    H=remove_isolated_nodes(H)
    return H
def find_non_leaf_nodes(graph):
    non_leaf_nodes = [node for node in graph.nodes() if len(list(graph.neighbors(node))) > 1]
    return non_leaf_nodes

def euclidean_distance(pos, node1, node2):
    x1, y1 = pos[node1]
    x2, y2 = pos[node2]
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


#### merge to global  
def find_non_leaf_nodes(G):
    return [n for n, d in G.degree() if d > 1]


def euclidean_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

def merge_close_nodes(G,threshold,obs_map):
    nodes_to_merge = []
    pos = nx.get_node_attributes(G, 'pos')
    for node1 in list(G.nodes()):
        for node2 in list(G.nodes()):
            if node1 != node2 and euclidean_distance(pos[node1], pos[node2]) < threshold :
                nodes_to_merge.append((node1, node2))
    for node1, node2 in nodes_to_merge:
        if node1 in G.nodes() and node2 in G.nodes() :
            if G.degree[node1] >= G.degree[node2]:
                node_to_keep = node1
                node_to_merge = node2
            else:
                node_to_keep = node2
                node_to_merge = node1
            G = nx.contracted_nodes(G,node_to_keep,node_to_merge,self_loops=False)
    return G


#     # clear the boundary


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
def projectCurrentAgentLoc(position,G):
    # for every new node
    pos = nx.get_node_attributes(G, 'pos') 
    # minimum distance to the existing nodes
    min_distance = float('inf')
    closest_node = None
    for existing_node in G.nodes(data=True):
        existing_coord = existing_node[1]['pos']
        distance = euclidean_distance(position, existing_coord)
        
        if distance < min_distance:
            min_distance = distance
            closest_node = existing_node[0]
    
    # minimum distance above the threshold: add the node and connect it
    if min_distance > 10:
        G.add_node(len(pos), pos=position)  # add the new node
        G.add_edge(len(pos), closest_node)  # connect it to the nearest node
        return G,len(pos)
    else :
        return G,closest_node
def merge_nodes(G,threshold=1):
    pos_graph = nx.get_node_attributes(G, 'pos')
    points = np.array(list(pos_graph.values()))
    nodes = np.array(list(pos_graph.keys()))
    if len(points) < 2:  # nothing to merge (an empty graph gives a 1-D array)
        return G
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    triu_indices = np.triu_indices(distances.shape[0], k=0)  # k=0 includes the diagonal (k=1 would exclude it)
    distances[triu_indices] = np.inf
    threshold_value = threshold * np.sqrt(2)
    indices = np.where(distances <= threshold_value)
    if len(indices[0]) > 0 :
        for i, j in zip(indices[0], indices[1]):
            if nodes[i] in G.nodes() and nodes[j] in G.nodes() :
                if G.degree(nodes[i]) < G.degree(nodes[j]) :
                    G = nx.contracted_nodes(G,nodes[i],nodes[j],self_loops=False)
                else :
                    G = nx.contracted_nodes(G,nodes[j],nodes[i],self_loops=False)
    return G