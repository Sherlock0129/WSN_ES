import random
import numpy as np


# def opportunistic_routing(nodes, source_node, destination_node, max_hops=3, t=100, receive_WET=0):
#     """
#     Perform Opportunistic Energy Cooperative Routing (OECR) to find the best path
#     from source_node to destination_node, considering energy and distance factors.
#
#     :param nodes: List of SensorNode objects
#     :param source_node: The source node from which data starts
#     :param destination_node: The destination node where data needs to be delivered
#     :param max_hops: The maximum number of hops allowed in the routing path
#     :param t: Time step in minutes (for energy generation)
#     :param receive_WET: The energy received through Wireless Energy Transfer (WET) (in Joules).
#     :return: The path (list of nodes) from source to destination
#     """
#     path = [source_node]
#     current_node = source_node
#     hops = 0
#
#     while current_node != destination_node and hops < max_hops:
#         best_node = None
#         best_score = float('-inf')  # Initialize with a very low score
#
#         # Evaluate the best next hop node based on energy, distance, and other factors
#         for node in nodes:
#             if node == current_node or node in path:
#                 continue
#
#             # Calculate the distance between the current node and the destination node
#             distance = current_node.distance_to(destination_node)
#
#             # Calculate the energy score (higher energy is better)
#             energy_score = node.current_energy / node.capacity  # Normalize energy
#             distance_score = 1 / (distance + 1)  # Minimize distance, larger distances get smaller score
#
#             # Energy generation score (if the node has solar panels)
#             energy_generation_score = node.energy_generation(t=t, receive_WET=receive_WET) / node.capacity  # Call energy_generation() method
#
#             # Total score for the node
#             score = 0.5 * energy_score + 0.3 * distance_score + 0.2 * energy_generation_score
#
#             # Update the best node if the current node has a better score
#             if score > best_score:
#                 best_score = score
#                 best_node = node
#
#         if best_node is None:
#             print("No valid next hop found, routing failed!")
#             break
#
#         path.append(best_node)
#         current_node = best_node
#         hops += 1
#
#     if current_node == destination_node:
#         print(f"Routing successful: Path found with {len(path)} hops.")
#         # plotter.plot_routing_path(path)  # 绘制路由路径
#         return path
#     else:
#         print(f"Routing failed after {max_hops} hops.")
#         return None
import heapq
import math

import heapq
import math

import heapq
import math

def opportunistic_routing(nodes, source_node, destination_node, max_hops=5, t=0, receive_WET=0):
    """
    Use Dijkstra algorithm to find shortest path (by distance) between source and destination node.
    """

    # 构建邻接表：连接所有节点（无距离限制）
    neighbor_map = {node.node_id: [] for node in nodes}
    node_dict = {node.node_id: node for node in nodes}

    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i != j:
                d = node_i.distance_to(node_j)
                neighbor_map[node_i.node_id].append((node_j.node_id, d))

    # 初始化优先队列： (distance, hops, current_node_id, path_id_list)
    queue = [(0, 0, source_node.node_id, [source_node.node_id])]
    visited = set()

    while queue:
        total_dist, hops, current_id, path_ids = heapq.heappop(queue)

        if current_id == destination_node.node_id:
            # 转换路径为 SensorNode 列表
            return [node_dict[nid] for nid in path_ids]

        if (current_id, hops) in visited or hops >= max_hops:
            continue
        visited.add((current_id, hops))

        for neighbor_id, dist in neighbor_map[current_id]:
            if neighbor_id not in path_ids:  # 避免回环
                heapq.heappush(queue, (
                    total_dist + dist,
                    hops + 1,
                    neighbor_id,
                    path_ids + [neighbor_id]
                ))

    # print(f"[Routing Failed] No path from {source_node.node_id} to {destination_node.node_id} within {max_hops} hops.")
    return None



def update_node_scores(nodes, source_node, destination_node):
    """
    Update node scores based on energy and distance for routing decisions.

    :param nodes: List of SensorNode objects
    :param source_node: The source node from which data starts
    :param destination_node: The destination node where data needs to be delivered
    :return: A list of scores for each node
    """
    scores = []
    for node in nodes:
        if node == source_node:
            continue  # Skip the source node

        # Calculate distance to the destination node
        distance = node.distance_to(destination_node)

        # Calculate the energy score (higher energy is better)
        energy_score = node.current_energy / node.capacity

        # Calculate the distance score (lower distance is better)
        distance_score = 1 / (distance + 1)

        # Energy generation score (if the node has solar panels)
        energy_generation_score = node.energy_generation_rate / node.capacity_mAh

        # Combine the scores into a total score
        total_score = 0.5 * energy_score + 0.3 * distance_score + 0.2 * energy_generation_score

        # Append the score for this node
        scores.append((node, total_score))

    return scores


def select_best_next_hop(scores):
    """
    Select the best next hop based on the computed scores.

    :param scores: A list of tuples containing node and score
    :return: The node with the highest score
    """
    best_node = max(scores, key=lambda x: x[1])[0]  # Select the node with the highest score
    return best_node
