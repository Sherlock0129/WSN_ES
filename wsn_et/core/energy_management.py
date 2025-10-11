import random
import numpy as np
# from SensorNode import SensorNode

def get_neighbors_for_node(node, nodes, distance_threshold=10):
    """
    Get the list of neighboring nodes within a certain distance threshold.

    :param node: The current node to find neighbors for.
    :param nodes: List of all nodes in the network.
    :param distance_threshold: The maximum distance for neighbors to be considered (in meters).
    :return: A list of neighboring SensorNode objects.
    """
    neighbors = []
    for other_node in nodes:
        if node != other_node and node.distance_to(other_node) <= distance_threshold:
            neighbors.append(other_node)
    return neighbors


def balance_energy(nodes):
    """
    Balance the energy in the network to ensure no node exceeds its energy thresholds.
    TEMPORARILY DISABLED - 临时禁用自动能量平衡

    :param nodes: List of SensorNode objects.
    """
    # 临时禁用自动能量平衡，避免非太阳能节点意外获取能量
    print("[DISABLED] balance_energy function temporarily disabled")
    return
    
    # 原始代码（已禁用）
    # for node in nodes:
    #     # Assuming we have a function to get neighbors for each node (this can be network topology-dependent)
    #     neighbors = get_neighbors_for_node(node, nodes)  # You should implement this function
    #
    #     # Check if the node's energy is below the low threshold
    #     if node.current_energy < node.low_threshold_energy:
    #         print(f"Node {node.node_id} energy below low threshold!")
    #         handle_low_energy(node, neighbors)
    #
    #     # Check if the node's energy is above the high threshold
    #     elif node.current_energy > node.high_threshold_energy:
    #         print(f"Node {node.node_id} energy above high threshold!")
    #         handle_high_energy(node, neighbors)


def handle_low_energy(node, neighbors):
    """
    Handle the case where a node's energy is below the low threshold.
    Possible actions: Trigger energy transfer from neighboring nodes or attempt to harvest energy.

    :param node: The SensorNode object with low energy.
    :param neighbors: List of neighboring nodes.
    """
    print(f"Node {node.node_id} attempting to harvest energy...")
    harvested_energy = node.energy_harvest(t=random.randint(0, 1440))  # Simulate energy harvesting at a random time
    print(f"  Harvested {harvested_energy:.2f} Joules")

    # If energy is still low, attempt to receive energy from other nodes
    if node.current_energy < node.low_threshold_energy:
        print(f"Node {node.node_id} still below low threshold, requesting energy from neighbors...")
        transfer_energy_from_neighbors(node, neighbors)



def handle_high_energy(node):
    """
    Handle the case where a node's energy is above the high threshold.
    Possible actions: Trigger energy transfer to neighboring nodes or offload excess energy.

    :param node: The SensorNode object with high energy.
    """
    # Try to transfer energy to neighboring nodes or offload excess energy
    print(f"Node {node.node_id} transferring excess energy...")
    transfer_energy_to_neighbors(node)


def transfer_energy_from_neighbors(node, neighbors):
    """
    Attempt to transfer energy from neighboring nodes to the current node.
    TEMPORARILY DISABLED - 临时禁用自动能量传输

    :param node: The SensorNode object that needs energy.
    :param neighbors: List of neighboring nodes from which energy can be transferred.
    """
    # 临时禁用自动能量传输，避免非太阳能节点意外获取能量
    print(f"  [DISABLED] Node {node.node_id} energy transfer request ignored")
    return
    
    # 原始代码（已禁用）
    # for neighbor in neighbors:
    #     if neighbor.current_energy > neighbor.low_threshold_energy:
    #         energy_transfer = min(neighbor.current_energy - neighbor.low_threshold_energy, 50)
    #         node.current_energy += energy_transfer
    #         neighbor.current_energy -= energy_transfer
    #         node.record_transfer(received=energy_transfer)
    #         neighbor.record_transfer(transferred=energy_transfer)
    #         print(f"  Node {node.node_id} received {energy_transfer:.2f} Joules from Node {neighbor.node_id}")


def transfer_energy_to_neighbors(node):
    """
    Attempt to transfer excess energy to neighboring nodes.

    :param node: The SensorNode object that has excess energy.
    """
    # For simplicity, assume random neighboring nodes (in real applications, this would depend on network topology)
    neighbors = random.sample(node.network.nodes, 3)  # Randomly select 3 neighboring nodes for energy transfer
    for neighbor in neighbors:
        if neighbor.current_energy < neighbor.high_threshold_energy:
            energy_transfer = min(node.current_energy - node.high_threshold_energy, 50)
            node.current_energy -= energy_transfer
            neighbor.current_energy += energy_transfer
            node.record_transfer(transferred=energy_transfer)
            neighbor.record_transfer(received=energy_transfer)
            print(f"  Node {node.node_id} transferred {energy_transfer:.2f} Joules to Node {neighbor.node_id}")


def energy_threshold_check(node):
    """
    Check if a node's energy is below its low threshold or above its high threshold.

    :param node: The SensorNode object to check.
    :return: True if the node's energy is within the threshold, False otherwise.
    """
    if node.current_energy < node.low_threshold_energy:
        print(f"Node {node.node_id} is below the low energy threshold!")
        return False
    elif node.current_energy > node.high_threshold_energy:
        print(f"Node {node.node_id} is above the high energy threshold!")
        return False
    return True


def optimize_energy_distribution(nodes):
    """
    Optimize energy distribution across the network. The goal is to ensure that nodes' energy
    consumption is balanced and that no node is depleted too quickly.

    :param nodes: List of SensorNode objects.
    """
    # For now, we simply balance the energy by checking energy thresholds.
    # This function can be expanded with more advanced energy optimization algorithms.
    balance_energy(nodes)

