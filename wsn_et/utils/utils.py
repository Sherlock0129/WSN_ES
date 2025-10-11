import os
import json
import logging
from random import random

import numpy as np

# 设置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_json(file_path):
    """
    Reads a JSON file and returns the parsed data.

    :param file_path: The path to the JSON file.
    :return: Parsed data from the JSON file.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        logger.info(f"Successfully read JSON file: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None


def write_json(data, file_path):
    """
    Writes data to a JSON file.

    :param data: The data to write to the file.
    :param file_path: The path to the JSON file.
    """
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        logger.info(f"Successfully wrote data to JSON file: {file_path}")
    except Exception as e:
        logger.error(f"Error writing to file {file_path}: {e}")


def save_to_csv(data, file_path):
    """
    Saves data to a CSV file.

    :param data: The data to write to the CSV file.
    :param file_path: The path to the CSV file.
    """
    try:
        import csv
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        logger.info(f"Successfully saved data to CSV file: {file_path}")
    except Exception as e:
        logger.error(f"Error writing to CSV file {file_path}: {e}")


def load_network_parameters(file_path):
    """
    Loads network parameters from a JSON configuration file.

    :param file_path: The path to the JSON configuration file.
    :return: A dictionary with the network parameters.
    """
    data = read_json(file_path)
    if data is None:
        return {}
    return data


def load_initial_energy_distribution(num_nodes, min_energy, max_energy):
    """
    Generates an initial energy distribution for all nodes in the network.

    :param num_nodes: Total number of nodes.
    :param min_energy: Minimum energy a node can have.
    :param max_energy: Maximum energy a node can have.
    :return: List of initial energy values for each node.
    """
    initial_energy = np.random.uniform(min_energy, max_energy, num_nodes)
    return initial_energy


def normalize_energy(energy, min_energy, max_energy):
    """
    Normalize energy to be within the given bounds.

    :param energy: Energy value to normalize.
    :param min_energy: Minimum allowable energy.
    :param max_energy: Maximum allowable energy.
    :return: Normalized energy within the given bounds.
    """
    return np.clip(energy, min_energy, max_energy)


def generate_random_position(network_area):
    """
    Generate a random position for a node within the specified area.

    :param network_area: A tuple specifying the area dimensions (width, height).
    :return: A tuple representing the node's position (x, y).
    """
    width, height = network_area
    x = np.random.uniform(0, width)
    y = np.random.uniform(0, height)
    return (x, y)


def log_node_energy(node_id, current_energy, low_threshold, high_threshold):
    """
    Log the energy status of a node.

    :param node_id: The ID of the node.
    :param current_energy: The current energy of the node.
    :param low_threshold: The low energy threshold.
    :param high_threshold: The high energy threshold.
    """
    if current_energy < low_threshold:
        logger.warning(f"Node {node_id} energy below low threshold: {current_energy:.2f}")
    elif current_energy > high_threshold:
        logger.warning(f"Node {node_id} energy above high threshold: {current_energy:.2f}")
    else:
        logger.info(f"Node {node_id} energy: {current_energy:.2f} (Within normal range)")


def generate_random_id():
    """
    Generates a random ID for a node.

    :return: A random integer ID.
    """
    return random.randint(1000, 9999)


def validate_network_parameters(parameters):
    """
    Validates the network parameters to ensure they are within acceptable ranges.

    :param parameters: The network configuration parameters.
    :return: True if all parameters are valid, False otherwise.
    """
    valid = True
    if parameters.get("num_nodes") <= 0:
        logger.error("Number of nodes must be positive.")
        valid = False
    if not (0 <= parameters.get("low_threshold", 0) <= 1):
        logger.error("Low threshold must be between 0 and 1.")
        valid = False
    if not (0 <= parameters.get("high_threshold", 0) <= 1):
        logger.error("High threshold must be between 0 and 1.")
        valid = False
    if parameters.get("node_initial_energy") <= 0:
        logger.error("Initial energy must be positive.")
        valid = False
    return valid


def calculate_distance(node1, node2):
    """
    Calculate the Euclidean distance between two nodes.

    :param node1: The first node (x1, y1).
    :param node2: The second node (x2, y2).
    :return: The Euclidean distance between the nodes.
    """
    x1, y1 = node1
    x2, y2 = node2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def setup_logger(log_file="simulation.log"):
    """
    Set up the logger for the simulation.

    :param log_file: The path to the log file.
    """
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Logger setup complete.")

