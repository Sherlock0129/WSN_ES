import math

import numpy as np


class SensorNode:
    def __init__(self, node_id, initial_energy, low_threshold, high_threshold,
                 position, has_solar=True, **kwargs):
        """
        Initialize the sensor node with energy and other parameters.

        :param node_id: The unique ID for the sensor node.
        :param initial_energy: Initial energy of the sensor node (in Joules).
        :param low_threshold: Low energy threshold (percentage of the total capacity).
        :param high_threshold: High energy threshold (percentage of the total capacity).
        :param position: The position of the node in the network (e.g., [x, y]).
        :param has_solar: Whether the node has a solar panel for energy harvesting.
        :param kwargs: Additional parameters for node configuration.
        """
        self.node_id = node_id
        self.position = position  # [x, y] position of the node
        self.has_solar = has_solar

        # Energy management parameters
        self.initial_energy = initial_energy
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.capacity = kwargs.get('capacity', 5200)  # Battery capacity in mAh
        self.V = kwargs.get('V', 3.7)  # Voltage (V)
        self.energy_history = []  # To track energy consumption and generation over time
        self.current_energy = initial_energy

        # Solar panel parameters (if the node has a solar panel)
        self.solar_efficiency = kwargs.get('solar_efficiency', 0.2)
        self.solar_area = kwargs.get('solar_area', 0.1)  # Area of a solar panel in m^2
        self.G_max = kwargs.get('G_max', 1500)  # Max solar irradiance in W/m^2
        self.env_correction_factor = kwargs.get('env_correction_factor', 1.0)  # Environmental factor for solar collection

        # Wireless Energy Transfer (WET) parameters
        self.E_char = kwargs.get('E_char', 500)  # Energy consumed for charging during WET (J)
        self.E_elec = kwargs.get('E_elec', 1e-4)  # Energy consumed for electronics (per bit) (J)
        self.epsilon_amp = kwargs.get('epsilon_amp', 1e-5)  # Amplification energy for transmission (per bit per distance^2) (J)
        self.B = kwargs.get('B', 1000000)  # Transmission bit rate in bits
        self.d = kwargs.get('d', 1)  # Distance between nodes
        self.tau = kwargs.get('tau', 2)  # Path loss exponent

        # Threshold energy calculation
        self.low_threshold_energy = self.low_threshold * self.capacity * self.V * 3600  # Convert to Joules
        self.high_threshold_energy = self.high_threshold * self.capacity * self.V * 3600  # Convert to Joules

        # To track energy transferred and received (for WET)
        self.received_energy = 0
        self.transferred_energy = 0
        self.received_history = []  # List to track received energy history
        self.transferred_history = []  # List to track transferred energy history

        self.is_mobile = kwargs.get('is_mobile', False)
        self.mobility_pattern = kwargs.get('mobility_pattern', None)  # e.g., "circle", "line", "oscillate"
        self.mobility_params = kwargs.get('mobility_params', {})  # custom parameters

        # 在__init__ 里：
        self.position_history = [tuple(self.position)]

    def record_transfer(self, received=0, transferred=0):
        """
        Record energy transfer activities: received and transferred energy.

        :param received: Energy received by the node (in Joules).
        :param transferred: Energy transferred by the node (in Joules).
        """
        self.received_history.append(received)
        self.transferred_history.append(transferred)
        self.received_energy += received
        self.transferred_energy += transferred

    def distance_to(self, other_node):
        """
        Calculate the Euclidean distance between this node and another node.

        :param other_node: The other node to calculate distance to.
        :return: The Euclidean distance between the two nodes.
        """
        x1, y1 = self.position
        x2, y2 = other_node.position
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


    def solar_irradiance(self, t):
        """
        Calculate the solar irradiance based on time of day.
        This is a simplified calculation using a sinusoidal model for the daylight period.

        :param t: Time in minutes since the start of the day (e.g., 0 to 1440 minutes).
        :return: Solar irradiance in W/m^2 at time `t`.
        """
        t = t % 1440  # Normalize time to cycle daily
        if 360 <= t <= 1080:  # Between sunrise and sunset (6:00 AM to 6:00 PM)
            return self.G_max * np.sin(np.pi * (t - 360) / 720)  # Simplified sinusoidal model
        return 0

    def energy_harvest(self, t):
        """
        Calculate the energy harvested from the solar panel at a given time.

        :param t: Time in minutes since the start of the day (e.g., 0 to 1440 minutes).
        :return: The energy harvested
        and received energy from WET.

        :param t: Time step in minutes.
        :param receive_WET: The energy received through Wireless Energy Transfer (WET) (in Joules).
        :return: Total energy generated (in Joules).
        """
        # 以分钟为步长的简化能量采集模型
        irradiance = self.solar_irradiance(t)
        harvested_energy = irradiance * self.solar_efficiency * self.solar_area / 60.0 * self.env_correction_factor
        return harvested_energy

    def energy_generation(self, t, receive_WET=0):
        """
        Calculate the total energy generated by the node at a given time from solar harvesting
        and received energy from WET.

        :param t: Time step in minutes.
        :param receive_WET: The energy received through Wireless Energy Transfer (WET) (in Joules).
        :return: Total energy generated (in Joules).
        """
        harvested_energy = self.energy_harvest(t)
        total_generated_energy = harvested_energy + receive_WET
        return total_generated_energy

    def energy_consumption(self, target_node=None, transfer_WET=False):
        """
        Calculate the total energy consumed for a single communication (TX + RX),
        optionally including Wireless Energy Transfer (WET) overhead.

        :param target_node: The node to which this node is communicating.
        :param transfer_WET: Whether this node also performs energy transfer (e.g., WET).
        :return: Total energy consumed (in Joules)
        """
        B = self.B
        d = self.d if target_node is None else self.distance_to(target_node)

        # 发射能耗
        E_tx = self.E_elec * B + self.epsilon_amp * B * (d ** self.tau)

        # 接收能耗（假设双向确认通信）
        E_rx = self.E_elec * B

        # 通信总能耗
        E_com = E_tx + E_rx
        E_com = E_com/2
        E_sen = 0.1 #J

        if transfer_WET:
            E_com += self.E_char  # 加上传能附加开销

        return E_com + E_sen  # 返回通信能耗 + 传感器能耗

    def energy_transfer_efficiency(self, target_node):
        """
        Calculate wireless energy transfer efficiency based on distance.

        :param target_node: Receiver node.
        :return: Efficiency (0~1)
        """
        d = self.distance_to(target_node)
        eta_0 = 0.6  # 1米处最大效率
        gamma = 2.0  # 衰减因子

        efficiency = eta_0 / (d ** gamma)
        return min(1.0, max(0.0, efficiency))  # 限定在 [0, 1] 之间

    def update_energy(self, t):
        """
        Update the energy state of the node at time t, only considering solar harvesting and decay.

        :param t: Time step in minutes.
        :return: Tuple of (generated_energy, decay_energy)
        """
        E_gen = self.energy_harvest(t)
        E_decay = self.energy_decay()

        self.current_energy = self.current_energy + E_gen - E_decay
        self.current_energy = max(0, min(self.current_energy, self.capacity * self.V * 3600))

        self.energy_history.append({"time": t, "generated": E_gen, "consumed": E_decay})
        return E_gen, E_decay

    def energy_decay(self):
        """简化的电池自放电模型（占位）"""
        return 0.0

    def update_position(self, t):
        """If the node is mobile, update its position based on its mobility pattern."""
        if not self.is_mobile or not self.mobility_pattern:
            return

        if self.mobility_pattern == "circle":
            radius = self.mobility_params.get('radius', 1.0)
            speed = self.mobility_params.get('speed', 0.01)  # radians per time step
            cx, cy = self.mobility_params.get('center', self.position)  # circle center
            angle = speed * t
            self.position[0] = cx + radius * math.cos(angle)
            self.position[1] = cy + radius * math.sin(angle)

        elif self.mobility_pattern == "line":
            amplitude = self.mobility_params.get('amplitude', 1.0)
            speed = self.mobility_params.get('speed', 0.1)
            direction = self.mobility_params.get('direction', [1, 0])  # [dx, dy]
            self.position[0] += direction[0] * speed
            self.position[1] += direction[1] * speed

        elif self.mobility_pattern == "oscillate":
            amplitude = self.mobility_params.get('amplitude', 1.0)
            freq = self.mobility_params.get('freq', 0.01)
            axis = self.mobility_params.get('axis', 'x')
            delta = amplitude * math.sin(freq * t)
            if axis == 'x':
                self.position[0] += delta
            else:
                self.position[1] += delta

        # 在 update_position(t) 最后：
        self.position_history.append((self.position[0], self.position[1]))
