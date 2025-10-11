import math
import numpy as np
from .SensorNode import SensorNode
# 调整相对导入到 routing 子包
from wsn_et.acdr.adcr_link_layer import ADCRLinkLayerVirtual
from ..routing.opportunistic_routing import opportunistic_routing
from .energy_management import balance_energy


class Network:
    def __init__(self, num_nodes, network_config, **kwargs):
        self.num_nodes = num_nodes
        self.nodes = []
        self.network_config = network_config

        self.low_threshold = network_config.get('low_threshold', 0.25)
        self.high_threshold = network_config.get('high_threshold', 0.75)
        self.node_initial_energy = network_config.get('node_initial_energy', 10000)
        self.max_hops = network_config.get('max_hops', 3)
        
        # 分布模式配置
        self.distribution_mode = network_config.get('distribution_mode', 'uniform')

        self.create_nodes()

        # ---- ADCR 信息层（虚拟中心版）----
        try:
            self.adcr_link = ADCRLinkLayerVirtual(
                self,
                round_period=60,
                plan_paths=True,
                consume_energy=True,
                max_hops=self.max_hops,
                output_dir=self.network_config.get("output_dir", "data")
            )
        except Exception as e:
            print("[ADCR-Link-Virtual] init failed:", e)
            self.adcr_link = None

    def create_nodes(self):
        """
        根据配置的分布模式创建节点
        """
        if self.distribution_mode == "uniform":
            self._create_uniform_nodes()
        elif self.distribution_mode == "random":
            self._create_random_nodes()
        else:
            print(f"未知的分布模式 {self.distribution_mode}, 使用默认的均匀分布")
            self._create_uniform_nodes()

    def _create_uniform_nodes(self):
        import random
        seed = self.network_config.get("random_seed", 42)
        random.seed(seed)
        np.random.seed(seed)

        horizontal_spacing = 1.0
        vertical_spacing = np.sqrt(3) / 2 * horizontal_spacing

        cols = int(np.ceil(np.sqrt(self.num_nodes)))
        rows = int(np.ceil(self.num_nodes / cols))

        all_node_ids = list(range(self.num_nodes))
        num_without_solar = int(0.4 * self.num_nodes)
        no_solar_nodes = set(random.sample(all_node_ids, num_without_solar))

        num_mobile = int(0.1 * self.num_nodes)
        mobile_nodes = set(random.sample(list(no_solar_nodes), num_mobile))

        node_id = 0
        for row in range(rows):
            for col in range(cols):
                if node_id >= self.num_nodes:
                    break

                x = col * horizontal_spacing
                if row % 2 == 1:
                    x += horizontal_spacing / 2
                y = row * vertical_spacing

                has_solar = node_id not in no_solar_nodes
                is_mobile = node_id in mobile_nodes

                mobility_pattern = "circle" if is_mobile else None
                mobility_params = {
                    "radius": 1.0,
                    "speed": 0.01,
                    "center": [x, y]
                } if is_mobile else {}

                node = SensorNode(
                    node_id=node_id,
                    initial_energy=self.node_initial_energy,
                    low_threshold=self.low_threshold,
                    high_threshold=self.high_threshold,
                    position=[x, y],
                    has_solar=has_solar,
                    is_mobile=is_mobile,
                    mobility_pattern=mobility_pattern,
                    mobility_params=mobility_params,
                    **{k: v for k, v in self.network_config.items() if k not in ['low_threshold', 'high_threshold']}
                )

                self.nodes.append(node)
                node_id += 1

    def _create_random_nodes(self):
        import random
        import math
        
        seed = self.network_config.get("random_seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        
        area = self.network_config.get("network_area", {"width": 10.0, "height": 10.0})
        width = area["width"]
        height = area["height"]
        min_dist = self.network_config.get("min_distance", 0.5)
        
        positions = []
        attempts = 0
        max_attempts = self.num_nodes * 100
        
        print(f"正在生成 {self.num_nodes} 个随机分布的节点...")
        
        while len(positions) < self.num_nodes and attempts < max_attempts:
            x = random.uniform(0, width)
            y = random.uniform(0, height)
            
            valid_position = True
            for existing_pos in positions:
                if math.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2) < min_dist:
                    valid_position = False
                    break
            
            if valid_position:
                positions.append([x, y])
            
            attempts += 1
        
        if len(positions) < self.num_nodes:
            print(f"警告：无法在最小距离{min_dist} 下生成所有节点，降低距离要求为 {min_dist/2}")
            min_dist = min_dist / 2
            
            while len(positions) < self.num_nodes and attempts < max_attempts * 2:
                x = random.uniform(0, width)
                y = random.uniform(0, height)
                
                valid_position = True
                for existing_pos in positions:
                    if math.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2) < min_dist:
                        valid_position = False
                        break
                
                if valid_position:
                    positions.append([x, y])
                
                attempts += 1
        
        if len(positions) < self.num_nodes:
            print(f"警告：使用完全随机分布（无距离限制）生成剩余节点")
            while len(positions) < self.num_nodes:
                x = random.uniform(0, width)
                y = random.uniform(0, height)
                positions.append([x, y])
        
        print(f"成功生成 {len(positions)} 个节点位置")
        
        self._assign_node_properties(positions)
    
    def _assign_node_properties(self, positions):
        import random
        
        all_node_ids = list(range(self.num_nodes))
        num_without_solar = int(0.4 * self.num_nodes)
        no_solar_nodes = set(random.sample(all_node_ids, num_without_solar))
        
        num_mobile = int(0.1 * self.num_nodes)
        mobile_nodes = set(random.sample(list(no_solar_nodes), num_mobile))
        
        for node_id in range(self.num_nodes):
            x, y = positions[node_id]
            has_solar = node_id not in no_solar_nodes
            is_mobile = node_id in mobile_nodes
            
            mobility_pattern = "circle" if is_mobile else None
            mobility_params = {
                "radius": 1.0,
                "speed": 0.01,
                "center": [x, y]
            } if is_mobile else {}
            
            node = SensorNode(
                node_id=node_id,
                initial_energy=self.node_initial_energy,
                low_threshold=self.low_threshold,
                high_threshold=self.high_threshold,
                position=[x, y],
                has_solar=has_solar,
                is_mobile=is_mobile,
                mobility_pattern=mobility_pattern,
                mobility_params=mobility_params,
                **{k: v for k, v in self.network_config.items() if k not in ['low_threshold', 'high_threshold']}
            )
            
            self.nodes.append(node)

    def run_routing(self, t, max_donors_per_receiver=3):
        """
        Each below-average node (prioritized by largest energy deficit) requests energy from up to N unique donors:
        - Donors must have energy > average
        - Among available donors, choose those with highest energy and shortest distance
        - Each donor can only be assigned to one receiver
        - Each receiver can have up to `max_donors_per_receiver` donors
        """
        routing_plans = []
        used_donors = set()
        receiver_donor_count = {node: 0 for node in self.nodes}

        average_energy = np.mean([node.current_energy for node in self.nodes])

        low_energy_nodes = sorted(
            [node for node in self.nodes if node.current_energy < average_energy],
            key=lambda n: average_energy - n.current_energy,
            reverse=True
        )

        for receiver in low_energy_nodes:
            if receiver_donor_count[receiver] >= max_donors_per_receiver:
                continue

            candidate_nodes = [
                node for node in self.nodes
                if node is not receiver and node.current_energy > average_energy and node not in used_donors
            ]
            if not candidate_nodes:
                continue

            sorted_candidates = sorted(
                candidate_nodes,
                key=lambda n: (-n.current_energy, receiver.distance_to(n))
            )

            quota = max_donors_per_receiver - receiver_donor_count[receiver]
            for donor in sorted_candidates:
                if quota <= 0:
                    break

                distance = receiver.distance_to(donor)

                if distance <= math.sqrt(3):
                    path = [donor, receiver]
                else:
                    path = opportunistic_routing(self.nodes, donor, receiver, max_hops=self.max_hops, t=t)
                    if path is None:
                        print(f"[Routing Failed] {donor.node_id} -> {receiver.node_id}")
                        continue

                used_donors.add(donor)
                receiver_donor_count[receiver] += 1
                quota -= 1

                routing_plans.append({
                    "receiver": receiver,
                    "donor": donor,
                    "path": path,
                    "distance": distance
                })

        return routing_plans

    def execute_energy_transfer(self, plans):
        for plan in plans:
            donor = plan["donor"]
            receiver = plan["receiver"]
            path = plan["path"]
            distance = plan["distance"]

            energy_sent = donor.E_char

            if distance <= math.sqrt(3):
                eta = donor.energy_transfer_efficiency(receiver)
                energy_received = energy_sent * eta

                donor.current_energy -= donor.energy_consumption(receiver, transfer_WET=True)
                receiver.current_energy += energy_received

                donor.transferred_history.append(energy_sent)
                receiver.received_history.append(energy_received)

                print(f"[Direct WET] {donor.node_id} -> {receiver.node_id}, η={eta:.2f}, +{energy_received:.2f}J")
            
            else:
                energy_left = energy_sent
                donor.transferred_history.append(energy_sent)

                for i in range(len(path) - 1):
                    sender = path[i]
                    receiver_i = path[i + 1]

                    eta = sender.energy_transfer_efficiency(receiver_i)
                    energy_delivered = energy_left * eta

                    transfer_WET = (i == 0)
                    sender.current_energy -= sender.energy_consumption(receiver_i, transfer_WET=transfer_WET)

                    receiver_i.current_energy += energy_delivered
                    receiver_i.received_history.append(energy_delivered)

                    print(
                        f"  [Hop {i + 1}] {sender.node_id} -> {receiver_i.node_id}, d={sender.distance_to(receiver_i):.2f}m, η={eta:.3f}, +{energy_delivered:.2f}J")

                    energy_left = energy_delivered

    def update_network_energy(self, t):
        for node in self.nodes:
            node.update_position(t)
            E_gen, E_decay = node.update_energy(t)
            self.display_energy_update(node, E_gen, E_decay)

    def balance_network_energy(self):
        balance_energy(self.nodes)

    def display_routing(self, path):
        print("Optimal routing path:")
        for node in path:
            print(f"Node {node.node_id} -> ", end="")
        print("End of path")

    def display_energy_update(self, node, E_gen, E_con):
        print(f"Node {node.node_id} energy updated:")
        print(f"  Energy generated (solar): {E_gen:.2f} J")
        print(f"  Energy decayed (battery loss): {E_con:.2f} J")
        print(f"  Current energy: {node.current_energy:.2f} J")

    def simulate_network(self, time_steps):
        for t in range(time_steps):
            print(f"\n--- Time step {t + 1} ---")
            self.run_routing(t)
            self.update_network_energy(t)
            self.balance_network_energy()

