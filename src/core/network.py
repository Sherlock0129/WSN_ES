import numpy as np
import math

from acdr.adcr_link_layer import ADCRLinkLayerVirtual
from .SensorNode import SensorNode
from .energy_management import balance_energy
from utils.gpu_compute import get_gpu_manager, compute_distance_matrix_gpu

# 尝试导入路由模块
try:
    from routing.opportunistic_routing import opportunistic_routing
except ImportError:
    opportunistic_routing = None


class Network:
    def __init__(self, 
                 num_nodes: int = 25,
                 low_threshold: float = 0.1,
                 high_threshold: float = 0.9,
                 node_initial_energy: float = 40000,
                 max_hops: int = 3,
                 distribution_mode: str = "random",
                 network_area_width: float = 10.0,
                 network_area_height: float = 10.0,
                 min_distance: float = 0.5,
                 random_seed: int = 129,
                 solar_node_ratio: float = 0.6,
                 mobile_node_ratio: float = 0.1,
                 output_dir: str = "data",
                 use_gpu: bool = False,
                 # 能量空洞模式参数
                 energy_hole_enabled: bool = False,
                 energy_hole_ratio: float = 0.4,
                 energy_hole_center_mode: str = "random",
                 energy_hole_cluster_radius: float = 2.0,
                 energy_hole_mobile_ratio: float = 0.1,
                 # 能量采集参数
                 enable_energy_harvesting: bool = True):
        """
        初始化网络
        
        :param num_nodes: 节点数量
        :param low_threshold: 低能量阈值
        :param high_threshold: 高能量阈值
        :param node_initial_energy: 节点初始能量
        :param max_hops: 最大跳数
        :param distribution_mode: 分布模式 (uniform/random/energy_hole)
        :param network_area_width: 网络区域宽度
        :param network_area_height: 网络区域高度
        :param min_distance: 节点间最小距离
        :param random_seed: 随机种子
        :param solar_node_ratio: 太阳能节点比例
        :param mobile_node_ratio: 移动节点比例
        :param output_dir: 输出目录
        :param use_gpu: 是否使用GPU加速
        :param energy_hole_enabled: 是否启用能量空洞模式
        :param energy_hole_ratio: 非太阳能节点比例（形成能量空洞）
        :param energy_hole_center_mode: 空洞中心选择模式
        :param energy_hole_cluster_radius: 能量空洞聚集半径
        :param energy_hole_mobile_ratio: 能量空洞中移动节点比例
        """
        self.num_nodes = num_nodes
        self.nodes = []
        self.use_gpu = use_gpu
        self.gpu_manager = get_gpu_manager(use_gpu)
        
        # 能量空洞模式参数
        self.energy_hole_enabled = energy_hole_enabled
        self.energy_hole_ratio = energy_hole_ratio
        self.energy_hole_center_mode = energy_hole_center_mode
        self.energy_hole_cluster_radius = energy_hole_cluster_radius
        self.energy_hole_mobile_ratio = energy_hole_mobile_ratio
        
        # 能量采集参数
        self.enable_energy_harvesting = enable_energy_harvesting
        
        # 网络参数
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.node_initial_energy = node_initial_energy
        self.max_hops = max_hops
        self.distribution_mode = distribution_mode
        self.network_area_width = network_area_width
        self.network_area_height = network_area_height
        self.min_distance = min_distance
        self.random_seed = random_seed
        self.solar_node_ratio = solar_node_ratio
        self.mobile_node_ratio = mobile_node_ratio
        self.output_dir = output_dir

        self.create_nodes()
        
        # 预计算距离矩阵（GPU加速）
        self._distance_matrix = None
        self._distance_matrix_valid = False
    
    def _update_distance_matrix(self):
        """更新距离矩阵（GPU加速）"""
        if self.use_gpu:
            self._distance_matrix = compute_distance_matrix_gpu(self.nodes, self.gpu_manager)
        else:
            # CPU版本：计算所有节点间距离
            n = len(self.nodes)
            self._distance_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        self._distance_matrix[i, j] = self.nodes[i].distance_to(self.nodes[j])
        self._distance_matrix_valid = True
    
    def get_distance(self, node1, node2):
        """获取两个节点间的距离（支持GPU加速）"""
        if not self._distance_matrix_valid:
            self._update_distance_matrix()
        
        idx1 = self.nodes.index(node1)
        idx2 = self.nodes.index(node2)
        return self._distance_matrix[idx1, idx2]



    def create_nodes(self):
        """
        根据配置的分布模式创建节点
        """
        if self.distribution_mode == "uniform":
            self._create_uniform_nodes()
        elif self.distribution_mode == "random":
            self._create_random_nodes()
        elif self.distribution_mode == "energy_hole":
            self._create_energy_hole_nodes()
        else:
            # 默认使用均匀分布
            print(f"未知的分布模式: {self.distribution_mode}, 使用默认的均匀分布")
            self._create_uniform_nodes()

    def _create_uniform_nodes(self):
        """
        Create nodes in an approximate square layout using equilateral triangle grid pattern.
        Randomly assign 40% of nodes to be without solar energy harvesting capability.
        """
        import random
        # 固定随机种子
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        horizontal_spacing = 1.0
        vertical_spacing = np.sqrt(3) / 2 * horizontal_spacing

        cols = int(np.ceil(np.sqrt(self.num_nodes)))
        rows = int(np.ceil(self.num_nodes / cols))

        # 生成所有 node_id
        all_node_ids = list(range(self.num_nodes))
        num_without_solar = int((1 - self.solar_node_ratio) * self.num_nodes)
        no_solar_nodes = set(random.sample(all_node_ids, num_without_solar))

        # 从 no_solar_nodes 中挑选移动节点
        num_mobile = int(self.mobile_node_ratio * self.num_nodes)
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

                # 是否具有太阳能能力
                has_solar = node_id not in no_solar_nodes

                is_mobile = node_id in mobile_nodes

                mobility_pattern = "circle" if is_mobile else None
                mobility_params = {
                    "radius": 1.0,
                    "speed": 0.01,
                    "center": [x, y]  # 以初始位置为圆心
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
                    enable_energy_harvesting=self.enable_energy_harvesting
                )

                self.nodes.append(node)
                node_id += 1

    def _create_random_nodes(self):
        """
        创建完全随机分布的节点
        """
        import random
        import math
        
        # 设置随机种子
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # 获取网络区域参数
        width = self.network_area_width
        height = self.network_area_height
        min_dist = self.min_distance
        
        # 生成随机位置
        positions = []
        attempts = 0
        max_attempts = self.num_nodes * 100  # 防止无限循环
        
        print(f"正在生成 {self.num_nodes} 个随机分布的节点...")
        
        while len(positions) < self.num_nodes and attempts < max_attempts:
            x = random.uniform(0, width)
            y = random.uniform(0, height)
            
            # 检查与已有节点的最小距离
            valid_position = True
            for existing_pos in positions:
                if math.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2) < min_dist:
                    valid_position = False
                    break
            
            if valid_position:
                positions.append([x, y])
            
            attempts += 1
        
        # 如果无法生成足够的节点，降低最小距离要求
        if len(positions) < self.num_nodes:
            print(f"警告：无法在最小距离 {min_dist} 下生成所有节点，降低距离要求到 {min_dist/2}")
            min_dist = min_dist / 2
            
            # 重新生成剩余节点
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
        
        # 如果仍然无法生成足够的节点，使用完全随机分布（无距离限制）
        if len(positions) < self.num_nodes:
            print(f"警告：使用完全随机分布（无距离限制）生成剩余节点")
            while len(positions) < self.num_nodes:
                x = random.uniform(0, width)
                y = random.uniform(0, height)
                positions.append([x, y])
        
        print(f"成功生成 {len(positions)} 个节点位置")
        
        # 分配太阳能和移动属性
        self._assign_node_properties(positions)
    
    def _create_energy_hole_nodes(self):
        """
        创建能量空洞模式的节点分布
        将非太阳能节点聚集在一个中心点附近，形成能量空洞
        """
        import random
        import math
        
        # 设置随机种子
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # 获取网络区域参数
        width = self.network_area_width
        height = self.network_area_height
        min_dist = self.min_distance
        
        print(f"正在生成 {self.num_nodes} 个能量空洞模式节点...")
        
        # 1) 先生成所有节点坐标（使用三角形网格模式）
        horizontal_spacing = 1.0
        vertical_spacing = np.sqrt(3) / 2 * horizontal_spacing
        
        cols = int(np.ceil(np.sqrt(self.num_nodes)))
        rows = int(np.ceil(self.num_nodes / cols))
        
        positions = []
        node_id = 0
        for row in range(rows):
            for col in range(cols):
                if node_id >= self.num_nodes:
                    break
                x = col * horizontal_spacing
                if row % 2 == 1:
                    x += horizontal_spacing / 2.0
                y = row * vertical_spacing
                positions.append([x, y])
                node_id += 1
        
        positions = np.array(positions)
        
        # 2) 计算非太阳能节点数
        all_node_ids = list(range(self.num_nodes))
        num_without_solar = int(self.energy_hole_ratio * self.num_nodes)
        
        # 3) 选择能量空洞中心
        if self.energy_hole_center_mode == "random":
            center_idx = random.choice(all_node_ids)
            center = positions[center_idx]
        elif self.energy_hole_center_mode == "corner":
            center = np.array([0.0, 0.0])  # 左下角
        elif self.energy_hole_center_mode == "center":
            center = np.mean(positions, axis=0)  # 几何中心
        else:
            center = positions[int(len(positions)/2)]  # 近似中心
        
        # 4) 按到中心的距离排序，取前 num_without_solar 个作为"无太阳能节点"
        diffs = positions - center
        d2 = np.sum(diffs * diffs, axis=1)
        order = np.argsort(d2)
        selected = order[:num_without_solar]
        no_solar_nodes = set(int(i) for i in selected)
        
        # 5) 从"无太阳能集合"里抽取移动节点
        num_mobile = int(self.energy_hole_mobile_ratio * self.num_nodes)
        mobile_pool = list(no_solar_nodes)
        mobile_nodes = set(random.sample(mobile_pool, min(num_mobile, len(mobile_pool))))
        
        # 6) 创建节点
        for node_id in range(self.num_nodes):
            x, y = positions[node_id]
            has_solar = node_id not in no_solar_nodes
            is_mobile = node_id in mobile_nodes
            
            mobility_pattern = "circle" if is_mobile else None
            mobility_params = {
                "radius": 1.0,
                "speed": 0.01,
                "center": [float(x), float(y)]
            } if is_mobile else {}
            
            node = SensorNode(
                node_id=node_id,
                initial_energy=self.node_initial_energy,
                low_threshold=self.low_threshold,
                high_threshold=self.high_threshold,
                position=[float(x), float(y)],
                has_solar=has_solar,
                is_mobile=is_mobile,
                mobility_pattern=mobility_pattern,
                mobility_params=mobility_params,
                enable_energy_harvesting=self.enable_energy_harvesting
            )
            
            self.nodes.append(node)
        
        print(f"能量空洞模式生成完成：{num_without_solar} 个非太阳能节点聚集在中心 {center} 附近")

    def _assign_node_properties(self, positions):
        """
        为节点分配太阳能和移动属性
        """
        import random
        
        # 生成所有 node_id
        all_node_ids = list(range(self.num_nodes))
        num_without_solar = int((1 - self.solar_node_ratio) * self.num_nodes)
        no_solar_nodes = set(random.sample(all_node_ids, num_without_solar))
        
        # 从 no_solar_nodes 中挑选移动节点
        num_mobile = int(self.mobile_node_ratio * self.num_nodes)
        mobile_nodes = set(random.sample(list(no_solar_nodes), num_mobile))
        
        # 创建节点
        for node_id in range(self.num_nodes):
            x, y = positions[node_id]
            
            # 是否具有太阳能能力
            has_solar = node_id not in no_solar_nodes
            is_mobile = node_id in mobile_nodes
            
            mobility_pattern = "circle" if is_mobile else None
            mobility_params = {
                "radius": 1.0,
                "speed": 0.01,
                "center": [x, y]  # 以初始位置为圆心
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
                enable_energy_harvesting=self.enable_energy_harvesting
            )
            
            self.nodes.append(node)

    # def create_nodes(self):
    #     """
    #     Create nodes in an approximate square layout using equilateral triangle grid pattern.
    #     Force nodes without solar to form a compact cluster near a chosen center.
    #     """
    #     import random
    #     horizontal_spacing = 1.0
    #     vertical_spacing = np.sqrt(3) / 2 * horizontal_spacing
    
    #     cols = int(np.ceil(np.sqrt(self.num_nodes)))
    #     rows = int(np.ceil(self.num_nodes / cols))
    
    #     # 1) 先生成所有节点坐标（索引= node_id）
    #     positions = []
    #     node_id = 0
    #     for row in range(rows):
    #         for col in range(cols):
    #             if node_id >= self.num_nodes:
    #                 break
    #             x = col * horizontal_spacing
    #             if row % 2 == 1:
    #                 x += horizontal_spacing / 2.0
    #             y = row * vertical_spacing
    #             positions.append([x, y])
    #             node_id += 1
    #     positions = np.array(positions)  # (N, 2)
    
    #     # 2) 计算无太阳能节点数
    #     all_node_ids = list(range(self.num_nodes))
    #     num_without_solar = int(0.4 * self.num_nodes)
    
    #     # 3) 选择一个“团簇中心”
    #     # 方案A：随机选一个现有节点做中心（默认）
    #     center_idx = random.choice(all_node_ids)
    #     center = positions[center_idx]
    
    #     # 可选 方案B：固定在版图左下角/中心等
    #     # center = np.array([0.0, 0.0])                          # 左下角
    #     # center = positions[int(len(positions)/2)]              # 近似中心
    #     # center = np.mean(positions, axis=0)                    # 幾何中心
    
    #     # 4) 按到中心的距离排序，取前 40% 个作为“无太阳能节点”
    #     diffs = positions - center
    #     d2 = np.sum(diffs * diffs, axis=1)  # 距离平方即可
    #     order = np.argsort(d2)  # 从近到远
    #     selected = order[:num_without_solar]
    #     no_solar_nodes = set(int(i) for i in selected)
    
    #     # 5) 从“无太阳能集合”里抽 10% 作为“移动节点”
    #     num_mobile = int(0.1 * self.num_nodes)
    #     mobile_pool = list(no_solar_nodes)
    #     mobile_nodes = set(random.sample(mobile_pool, min(num_mobile, len(mobile_pool))))
    
    #     # 6) 实例化节点
    #     self.nodes = []
    #     for node_id in range(self.num_nodes):
    #         x, y = positions[node_id]
    #         has_solar = node_id not in no_solar_nodes
    #         is_mobile = node_id in mobile_nodes
    
    #         mobility_pattern = "circle" if is_mobile else None
    #         mobility_params = {
    #             "radius": 1.0,
    #             "speed": 0.01,
    #             "center": [float(x), float(y)]
    #         } if is_mobile else {}
    
    #         node = SensorNode(
    #             node_id=node_id,
    #             initial_energy=self.node_initial_energy,
    #             low_threshold=self.low_threshold,
    #             high_threshold=self.high_threshold,
    #             position=[float(x), float(y)],
    #             has_solar=has_solar,
    #             is_mobile=is_mobile,
    #             mobility_pattern=mobility_pattern,
    #             mobility_params=mobility_params,
    #             **{k: v for k, v in self.network_config.items() if k not in ['low_threshold', 'high_threshold']}
    #         )
    #         self.nodes.append(node)

    # def run_routing1(self, t):
    #     """
    #     Each below-average node (prioritized from lowest energy) requests energy from a unique donor:
    #     - Donor must have energy > average
    #     - Among available donors, choose the one with highest energy and shortest distance
    #     - Donor can only be assigned to one receiver
    #     """
    #     routing_plans = []
    #     used_donors = set()  # 记录已经分配的 donor 节点
    #
    #     # 计算平均能量
    #     average_energy = np.mean([node.current_energy for node in self.nodes])
    #
    #     # 将低于平均能量的节点按 current_energy 升序排列
    #     low_energy_nodes = sorted(
    #         [node for node in self.nodes if node.current_energy < average_energy],
    #         key=lambda n: n.current_energy
    #     )
    #
    #     for receiver in low_energy_nodes:
    #         # 筛选未被占用、能量 > 平均 的节点
    #         candidate_nodes = [
    #             node for node in self.nodes
    #             if node != receiver and node.current_energy > average_energy and node not in used_donors
    #         ]
    #
    #         if not candidate_nodes:
    #             continue  # 无可用供能节点
    #
    #         # 优先选 energy 最大的，再在其中找距离最近的
    #         max_energy = max(n.current_energy for n in candidate_nodes)
    #         max_energy_nodes = [n for n in candidate_nodes if n.current_energy == max_energy]
    #
    #         # 距离最近的最大能量节点
    #         closest_donor = min(max_energy_nodes, key=lambda node: receiver.distance_to(node))
    #         distance = receiver.distance_to(closest_donor)
    #
    #         # 标记该 donor 已被分配
    #         used_donors.add(closest_donor)
    #
    #         # 规划路径
    #         if distance <= math.sqrt(3):
    #             path = [closest_donor, receiver]
    #         else:
    #             path = opportunistic_routing(self.nodes, closest_donor, receiver, max_hops=self.max_hops, t=t)
    #             if path is None:
    #                 print(f"[Routing Failed] {closest_donor.node_id} → {receiver.node_id}")
    #                 continue
    #
    #         routing_plans.append({
    #             "receiver": receiver,
    #             "donor": closest_donor,
    #             "path": path,
    #             "distance": distance
    #         })
    #
    #     return routing_plans
    #
    # def run_routing2(self, t):
    #     """
    #     Each below-average node (prioritized by largest energy deficit) requests energy from up to three unique donors:
    #     - Donors must have energy > average
    #     - Among available donors, choose those with highest energy and shortest distance
    #     - Each donor can only be assigned to one receiver
    #     - Each receiver can have up to three donors
    #
    #     Args:
    #         t: Current time step
    #     Returns:
    #         List of routing plans, each containing receiver, donor, path, and distance
    #     """
    #     routing_plans = []
    #     used_donors = set()  # Track donors assigned to any receiver
    #     receiver_donor_count = {node: 0 for node in self.nodes}  # Track number of donors per receiver
    #
    #     # Calculate average energy
    #     average_energy = np.mean([node.current_energy for node in self.nodes])
    #
    #     # Sort nodes by largest energy deficit (average_energy - current_energy)
    #     low_energy_nodes = sorted(
    #         [node for node in self.nodes if node.current_energy < average_energy],
    #         key=lambda n: average_energy - n.current_energy,
    #         reverse=True
    #     )
    #
    #     for receiver in low_energy_nodes:
    #         # Skip if receiver already has 3 donors
    #         if receiver_donor_count[receiver] >= 3:
    #             continue
    #
    #         # Select up to 3 donors for this receiver
    #         candidate_nodes = [
    #             node for node in self.nodes
    #             if node != receiver and node.current_energy > average_energy and node not in used_donors
    #         ]
    #
    #         # If no candidates, move to next receiver
    #         if not candidate_nodes:
    #             continue
    #
    #         # Sort candidates by energy (descending) and distance (ascending)
    #         sorted_candidates = sorted(
    #             candidate_nodes,
    #             key=lambda n: (-n.current_energy, receiver.distance_to(n))
    #         )
    #
    #         # Select up to 3 donors or as many as available
    #         selected_donors = sorted_candidates[:3 - receiver_donor_count[receiver]]
    #
    #         for donor in selected_donors:
    #             distance = receiver.distance_to(donor)
    #
    #             # Mark donor as used
    #             used_donors.add(donor)
    #             receiver_donor_count[receiver] += 1
    #
    #             # Plan path
    #             if distance <= math.sqrt(3):
    #                 path = [donor, receiver]
    #             else:
    #                 path = opportunistic_routing(self.nodes, donor, receiver, max_hops=self.max_hops, t=t)
    #                 if path is None:
    #                     print(f"[Routing Failed] {donor.node_id} → {receiver.node_id}")
    #                     continue
    #
    #             routing_plans.append({
    #                 "receiver": receiver,
    #                 "donor": donor,
    #                 "path": path,
    #                 "distance": distance
    #             })
    #
    #     return routing_plans

    def run_routing(self, t, max_donors_per_receiver=3):
        """
        Each below-average node (prioritized by largest energy deficit) requests energy from up to N unique donors:
        - Donors must have energy > average
        - Among available donors, choose those with highest energy and shortest distance
        - Each donor can only be assigned to one receiver
        - Each receiver can have up to `max_donors_per_receiver` donors
        """
        routing_plans = []
        used_donors = set()  # Track donors assigned to any receiver
        receiver_donor_count = {node: 0 for node in self.nodes}  # Track number of donors per receiver

        # Calculate average energy
        average_energy = np.mean([node.current_energy for node in self.nodes])

        # Sort nodes by largest energy deficit (average_energy - current_energy)
        low_energy_nodes = sorted(
            [node for node in self.nodes if node.current_energy < average_energy],
            key=lambda n: average_energy - n.current_energy,
            reverse=True
        )

        for receiver in low_energy_nodes:
            # 若该接收端已达上限，跳过
            if receiver_donor_count[receiver] >= max_donors_per_receiver:
                continue

            # 候选 donor：未被占用、能量 > 平均
            candidate_nodes = [
                node for node in self.nodes
                if node is not receiver and node.current_energy > average_energy and node not in used_donors
            ]
            if not candidate_nodes:
                continue

            # 按能量降序 + 距离升序排序
            if self.use_gpu:
                # 使用预计算的距离矩阵
                sorted_candidates = sorted(
                    candidate_nodes,
                    key=lambda n: (-n.current_energy, self.get_distance(receiver, n))
                )
            else:
                # 使用原始方法
                sorted_candidates = sorted(
                    candidate_nodes,
                    key=lambda n: (-n.current_energy, receiver.distance_to(n))
                )

            # 逐个尝试候选 donor，直到达到上限或没有可行路径
            quota = max_donors_per_receiver - receiver_donor_count[receiver]
            for donor in sorted_candidates:
                if quota <= 0:
                    break

                if self.use_gpu:
                    distance = self.get_distance(receiver, donor)
                else:
                    distance = receiver.distance_to(donor)

                # 规划路径 - 统一使用路由算法
                path = opportunistic_routing(self.nodes, donor, receiver, max_hops=self.max_hops, t=t)
                # path = eeor_find_path(self.nodes, donor, receiver, max_hops=self.max_hops)
                if path is None:
                    print(f"[Routing Failed] {donor.node_id} → {receiver.node_id}")
                    continue

                # ★ 只有在 path 成功后，才占用 donor 并计数
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
        """
        Execute energy transfers based on the routing plans returned by run_routing().
        Updates donor and receiver energies and histories.
        """
        for plan in plans:
            donor = plan["donor"]
            receiver = plan["receiver"]
            path = plan["path"]
            distance = plan["distance"]

            # 传输能量上限（可调策略）
            energy_sent = donor.E_char

            if distance <= math.sqrt(3):
                # 近距离直接传输：效率计算
                eta = donor.energy_transfer_efficiency(receiver)
                energy_received = energy_sent * eta
                energy_loss = energy_sent - energy_received

                donor.current_energy -= donor.energy_consumption(receiver, transfer_WET=True)
                receiver.current_energy += energy_received

                donor.transferred_history.append(energy_sent)
                receiver.received_history.append(energy_received)

                print(f"[Direct WET] {donor.node_id} → {receiver.node_id}, η={eta:.2f}, +{energy_received:.2f}J, loss={energy_loss:.2f}J")
            
            else:
                # 多跳传输：逐跳转发，每跳能量衰减
                energy_left = energy_sent
                donor.transferred_history.append(energy_sent)

                for i in range(len(path) - 1):
                    sender = path[i]
                    receiver_i = path[i + 1]

                    # 计算本跳的效率
                    eta = sender.energy_transfer_efficiency(receiver_i)

                    # 计算本跳实际接收能量
                    energy_delivered = energy_left * eta
                    energy_loss_this_hop = energy_left - energy_delivered

                    # 发射方消耗能量（包括WET只在第一跳加）
                    transfer_WET = (i == 0)  # 仅donor计算WET模块消耗
                    sender.current_energy -= sender.energy_consumption(receiver_i, transfer_WET=transfer_WET)

                    # 接收方获得能量
                    receiver_i.current_energy += energy_delivered
                    receiver_i.received_history.append(energy_delivered)

                    # 打印日志
                    print(
                        f"  [Hop {i + 1}] {sender.node_id} → {receiver_i.node_id}, d={sender.distance_to(receiver_i):.2f}m, η={eta:.3f}, +{energy_delivered:.2f}J, loss={energy_loss_this_hop:.2f}J")

                    # 准备下一跳
                    energy_left = energy_delivered

    def update_network_energy(self, t):
        """Update the energy of all nodes considering only harvesting and decay."""
        for node in self.nodes:
            node.update_position(t)  # 新增：移动节点更新位置
            E_gen, E_decay = node.update_energy(t)
            self.display_energy_update(node, E_gen, E_decay)

    def balance_network_energy(self):
        """Balance the energy in the network to ensure no node exceeds its energy thresholds."""
        balance_energy(self.nodes)

    def display_routing(self, path):
        """Display the optimal path found by the routing algorithm."""
        print("Optimal routing path:")
        for node in path:
            print(f"Node {node.node_id} -> ", end="")
        print("End of path")

    def display_energy_update(self, node, E_gen, E_con):
        # print(f"Node {node.node_id} energy updated:")  # 注释掉详细节点能量输出，影响可读性
        # print(f"  Energy generated (solar): {E_gen:.2f} J")
        # print(f"  Energy decayed (battery loss): {E_con:.2f} J")
        # print(f"  Current energy: {node.current_energy:.2f} J")
        pass

    def simulate_network(self, time_steps):
        """Simulate the network for a given number of time steps."""
        for t in range(time_steps):
            # print(f"\n--- Time step {t + 1} ---")  # 注释掉时间步输出，影响可读性
            self.run_routing(t)  # Pass t as a parameter
            self.update_network_energy(t)  # Pass t as a parameter
            self.balance_network_energy()
