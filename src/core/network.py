import numpy as np
import math

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
                 enable_energy_hole: bool = False,
                 energy_hole_center_mode: str = "random",
                 energy_hole_mobile_ratio: float = 0.1,
                 # 能量采集参数
                 enable_energy_harvesting: bool = True,
                 # 能量分配模式参数
                 energy_distribution_mode: str = "uniform",
                 center_energy: float = 60000.0,
                 edge_energy: float = 20000.0,
                 # NodeConfig参数（传递给SensorNode）
                 capacity: float = 3.5,
                 voltage: float = 3.7,
                 solar_efficiency: float = 0.2,
                 solar_area: float = 0.1,
                 max_solar_irradiance: float = 1500.0,
                 env_correction_factor: float = 1.0,
                 energy_char: float = 1000.0,
                 energy_elec: float = 1e-4,
                 epsilon_amp: float = 1e-5,
                 bit_rate: float = 1000000.0,
                 path_loss_exponent: float = 2.0,
                 energy_decay_rate: float = 5.0,
                 sensor_energy: float = 0.1,
                 # 物理中心节点参数
                 enable_physical_center: bool = True,
                 center_initial_energy_multiplier: float = 10.0):
        """
        初始化网络
        
        :param num_nodes: 普通节点数量（不包括物理中心）
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
        :param enable_energy_hole: 是否启用能量空洞模式（非太阳能节点聚集分布）
        :param energy_hole_center_mode: 空洞中心选择模式
        :param energy_hole_mobile_ratio: 能量空洞区域中移动节点比例
        :param enable_physical_center: 是否启用物理中心节点
        :param center_initial_energy_multiplier: 物理中心初始能量倍数
        """
        self.num_nodes = num_nodes
        self.nodes = []
        self.use_gpu = use_gpu
        self.gpu_manager = get_gpu_manager(use_gpu)
        
        # 物理中心节点参数
        self.enable_physical_center = enable_physical_center
        self.center_initial_energy_multiplier = center_initial_energy_multiplier
        self.physical_center = None  # 物理中心节点引用
        
        # 能量空洞模式参数
        self.enable_energy_hole = enable_energy_hole
        self.energy_hole_center_mode = energy_hole_center_mode
        self.energy_hole_mobile_ratio = energy_hole_mobile_ratio
        
        # 能量采集参数
        self.enable_energy_harvesting = enable_energy_harvesting
        
        # 能量分配模式参数
        self.energy_distribution_mode = energy_distribution_mode
        self.center_energy = center_energy
        self.edge_energy = edge_energy
        
        # NodeConfig参数（传递给SensorNode）
        self.node_capacity = capacity
        self.node_voltage = voltage
        self.node_solar_efficiency = solar_efficiency
        self.node_solar_area = solar_area
        self.node_max_solar_irradiance = max_solar_irradiance
        self.node_env_correction_factor = env_correction_factor
        self.node_energy_char = energy_char
        self.node_energy_elec = energy_elec
        self.node_epsilon_amp = epsilon_amp
        self.node_bit_rate = bit_rate
        self.node_path_loss_exponent = path_loss_exponent
        self.node_energy_decay_rate = energy_decay_rate
        self.node_sensor_energy = sensor_energy
        
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
        
        # 根据能量分布模式分配能量
        self._assign_energy_by_virtual_center()
        
        # 预计算距离矩阵（GPU加速）
        self._distance_matrix = None
        self._distance_matrix_valid = False
        
        # 路径信息收集器（稍后由外部设置）
        self.path_info_collector = None
    
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
    
    def get_regular_nodes(self) -> list:
        """获取所有普通节点（排除物理中心）"""
        return [n for n in self.nodes if not n.is_physical_center]
    
    def get_physical_center(self) -> SensorNode:
        """获取物理中心节点（位置固定不变）"""
        return self.physical_center



    def create_nodes(self):
        """
        根据配置的分布模式创建节点
        采用职责分离的三步法：
        1. 生成节点位置（根据 distribution_mode）
        2. 分配节点属性（根据 enable_energy_hole）
        3. 创建节点对象
        """
        # Step 1: 生成位置
        if self.distribution_mode == "uniform":
            positions = self._generate_uniform_positions()
        elif self.distribution_mode == "random":
            positions = self._generate_random_positions()
        else:
            print(f"未知的分布模式: {self.distribution_mode}, 使用默认的随机分布")
            positions = self._generate_random_positions()
        
        # Step 2: 分配属性（太阳能、移动性）
        no_solar_nodes, mobile_nodes = self._assign_solar_properties(positions)
        
        # Step 3: 创建节点
        self._create_nodes_from_positions_and_properties(positions, no_solar_nodes, mobile_nodes)
        
        # Step 4: 根据能量分配模式调整初始能量
        self._assign_energy_by_virtual_center()
    
    def _generate_uniform_positions(self) -> np.ndarray:
        """
        生成均匀网格分布的节点位置（三角形网格模式）
        
        Returns:
            np.ndarray: (N, 2) 数组，每行为 [x, y] 坐标
        """
        import random
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
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
                    x += horizontal_spacing / 2
                y = row * vertical_spacing
                
                positions.append([x, y])
                node_id += 1
        
        print(f"[OK] 生成 {len(positions)} 个均匀网格位置")
        return np.array(positions)
    
    def _generate_random_positions(self) -> np.ndarray:
        """
        生成随机分布的节点位置（带最小距离约束）
        
        Returns:
            np.ndarray: (N, 2) 数组，每行为 [x, y] 坐标
        """
        import random
        import math
        
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        width = self.network_area_width
        height = self.network_area_height
        min_dist = self.min_distance
        
        positions = []
        attempts = 0
        max_attempts = self.num_nodes * 100
        
        while len(positions) < self.num_nodes and attempts < max_attempts:
            x = random.uniform(0, width)
            y = random.uniform(0, height)
            
            # 检查最小距离约束
            valid_position = True
            for existing_pos in positions:
                if math.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2) < min_dist:
                    valid_position = False
                    break
            
            if valid_position:
                positions.append([x, y])
            
            attempts += 1
        
        # 如果无法满足最小距离，降低要求
        if len(positions) < self.num_nodes:
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
        
        # 最后兜底：完全随机
        if len(positions) < self.num_nodes:
            while len(positions) < self.num_nodes:
                x = random.uniform(0, width)
                y = random.uniform(0, height)
                positions.append([x, y])
        
        print(f"[OK] 生成 {len(positions)} 个随机位置")
        return np.array(positions)
    
    def _assign_solar_properties(self, positions: np.ndarray) -> tuple:
        """
        根据 enable_energy_hole 决定太阳能属性分配方式
        
        Args:
            positions: (N, 2) 节点位置数组
            
        Returns:
            tuple: (no_solar_nodes, mobile_nodes) 两个集合
        """
        if self.enable_energy_hole:
            # 能量空洞模式：非太阳能节点聚集在某个中心附近
            return self._assign_solar_clustered(positions)
        else:
            # 正常模式：随机均匀分配
            return self._assign_solar_random(positions)
    
    def _assign_solar_random(self, positions: np.ndarray) -> tuple:
        """
        随机分配太阳能属性（默认方式）
        
        Args:
            positions: (N, 2) 节点位置数组
            
        Returns:
            tuple: (no_solar_nodes, mobile_nodes) 两个集合
        """
        import random
        
        # 重置随机种子，确保属性分配独立于位置生成
        random.seed(self.random_seed)
        
        all_node_ids = list(range(self.num_nodes))
        num_without_solar = int((1 - self.solar_node_ratio) * self.num_nodes)
        no_solar_nodes = set(random.sample(all_node_ids, num_without_solar))
        
        # 从非太阳能节点中选择移动节点
        num_mobile = int(self.mobile_node_ratio * self.num_nodes)
        mobile_pool = list(no_solar_nodes) if no_solar_nodes else all_node_ids
        mobile_nodes = set(random.sample(mobile_pool, min(num_mobile, len(mobile_pool))))
        
        print(f"[OK] 随机分配属性：{len(no_solar_nodes)} 个非太阳能节点，{len(mobile_nodes)} 个移动节点")
        return no_solar_nodes, mobile_nodes
    
    def _assign_solar_clustered(self, positions: np.ndarray) -> tuple:
        """
        聚集分配太阳能属性（能量空洞方式）
        非太阳能节点聚集在某个中心点附近，形成能量空洞
        
        Args:
            positions: (N, 2) 节点位置数组
            
        Returns:
            tuple: (no_solar_nodes, mobile_nodes) 两个集合
        """
        import random
        import math
        
        # 重置随机种子，确保属性分配独立于位置生成
        random.seed(self.random_seed)
        
        all_node_ids = list(range(self.num_nodes))
        num_without_solar = int((1 - self.solar_node_ratio) * self.num_nodes)
        
        # 选择能量空洞中心
        if self.energy_hole_center_mode == "random":
            center_idx = random.choice(all_node_ids)
            center = positions[center_idx]
        elif self.energy_hole_center_mode == "corner":
            center = np.array([0.0, 0.0])  # 左下角
        elif self.energy_hole_center_mode == "center":
            center = np.mean(positions, axis=0)  # 几何中心
        else:
            center = positions[int(len(positions)/2)]  # 近似中心
        
        # 按到中心的距离排序，取最近的 N 个作为非太阳能节点
        diffs = positions - center
        distances_squared = np.sum(diffs * diffs, axis=1)
        order = np.argsort(distances_squared)
        selected = order[:num_without_solar]
        no_solar_nodes = set(int(i) for i in selected)
        
        # 从非太阳能节点（能量空洞区域）中选择移动节点
        num_mobile = int(self.energy_hole_mobile_ratio * self.num_nodes)
        mobile_pool = list(no_solar_nodes)
        mobile_nodes = set(random.sample(mobile_pool, min(num_mobile, len(mobile_pool))))
        
        # 计算能量空洞半径
        max_distance = max([math.sqrt(distances_squared[i]) for i in no_solar_nodes])
        
        print(f"\n[OK] 能量空洞模式配置：")
        print(f"  - 能量空洞中心：({center[0]:.2f}, {center[1]:.2f}) [{self.energy_hole_center_mode}]")
        print(f"  - 非太阳能节点：{num_without_solar}/{self.num_nodes} ({(1-self.solar_node_ratio)*100:.1f}%)")
        print(f"  - 移动节点数：{len(mobile_nodes)}/{self.num_nodes} ({self.energy_hole_mobile_ratio*100:.1f}%)")
        print(f"  - 能量空洞半径：{max_distance:.2f}m\n")
        
        return no_solar_nodes, mobile_nodes
    
    def _create_physical_center_node(self, regular_nodes: list) -> SensorNode:
        """
        创建物理中心节点
        
        特性：
        - node_id = 0（固定）
        - 位置 = 普通节点的几何中心
        - 初始能量 = 普通节点初始能量 × 倍数
        - 电池容量 = 普通节点电池容量 × 倍数（能量上限也是倍数）
        - is_physical_center = True
        - has_solar = False（物理中心不需要太阳能）
        - is_mobile = False（物理中心不移动）
        
        :param regular_nodes: 普通节点列表（用于计算几何中心）
        :return: 物理中心节点
        """
        # 1. 计算几何中心
        center_x = sum(n.position[0] for n in regular_nodes) / len(regular_nodes)
        center_y = sum(n.position[1] for n in regular_nodes) / len(regular_nodes)
        
        # 2. 计算物理中心初始能量和电池容量（都是普通节点的倍数）
        center_energy = self.node_initial_energy * self.center_initial_energy_multiplier
        center_capacity = self.node_capacity * self.center_initial_energy_multiplier
        
        # 3. 创建物理中心节点
        physical_center = SensorNode(
            node_id=0,  # 固定ID为0
            position=[center_x, center_y],
            initial_energy=center_energy,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
            has_solar=False,  # 物理中心不需要太阳能
            is_mobile=False,  # 物理中心不移动
            is_physical_center=True,  # 标记为物理中心
            # 物理中心的电池容量是普通节点的倍数（使能量上限也是倍数）
            capacity=center_capacity,
            voltage=self.node_voltage,
            enable_energy_harvesting=False,  # 物理中心不采集能量
            solar_efficiency=self.node_solar_efficiency,
            solar_area=self.node_solar_area,
            max_solar_irradiance=self.node_max_solar_irradiance,
            env_correction_factor=self.node_env_correction_factor,
            energy_char=self.node_energy_char,
            energy_elec=self.node_energy_elec,
            epsilon_amp=self.node_epsilon_amp,
            bit_rate=self.node_bit_rate,
            path_loss_exponent=self.node_path_loss_exponent,
            energy_decay_rate=self.node_energy_decay_rate,
            sensor_energy=self.node_sensor_energy
        )
        
        return physical_center
    
    def _create_nodes_from_positions_and_properties(
        self, positions: np.ndarray, no_solar_nodes: set, mobile_nodes: set):
        """
        根据位置和属性创建所有普通节点对象（ID从1开始）
        
        Args:
            positions: (N, 2) 节点位置数组
            no_solar_nodes: 非太阳能节点的ID集合（基于0索引）
            mobile_nodes: 移动节点的ID集合（基于0索引）
        """
        regular_nodes = []
        for idx in range(self.num_nodes):
            x, y = positions[idx]
            
            # 普通节点ID从1开始
            node_id = idx + 1
            
            # 确定节点属性（使用idx作为索引）
            has_solar = idx not in no_solar_nodes
            is_mobile = idx in mobile_nodes
            
            mobility_pattern = "circle" if is_mobile else None
            mobility_params = {
                "radius": 1.0,
                "speed": 0.01,
                "center": [float(x), float(y)]
            } if is_mobile else {}
            
            # 创建普通节点
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
                is_physical_center=False,  # 普通节点
                # NodeConfig参数
                capacity=self.node_capacity,
                voltage=self.node_voltage,
                enable_energy_harvesting=self.enable_energy_harvesting,
                solar_efficiency=self.node_solar_efficiency,
                solar_area=self.node_solar_area,
                max_solar_irradiance=self.node_max_solar_irradiance,
                env_correction_factor=self.node_env_correction_factor,
                energy_char=self.node_energy_char,
                energy_elec=self.node_energy_elec,
                epsilon_amp=self.node_epsilon_amp,
                bit_rate=self.node_bit_rate,
                path_loss_exponent=self.node_path_loss_exponent,
                energy_decay_rate=self.node_energy_decay_rate,
                sensor_energy=self.node_sensor_energy
            )
            
            regular_nodes.append(node)
        
        print(f"[OK] 创建 {len(regular_nodes)} 个普通节点对象完成")
        
        # 创建物理中心节点（如果启用）
        if self.enable_physical_center:
            self.physical_center = self._create_physical_center_node(regular_nodes)
            # 物理中心节点ID=0，放在列表开头
            self.nodes = [self.physical_center] + regular_nodes
            print(f"[OK] 网络初始化完成: 物理中心(ID=0) + {len(regular_nodes)}个普通节点(ID=1-{len(regular_nodes)})")
        else:
            self.nodes = regular_nodes
            self.physical_center = None
            print(f"[OK] 网络初始化完成: {len(regular_nodes)}个普通节点(ID=1-{len(regular_nodes)})，无物理中心")

    def _assign_energy_by_virtual_center(self):
        """基于虚拟中心（几何中心）分配能量，实现从中心到边缘的线性递减"""
        if self.energy_distribution_mode != "center_decreasing":
            return
        
        # 计算网络几何中心（与ADCR虚拟中心相同）
        xs, ys = zip(*[node.position for node in self.nodes])
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        network_center = (center_x, center_y)
        
        # 计算所有节点到中心的距离
        distances = []
        for node in self.nodes:
            distance = math.sqrt((node.position[0] - center_x)**2 + (node.position[1] - center_y)**2)
            distances.append(distance)
        
        max_distance = max(distances)
        
        print(f"[Energy-Distribution] 网络中心: ({center_x:.2f}, {center_y:.2f})")
        print(f"[Energy-Distribution] 最大距离: {max_distance:.2f}m")
        print(f"[Energy-Distribution] 中心能量: {self.center_energy}J, 边缘能量: {self.edge_energy}J")
        
        # 根据距离分配能量（线性递减）
        for i, node in enumerate(self.nodes):
            # 跳过物理中心节点，保护其特殊能量值
            if node.is_physical_center:
                continue

            distance = distances[i]
            
            # 线性衰减：ratio = 1.0 - (distance / max_distance)
            ratio = 1.0 - (distance / max_distance) if max_distance > 0 else 1.0
            
            # 计算能量
            energy = self.edge_energy + (self.center_energy - self.edge_energy) * ratio
            
            # 更新节点能量
            node.initial_energy = energy
            node.current_energy = energy
            
            print(f"[Energy-Distribution] Node {node.node_id}: 距离={distance:.2f}m, 比例={ratio:.3f}, 能量={energy:.0f}J")

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
                # path = eetor_find_path_adaptive(self.nodes, donor, receiver, max_hops=self.max_hops)
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

    def execute_energy_transfer(self, plans, current_time: int = None):
        """
        Execute energy transfers based on the routing plans returned by run_routing().
        Updates donor and receiver energies and histories.
        
        :param plans: 传能计划列表
        :param current_time: 当前时间步（用于路径信息收集）
        """
        for plan in plans:
            donor = plan["donor"]
            receiver = plan["receiver"]
            path = plan["path"]
            distance = plan["distance"]

            # 传输能量计算（支持传输时长）
            # 如果plan中包含duration（传输时长），则能量 = duration × E_char
            # 否则使用默认的 E_char（1分钟的传输量）
            duration = plan.get("duration", 1)  # 默认1分钟
            energy_sent = duration * donor.E_char

            # 根据路径长度判断单跳或多跳（自适应路径查找已确定最优路径）
            if len(path) == 2:
                # 单跳直接传输：效率计算
                eta = donor.energy_transfer_efficiency(receiver)
                energy_received = energy_sent * eta
                energy_loss = energy_sent - energy_received

                # 能耗计算：WET模块能耗需要乘以duration（因为传输了duration分钟）
                # 通信能耗（E_tx + E_rx）是一次性的，不需要乘以duration
                # 但WET模块能耗（E_char）是持续传输的，需要乘以duration
                base_consumption = donor.energy_consumption(receiver, transfer_WET=False)
                wet_consumption = donor.E_char * duration  # WET模块能耗乘以duration
                total_consumption = base_consumption + wet_consumption
                
                # 检查donor能量是否足够
                if donor.current_energy < total_consumption:
                    print(f"[警告] Donor {donor.node_id} 能量不足，跳过传输")
                    print(f"  需要: {total_consumption:.2f}J, 拥有: {donor.current_energy:.2f}J")
                    print(f"  计划传输: {energy_sent:.2f}J (duration={duration}min)")
                    continue  # 跳过此传输
                
                # 对于duration>1的传输，逐分钟更新能量（用于可视化）
                if duration > 1:
                    # 每分钟传输的能量
                    energy_per_minute = energy_sent / duration
                    received_per_minute = energy_received / duration
                    consumption_per_minute = total_consumption / duration
                    
                    # 逐分钟更新能量并记录到energy_history
                    for minute in range(duration):
                        donor.current_energy -= consumption_per_minute
                        receiver.current_energy += received_per_minute
                        
                        # 记录到energy_history（用于可视化）
                        donor.energy_history.append(donor.current_energy)
                        receiver.energy_history.append(receiver.current_energy)
                    
                    # 记录总传输量到历史
                    donor.transferred_history.append(energy_sent)
                    receiver.received_history.append(energy_received)
                else:
                    # duration=1时，一次性传输（保持原有逻辑）
                    donor.current_energy -= total_consumption
                    receiver.current_energy += energy_received
                    donor.transferred_history.append(energy_sent)
                    receiver.received_history.append(energy_received)

                # 显示完整路径（所有节点）
                path_str = " → ".join([str(node.node_id) for node in path])
                duration_str = f", {duration}分钟" if duration > 1 else ""
                print(f"[Direct WET] 路径: {path_str}, η={eta:.2f}, +{energy_received:.2f}J, loss={energy_loss:.2f}J{duration_str}")
            
            else:
                # 多跳传输：逐跳转发，每跳能量衰减
                energy_left = energy_sent
                
                # 记录donor的transferred_history
                donor.transferred_history.append(energy_sent)

                # 计算总路径效率（各跳效率的乘积）
                total_eta = 1.0
                for i in range(len(path) - 1):
                    total_eta *= path[i].energy_transfer_efficiency(path[i + 1])
                final_energy_received = energy_sent * total_eta
                total_energy_loss = energy_sent - final_energy_received

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
                    if transfer_WET:
                        # 第一跳（donor）：WET模块能耗需要乘以duration（因为传输了duration分钟）
                        base_consumption = sender.energy_consumption(receiver_i, transfer_WET=False)
                        wet_consumption = sender.E_char * duration  # WET模块能耗乘以duration
                        total_consumption = base_consumption + wet_consumption
                        
                        # 检查donor能量是否足够
                        if sender.current_energy < total_consumption:
                            print(f"[警告] 多跳传输中Donor {sender.node_id} 能量不足，终止路径传输")
                            print(f"  需要: {total_consumption:.2f}J, 拥有: {sender.current_energy:.2f}J")
                            break  # 终止整条路径的传输
                        
                        sender.current_energy -= total_consumption
                    else:
                        # 中间跳：只消耗通信能耗（瞬时转发，不需要乘以duration）
                        consumption = sender.energy_consumption(receiver_i, transfer_WET=False)
                        
                        # 检查中继节点能量是否足够
                        if sender.current_energy < consumption:
                            print(f"[警告] 中继节点 {sender.node_id} 能量不足，终止路径传输")
                            print(f"  需要: {consumption:.2f}J, 拥有: {sender.current_energy:.2f}J")
                            break  # 终止整条路径的传输
                        
                        sender.current_energy -= consumption

                    # 接收方获得能量
                    receiver_i.current_energy += energy_delivered
                    receiver_i.received_history.append(energy_delivered)

                    # 准备下一跳
                    energy_left = energy_delivered
                
                # 显示完整路径（所有节点）
                path_str = " → ".join([str(node.node_id) for node in path])
                duration_str = f", {duration}分钟" if duration > 1 else ""
                print(f"[Multi-hop WET] 路径: {path_str}, 总效率η={total_eta:.3f}, 终点接收={final_energy_received:.2f}J, 总损失={total_energy_loss:.2f}J{duration_str}")
            
            # ✨ 新增：路径信息收集（如果启用）
            if self.path_info_collector is not None and current_time is not None:
                try:
                    self.path_info_collector.collect_and_report(
                        path=path,
                        all_nodes=self.nodes,
                        current_time=current_time
                    )
                except Exception as e:
                    print(f"[PathCollector] 警告：信息收集失败 - {e}")

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
