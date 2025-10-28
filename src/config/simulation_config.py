"""
无线传感器网络仿真配置管理
--------------------------------

本模块集中定义并管理仿真所需的全部配置参数，采用 dataclass 进行强类型分组：

- NodeConfig：单个节点的物理/能量模型参数（电池、太阳能、链路能耗等）
- NetworkConfig：网络层面的拓扑与规模参数（节点数、区域尺寸、分布方式等）
- SimulationConfig：仿真运行控制参数（步长、日志、K值自适应、GPU 等）
- SchedulerConfig：调度/路由相关的策略与超参数（Lyapunov、Cluster、Prediction、PowerControl）
- ADCRConfig：ADCR 链路层算法相关参数（聚类、路径到虚拟中心、能耗结算、可视化）
- ParallelConfig：并行实验与参数扫描控制（多次运行、权重扫描、输出管理）

使用说明：
- dataclass 字段即为默认值的单一真实来源（Single Source of Truth）。
- 可通过 `ConfigManager.load_from_file()` 从 JSON 覆盖默认值。
- 运行期由 `ConfigManager.create_*()` 工厂方法将配置注入到对应对象的构造函数。

约定与单位：
- 能量单位默认采用焦耳 J；电池容量以 mAh、电压 V，仅用于推导最大可用能量或参考。
- 距离单位为米 m；时间以分钟为主（与 time_steps、round_period 等一致）。
- 记号 K 多用于“同时捐能者个数上限/并行度”，由自适应逻辑按步调整。
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class NodeConfig:
    """节点配置参数（物理与能耗模型）

    这些参数用于构建 `SensorNode` 的能量状态与通信消耗模型：
    - 电池/初始能量用于限定可用能量与起始状态；
    - 太阳能条目用于估算每步采集能量（若启用）；
    - 通信能耗基于经典无线能耗模型：
      E_tx = E_elec × B + ε_amp × B × d^τ，E_rx = E_elec × B，E_com = E_tx + E_rx。
      当前实现中通信总能耗再除以 2（平均化 ACK 往返），并额外加上固定传感能耗 E_sen：
      E_total = (E_com/2) + E_sen；若执行 WET，则 E_total += E_char。
      多跳路径效率为各跳效率乘积：η_path = ∏ η(hop)。
    """

    initial_energy: float = 40000.0  # 初始能量 J，仿真起点每个节点的可用能量
    low_threshold: float = 0.1       # 低能量阈值（0~1，相对容量），用于判定“接收/需要能量”的节点
    high_threshold: float = 0.9      # 高能量阈值（0~1，相对容量），用于判定“捐能/富余能量”的节点

    # 电池模型（用于参考或扩展），如需将 mAh/Volt 映射到焦耳：J ≈ mAh * 3.6 * V
    capacity: float = 3.5         # 电池容量 mAh（参考值）
    voltage: float = 3.7             # 电池电压 V（参考值）

    # 太阳能采集模型（仅当节点具备太阳能能力时参与估算）
    enable_energy_harvesting: bool = True  # 是否启用能量采集，False时能量采集量=0
    solar_efficiency: float = 0.2    # 光伏转换效率（0~1）
    solar_area: float = 0.1          # 光伏面积 m^2
    max_solar_irradiance: float = 1500.0  # 峰值太阳辐照 W/m^2
    env_correction_factor: float = 1.0    # 环境修正系数（天气/遮挡等），乘在辐照上

    # 无线能量传输/发送参数
    energy_char: float = 1000.0      # 单次名义可下发能量 J（捐能上限/步）
    energy_elec: float = 1e-4        # 电子学能耗 J/bit（发送/接收基底损耗）
    epsilon_amp: float = 1e-5        # 功放损耗系数 J/bit/m^path_loss_exponent
    bit_rate: float = 1000000.0      # 数据量 bits（用于估算一次发送/接收消耗）
    path_loss_exponent: float = 2.0  # 路损指数（自由空间≈2，障碍更高）

    # 其他每步能量项
    energy_decay_rate: float = 5.0   # 自然/维护消耗 J/步（待机、维护等损耗）
    sensor_energy: float = 0.1       # 传感工作消耗 J/步（采样等）
    # 注意：当前 `SensorNode.energy_consumption()` 内部使用固定 E_sen = 0.1 J 参与一次通信能耗，
    # 未从配置项读取该值；即本字段与通信能耗计算尚未绑定，仅用于后续扩展或独立统计。


@dataclass
class NetworkConfig:
    """网络配置参数（拓扑与规模）

    控制网络生成与路径约束：
    - 节点数、区域尺寸、最小间距共同决定布局密度与可连通性；
    - 分布模式决定初始位置生成方式；
    - max_hops 限制多跳路径长度（影响 EEOR/机会路由）。
    """
    num_nodes: int = 30
    max_hops: int = 3
    distribution_mode: str = "random"  # 节点位置分布模式："uniform"（网格/规则）、"random"（随机）
    network_area_width: float = 5.0    # 区域宽度 m
    network_area_height: float = 5.0   # 区域高度 m
    min_distance: float = 0.5          # 节点间最小生成距离 m（避免过近重叠）
    random_seed: int = 128            # 随机种子（影响位置与属性抽样）
    solar_node_ratio: float = 0.6      # 具备太阳能节点比例（0~1）
    mobile_node_ratio: float = 0.0     # 可移动节点比例（0~1，若启用移动模型）
    
    # 能量空洞模式配置参数（独立开关，可与任意 distribution_mode 组合）
    # 说明：启用后，非太阳能节点（1-solar_node_ratio）会聚集在某个中心附近，形成能量空洞区域
    enable_energy_hole: bool = False    # 是否启用能量空洞模式
    energy_hole_center_mode: str = "random"  # 空洞中心选择模式："random"（随机节点）、"corner"（左下角）、"center"（几何中心）
    energy_hole_mobile_ratio: float = 0.0    # 能量空洞区域中移动节点比例（0~1）

    # 物理中心节点配置
    enable_physical_center: bool = True  # 是否启用物理中心节点（ID=0）
    center_initial_energy_multiplier: float = 10.0  # 物理中心初始能量倍数（相对普通节点）
    
    # 能量分配模式配置
    energy_distribution_mode: str = "uniform"  # 能量分配模式："uniform"（固定）、"center_decreasing"（中心递减）
    center_energy: float = 40000.0      # 中心节点能量（最高，当energy_distribution_mode="center_decreasing"时使用）
    edge_energy: float = 30000.0        # 边缘节点能量（最低，当energy_distribution_mode="center_decreasing"时使用）


@dataclass
class SimulationConfig:
    """仿真配置参数（运行控制与K值自适应）

    - time_steps / energy_transfer_interval 控制仿真时间与传能频率；
    - K 自适应相关参数用于平衡供需、送达与损耗三者；
    - use_lookahead 可启用“前瞻模拟”对 K 作短期评估（需额外计算量）；
    - use_gpu_acceleration 控制是否使用 GPU（CuPy）加速统计/距离矩阵等并行计算。
    """

    time_steps: int = 10080              # 总时间步数（分钟），默认 7 天
    energy_transfer_interval: int = 60   # 传能/调度触发间隔（分钟）
    # 注意：当前实现中触发条件写死为 `if t % 60 == 0`，未读取该配置值；后续可将其接入。
    output_dir: str = "data"             # 仿真输出根目录，由 OutputManager 管理会话子目录
    log_level: str = "INFO"              # 日志等级：DEBUG/INFO/WARNING/ERROR
    
    # 能量传输控制
    enable_energy_sharing: bool = True     # 是否启用节点间能量传输（WET）

    # 智能被动传能参数
    passive_mode: bool = True          # 是否启用智能被动传能模式（False为定时主动传能）
    check_interval: int = 1                # 智能检查间隔（分钟）
    critical_ratio: float = 0.2             # 低能量节点临界比例（0-1）
    energy_variance_threshold: float = 0.1  # 能量方差阈值，超过则触发传能
    cooldown_period: int = 1               # 传能冷却期（分钟），避免频繁触发
    predictive_window: int = 60             # 预测窗口（分钟），用于预测性触发
    
    # K 值自适应（影响每个接收端可匹配的捐能者数量上限）
    enable_k_adaptation: bool = False     # 是否启用K值自适应，False时使用固定K值
    fixed_k: int = 1                     # 固定K值（当不使用自适应时）
    initial_K: int = 1                   # 初始 K 值（并行捐能数起点）
    K_max: int = 24                      # K 的上限（避免过度并行导致损耗/拥塞）
    hysteresis: float = 0.2              # 滞回阈值（防抖，避免频繁增减 K）
    w_b: float = 0.8                     # 均衡改进权重（更关注能量标准差降低）
    w_d: float = 0.8                     # 有效送达量权重（关注到达接收端的总能量）
    w_l: float = 1.5                     # 损耗惩罚权重（抑制途损/无效能量开销）
    use_lookahead: bool = False          # 是否进行短期前瞻评估以辅助 K 调整
    
    # ADCR链路层
    enable_adcr_link_layer: bool = False # 是否启用ADCR链路层参与仿真（聚类、路径规划、能耗结算）
    
    # 指令下发能量计算
    enable_command_downlink: bool = True  # 是否启用指令下发能量计算（物理中心→节点）
    command_packet_size: int = 1          # 指令包大小（B，与base_data_size相同）
    # 说明：启用后，每轮调度前会计算中心向所有plan涉及节点下发指令的能量消耗
    #      计算双向能量：中心的E_tx + 节点的E_rx，并扣除相应能量
    #      若中心能量不足，则本轮传能取消
    
    # 计算加速
    use_gpu_acceleration: bool = False   # GPU 加速开关（需要安装 CuPy 与 CUDA 驱动）


@dataclass
class SchedulerConfig:
    """调度器配置参数（策略与超参数）

    通过 `scheduler_type` 切换策略，不同策略读取不同字段：
    - LyapunovScheduler：排队长度/能量差驱动的机会捐能匹配；
    - ClusterScheduler：类似 LEACH 的轮换簇首与簇内均衡；
    - PredictionScheduler：基于能量趋势/滑动估计的前瞻调度；
    - PowerControlScheduler：以达成目标效率 `target_eta` 为导向的功率/送能控制。
    """

    scheduler_type: str = "LyapunovScheduler"  # 默认调度器类型

    # LyapunovScheduler 超参数
    lyapunov_v: float = 0.5                  # Lyapunov 控制强度（越大越保守/稳定）
    lyapunov_k: int = 3                      # 基线 K（可与全局 K 自适应结合/对比）

    # ClusterScheduler 超参数
    cluster_round_period: int = 360          # 轮换簇首周期（分钟）
    cluster_p_ch: float = 0.05               # 成为簇头的期望概率（影响簇规模）

    # PredictionScheduler 超参数
    prediction_alpha: float = 0.6            # 指数平滑系数/趋势权重
    prediction_horizon: int = 60             # 预测窗口（分钟）

    # PowerControlScheduler 超参数
    power_target_eta: float = 0.25           # 目标传输效率（0~1），用于反推送能

    # 若策略内部也采用 K 自适应，可在此设置其上限与权重（与 SimulationConfig 的 K 互不冲突）
    adaptive_k_max: int = 24
    adaptive_hysteresis: float = 0.2
    adaptive_w_b: float = 0.8               # 均衡改进权重
    adaptive_w_d: float = 0.8               # 有效送达量权重
    adaptive_w_l: float = 1.5               # 损耗惩罚权重
    # 统计口径提示：`SimulationStats.compute_step_stats()` 以 donor.E_char 估算本轮发送总量；
    # 若启用 PowerControlScheduler，会按 `energy_sent = min(1, target_eta/η_path) * E_char` 缩放实际发送量，
    # 因此“发送总量/损耗”的统计与真实发送存在偏差。后续可改为读取 plans 中的 energy_sent。


@dataclass
class ADCRConfig:
    """ADCR 链路层配置参数（聚类与上行路径）

    三部分：
    - 核心：聚类轮换、邻居半径、K 估计、跳数限制、是否规划路径与能耗结算；
    - 成簇：簇头选择概率边界，控制簇规模与稳定性；
    - 成本：聚类代价函数的距离/能量权重；
    - 可视化：图像导出尺寸、节点/簇头/虚拟中心的标记大小与连线宽度。
    """

    # 核心算法参数
    round_period: int = 1440        # 重聚类周期（分钟），例如每日重选簇头
    r_neighbor: float = 1.732       # 邻居检测半径 ≈ sqrt(3.0)（与直传判定阈值一致）
    r_min_ch: float = 1.0           # 簇头间最小距离（避免过密簇头）
    c_k: float = 1.2                # K 值估计系数（与 SimulationConfig.K 的关系：用于 ADCR 内部估计/限制）
    max_hops: int = 5               # 允许的最大多跳数
    plan_paths: bool = True         # 是否为簇头到虚拟中心规划真实节点路径
    consume_energy: bool = True     # 是否对通信过程执行能耗结算（扣减节点能量）
    output_dir: str = "adcr"        # ADCR 专属输出子目录（会在 session_dir 下创建）
    
    # 簇头选择参数
    max_probability: float = 0.9    # 最大簇头选择概率（上界）
    min_probability: float = 0.05   # 最小簇头选择概率（下界）
    
    # 聚类成本函数参数（距离/能量折衷）
    distance_weight: float = 1.0    # 距离权重（越大越偏好更近的聚类）
    energy_weight: float = 0.2      # 能量权重（越大越偏好能量更高的作为簇头）
    
    # 通信能耗参数
    tx_rx_ratio: float = 0.5        # 发送/接收能耗比例（0.5 表示均摊）
    # 注意：当前 ADCR 实现中通信扣能调用 `SensorNode.energy_consumption()`，未使用该比例参数；保留项。
    sensor_energy: float = 0.1      # 感知能耗 J/步（与 NodeConfig.sensor_energy 对齐）
    # 提示：同 NodeConfig，当前通信能耗中的传感能耗为固定常量 0.1 J，未与该配置绑定。
    
    # 信息聚合参数
    base_data_size: int = 1000000      # 基础数据大小（bits），每个节点贡献的基础信息量
    aggregation_ratio: float = 1.0      # 信息聚合比例（1.0表示完全聚合，0.5表示压缩50%）
    enable_dynamic_data_size: bool = True  # 是否启用基于簇大小的动态数据量
    
    # 直接传输优化参数
    enable_direct_transmission_optimization: bool = True  # 是否启用直接传输优化
    direct_transmission_threshold: float = 0.1  # 直接传输阈值（能耗比例，0.1表示直接传输能耗不超过锚点传输的110%）
    
    # 自动画图参数
    auto_plot: bool = False  # 是否每次重聚类后自动画图
    plot_filename_template: str = "adcr_day{day}_t{t}.png"  # 画图文件名模板

    # 可视化与导出
    image_width: int = 900          # 输出图像宽度 px
    image_height: int = 700         # 输出图像高度 px
    image_scale: int = 3            # 图像缩放（放大导出细节）
    node_marker_size: int = 7       # 普通节点标记大小
    ch_marker_size: int = 10        # 簇头标记大小
    vc_marker_size: int = 12        # 虚拟中心标记大小
    line_width: float = 1.0         # 一般连线宽度
    path_line_width: float = 2.0    # 路径线条宽度（关键路径更粗）


@dataclass
class PathCollectorConfig:
    """基于路径的信息收集器配置参数
    
    用于替代或补充ADCR的信息收集机制：
    - 利用能量传输路径收集节点信息
    - 路径节点：实时采集
    - 非路径节点：基于历史 + 模型估算
    
    能量消耗模式：
    - free: 零能耗（默认），信息完全搭载在传能路径上
    - full: 完全真实，路径逐跳 + 虚拟跳都消耗能量
    """
    
    # 基本开关
    enable_path_collector: bool = True  # 是否启用路径信息收集器
    replace_adcr: bool = True  # 是否替代ADCR（如果True，ADCR仅做聚类不更新虚拟中心）
    
    # 能量消耗模式
    energy_mode: str = "full"  # 能量消耗模式："free"（零能耗，默认）
                            # 或 "full"（完全真实，路径逐跳 + 虚拟跳都消耗能量）
    
    # 数据包大小配置（与ADCR保持一致）
    base_data_size: int = 1000000  # 基础数据大小（bits），每个节点贡献的基础信息量
    
    # 估算参数
    decay_rate: float = 5.0  # 自然衰减率（J/分钟，用于估算路径外节点能量）
    use_solar_model: bool = True  # 是否使用太阳能模型进行估算
    
    # 优化选项
    batch_update: bool = True  # 是否批量更新虚拟中心（减少开销）
    
    # 日志输出
    enable_logging: bool = True  # 是否启用详细日志


@dataclass
class ParallelConfig:
    """并行仿真配置参数（批量实验与扫描）

    - enabled 打开后，使用多进程进行多次独立仿真；
    - use_same_seed=True 可保证不同 run 的网络结构一致（用于对比不同策略/参数）；
    - enable_weight_scan 可在固定 K 自适应外，对 (w_b, w_d, w_l) 进行网格/序列扫描。
    """

    # 基本并行参数
    enabled: bool = False            # 是否启用并行模式
    num_runs: int = 10               # 并行/批量运行次数
    max_workers: int = 4             # 最大并行进程数
    
    # 种子管理
    use_same_seed: bool = True       # 是否使用相同种子（对比实验可控变量）
    base_seed: int = 42              # 基础种子值（用于派生各 run 的子种子）
    
    # 权重扫描实验（固定其中两项，扫描另一项或序列）
    enable_weight_scan: bool = False # 是否启用 (w_b, w_d, w_l) 权重扫描
    w_b_start: float = 0.1           # w_b 起始值
    w_b_step: float = 0.1            # w_b 步长（线性扫描）
    w_d_fixed: float = 0.8           # w_d 固定值
    w_l_fixed: float = 1.5           # w_l 固定值
    
    # 输出管理
    output_base_dir: str = "data/parallel"  # 并行输出基础目录（每 run 建立子目录）
    save_individual_results: bool = True     # 是否保存每次运行的独立结果
    generate_summary: bool = True            # 是否生成汇总报告（均值/方差等）


class ConfigManager:
    """配置管理器 - 使用 dataclass 默认值"""
    
    def __init__(self, config_file: Optional[str] = None):
        # 使用 dataclass 的默认值创建配置对象
        self.node_config = NodeConfig()
        self.network_config = NetworkConfig()
        self.simulation_config = SimulationConfig()
        self.scheduler_config = SchedulerConfig()
        self.adcr_config = ADCRConfig()
        self.path_collector_config = PathCollectorConfig()
        self.parallel_config = ParallelConfig()
        
        print("使用默认配置（来自 dataclass 默认值）")
        
        # 如果提供了其他配置文件，则覆盖默认配置
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str) -> None:
        """从JSON文件加载配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 更新各个配置对象
            if 'node' in config_data:
                self._update_dataclass(self.node_config, config_data['node'])
            if 'network' in config_data:
                self._update_dataclass(self.network_config, config_data['network'])
            if 'simulation' in config_data:
                self._update_dataclass(self.simulation_config, config_data['simulation'])
            if 'scheduler' in config_data:
                self._update_dataclass(self.scheduler_config, config_data['scheduler'])
            if 'adcr' in config_data:
                self._update_dataclass(self.adcr_config, config_data['adcr'])
            if 'path_collector' in config_data:
                self._update_dataclass(self.path_collector_config, config_data['path_collector'])
            if 'parallel' in config_data:
                self._update_dataclass(self.parallel_config, config_data['parallel'])
                
            print(f"配置已从 {config_file} 加载")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            print("使用默认配置")
    
    def save_to_file(self, config_file: str) -> None:
        """保存配置到JSON文件"""
        try:
            config_data = {
                'node': asdict(self.node_config),
                'network': asdict(self.network_config),
                'simulation': asdict(self.simulation_config),
                'scheduler': asdict(self.scheduler_config),
                'adcr': asdict(self.adcr_config),
                'parallel': asdict(self.parallel_config)
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)
            
            print(f"配置已保存到 {config_file}")
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def export_config_to_session(self, session_dir: str) -> str:
        """
        导出当前配置参数到会话目录
        
        Args:
            session_dir: 会话目录路径
            
        Returns:
            str: 配置文件路径
        """
        try:
            from utils.output_manager import OutputManager
            
            # 生成配置文件路径
            config_file = OutputManager.get_file_path(session_dir, "simulation_config.json")
            
            # 准备配置数据
            config_data = {
                'metadata': {
                    'export_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'description': '无线传感器网络仿真配置参数',
                    'version': '1.0'
                },
                'node': asdict(self.node_config),
                'network': asdict(self.network_config),
                'simulation': asdict(self.simulation_config),
                'scheduler': asdict(self.scheduler_config),
                'adcr': asdict(self.adcr_config),
                'parallel': asdict(self.parallel_config)
            }
            
            # 保存到文件
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)
            
            print(f"配置参数已导出到: {config_file}")
            return config_file
            
        except Exception as e:
            print(f"导出配置参数失败: {e}")
            return ""
    
    def _update_dataclass(self, obj, data: Dict[str, Any]) -> None:
        """更新dataclass对象的字段"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def get_network_config_dict(self) -> Dict[str, Any]:
        """获取网络配置字典（兼容现有代码）"""
        return {
            "num_nodes": self.network_config.num_nodes,
            "low_threshold": self.node_config.low_threshold,
            "high_threshold": self.node_config.high_threshold,
            "node_initial_energy": self.node_config.initial_energy,
            "max_hops": self.network_config.max_hops,
            "random_seed": self.network_config.random_seed,
            "distribution_mode": self.network_config.distribution_mode,
            "network_area": {
                "width": self.network_config.network_area_width,
                "height": self.network_config.network_area_height
            },
            "min_distance": self.network_config.min_distance,
            "output_dir": self.simulation_config.output_dir
        }
    
    def get_scheduler_params(self) -> Dict[str, Any]:
        """获取调度器参数字典"""
        scheduler_type = self.scheduler_config.scheduler_type
        
        if scheduler_type == "LyapunovScheduler":
            return {
                "V": self.scheduler_config.lyapunov_v,
                "K": self.scheduler_config.lyapunov_k,
                "max_hops": self.network_config.max_hops
            }
        elif scheduler_type == "ClusterScheduler":
            return {
                "round_period": self.scheduler_config.cluster_round_period,
                "K": self.scheduler_config.lyapunov_k,
                "max_hops": self.network_config.max_hops,
                "p_ch": self.scheduler_config.cluster_p_ch
            }
        elif scheduler_type == "PredictionScheduler":
            return {
                "alpha": self.scheduler_config.prediction_alpha,
                "horizon_min": self.scheduler_config.prediction_horizon,
                "K": self.scheduler_config.lyapunov_k,
                "max_hops": self.network_config.max_hops
            }
        elif scheduler_type == "PowerControlScheduler":
            return {
                "target_eta": self.scheduler_config.power_target_eta,
                "K": self.scheduler_config.lyapunov_k,
                "max_hops": self.network_config.max_hops
            }
        elif scheduler_type == "BaselineHeuristic":
            return {
                "K": self.scheduler_config.lyapunov_k,
                "max_hops": self.network_config.max_hops
            }
        else:
            raise ValueError(f"未知的调度器类型: {scheduler_type}")
    
    def create_network(self):
        """创建Network对象"""
        from core.network import Network
        return Network(
            num_nodes=self.network_config.num_nodes,
            low_threshold=self.node_config.low_threshold,
            high_threshold=self.node_config.high_threshold,
            node_initial_energy=self.node_config.initial_energy,
            max_hops=self.network_config.max_hops,
            distribution_mode=self.network_config.distribution_mode,
            network_area_width=self.network_config.network_area_width,
            network_area_height=self.network_config.network_area_height,
            min_distance=self.network_config.min_distance,
            random_seed=self.network_config.random_seed,
            solar_node_ratio=self.network_config.solar_node_ratio,
            mobile_node_ratio=self.network_config.mobile_node_ratio,
            output_dir=self.simulation_config.output_dir,
            use_gpu=self.simulation_config.use_gpu_acceleration,
            # 能量空洞模式参数
            enable_energy_hole=self.network_config.enable_energy_hole,
            energy_hole_center_mode=self.network_config.energy_hole_center_mode,
            energy_hole_mobile_ratio=self.network_config.energy_hole_mobile_ratio,
            # 能量采集参数
            enable_energy_harvesting=self.node_config.enable_energy_harvesting,
            # 能量分配模式参数
            energy_distribution_mode=self.network_config.energy_distribution_mode,
            center_energy=self.network_config.center_energy,
            edge_energy=self.network_config.edge_energy,
            # NodeConfig参数（传递给SensorNode）
            capacity=self.node_config.capacity,
            voltage=self.node_config.voltage,
            solar_efficiency=self.node_config.solar_efficiency,
            solar_area=self.node_config.solar_area,
            max_solar_irradiance=self.node_config.max_solar_irradiance,
            env_correction_factor=self.node_config.env_correction_factor,
            energy_char=self.node_config.energy_char,
            energy_elec=self.node_config.energy_elec,
            epsilon_amp=self.node_config.epsilon_amp,
            bit_rate=self.node_config.bit_rate,
            path_loss_exponent=self.node_config.path_loss_exponent,
            energy_decay_rate=self.node_config.energy_decay_rate,
            sensor_energy=self.node_config.sensor_energy,
            # 物理中心节点参数
            enable_physical_center=self.network_config.enable_physical_center,
            center_initial_energy_multiplier=self.network_config.center_initial_energy_multiplier
        )
    
    def create_sensor_node(self, node_id: int, position: list, 
                          has_solar: bool = True, is_mobile: bool = False,
                          mobility_pattern: str = None, mobility_params: dict = None):
        """创建SensorNode对象"""
        from core.SensorNode import SensorNode
        return SensorNode(
            node_id=node_id,
            initial_energy=self.node_config.initial_energy,
            low_threshold=self.node_config.low_threshold,
            high_threshold=self.node_config.high_threshold,
            position=position,
            has_solar=has_solar,
            # 电池参数
            capacity=self.node_config.capacity,
            voltage=self.node_config.voltage,
            # 太阳能参数
            enable_energy_harvesting=self.node_config.enable_energy_harvesting,
            solar_efficiency=self.node_config.solar_efficiency,
            solar_area=self.node_config.solar_area,
            max_solar_irradiance=self.node_config.max_solar_irradiance,
            env_correction_factor=self.node_config.env_correction_factor,
            # 传输参数
            energy_char=self.node_config.energy_char,
            energy_elec=self.node_config.energy_elec,
            epsilon_amp=self.node_config.epsilon_amp,
            bit_rate=self.node_config.bit_rate,
            path_loss_exponent=self.node_config.path_loss_exponent,
            energy_decay_rate=self.node_config.energy_decay_rate,
            sensor_energy=self.node_config.sensor_energy,
            # 移动性参数
            is_mobile=is_mobile,
            mobility_pattern=mobility_pattern,
            mobility_params=mobility_params
        )
    
    def create_energy_simulation(self, network, scheduler=None):
        """创建EnergySimulation对象"""
        from core.energy_simulation import EnergySimulation
        return EnergySimulation(
            network=network,
            time_steps=self.simulation_config.time_steps,
            scheduler=scheduler,
            enable_energy_sharing=self.simulation_config.enable_energy_sharing,
            enable_k_adaptation=self.simulation_config.enable_k_adaptation,
            initial_K=self.simulation_config.initial_K,
            K_max=self.simulation_config.K_max,
            hysteresis=self.simulation_config.hysteresis,
            w_b=self.simulation_config.w_b,
            w_d=self.simulation_config.w_d,
            w_l=self.simulation_config.w_l,
            use_lookahead=self.simulation_config.use_lookahead,
            fixed_k=self.simulation_config.fixed_k,
            output_dir=self.simulation_config.output_dir,
            use_gpu=self.simulation_config.use_gpu_acceleration,
            # 智能被动传能参数
            passive_mode=self.simulation_config.passive_mode,
            check_interval=self.simulation_config.check_interval,
            critical_ratio=self.simulation_config.critical_ratio,
            energy_variance_threshold=self.simulation_config.energy_variance_threshold,
            cooldown_period=self.simulation_config.cooldown_period,
            predictive_window=self.simulation_config.predictive_window,
            # 指令下发参数
            enable_command_downlink=self.simulation_config.enable_command_downlink,
            command_packet_size=self.simulation_config.command_packet_size
        )
    
    def create_adcr_link_layer(self, network):
        """创建ADCRLinkLayerVirtual对象"""
        from acdr.adcr_link_layer import ADCRLinkLayerVirtual
        return ADCRLinkLayerVirtual(
            network=network,
            # 核心算法参数
            round_period=self.adcr_config.round_period,
            r_neighbor=self.adcr_config.r_neighbor,
            r_min_ch=self.adcr_config.r_min_ch,
            c_k=self.adcr_config.c_k,
            max_hops=self.adcr_config.max_hops,
            plan_paths=self.adcr_config.plan_paths,
            consume_energy=self.adcr_config.consume_energy,
            output_dir=self.adcr_config.output_dir,
            # 簇头选择参数
            max_probability=self.adcr_config.max_probability,
            min_probability=self.adcr_config.min_probability,
            # 聚类成本函数参数
            distance_weight=self.adcr_config.distance_weight,
            energy_weight=self.adcr_config.energy_weight,
            # 通信能耗参数
            tx_rx_ratio=self.adcr_config.tx_rx_ratio,
            sensor_energy=self.adcr_config.sensor_energy,
            # 信息聚合参数
            base_data_size=self.adcr_config.base_data_size,
            aggregation_ratio=self.adcr_config.aggregation_ratio,
            enable_dynamic_data_size=self.adcr_config.enable_dynamic_data_size,
            # 直接传输优化参数
            enable_direct_transmission_optimization=self.adcr_config.enable_direct_transmission_optimization,
            direct_transmission_threshold=self.adcr_config.direct_transmission_threshold,
            # 自动画图参数
            auto_plot=self.adcr_config.auto_plot,
            plot_filename_template=self.adcr_config.plot_filename_template,
            # 可视化参数
            image_width=self.adcr_config.image_width,
            image_height=self.adcr_config.image_height,
            image_scale=self.adcr_config.image_scale,
            node_marker_size=self.adcr_config.node_marker_size,
            ch_marker_size=self.adcr_config.ch_marker_size,
            vc_marker_size=self.adcr_config.vc_marker_size,
            line_width=self.adcr_config.line_width,
            path_line_width=self.adcr_config.path_line_width
        )
    
    def create_path_collector(self, virtual_center, physical_center=None):
        """
        创建PathBasedInfoCollector对象
        
        :param virtual_center: 虚拟中心实例（用于节点信息表管理）
        :param physical_center: 物理中心节点（ID=0，信息上报目标）
        """
        from info_collection.path_based_collector import PathBasedInfoCollector
        return PathBasedInfoCollector(
            virtual_center=virtual_center,
            physical_center=physical_center,
            energy_mode=self.path_collector_config.energy_mode,
            base_data_size=self.path_collector_config.base_data_size,
            enable_logging=self.path_collector_config.enable_logging,
            decay_rate=self.path_collector_config.decay_rate,
            use_solar_model=self.path_collector_config.use_solar_model,
            batch_update=self.path_collector_config.batch_update
        )
    
    def __str__(self) -> str:
        return json.dumps({
            'node': asdict(self.node_config),
            'network': asdict(self.network_config),
            'simulation': asdict(self.simulation_config),
            'scheduler': asdict(self.scheduler_config)
        }, indent=2, ensure_ascii=False)


# 全局配置实例
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """获取全局配置管理器实例"""
    return config_manager


def load_config(config_file: str) -> ConfigManager:
    """加载配置文件"""
    global config_manager
    config_manager = ConfigManager(config_file)
    return config_manager
