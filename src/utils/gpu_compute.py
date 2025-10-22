"""
GPU加速计算工具类
提供可选的GPU加速功能，支持CuPy和NumPy之间的无缝切换
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import warnings

# 尝试导入CuPy，如果失败则使用NumPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("GPU加速可用: CuPy已安装")
except ImportError:
    cp = np
    CUPY_AVAILABLE = False
    print("GPU加速不可用: CuPy未安装，使用NumPy")


class GPUComputeManager:
    """GPU计算管理器，提供CPU/GPU计算的无缝切换"""
    
    def __init__(self, use_gpu: bool = False):
        """
        初始化GPU计算管理器
        
        Args:
            use_gpu: 是否使用GPU加速
        """
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        if use_gpu and not CUPY_AVAILABLE:
            warnings.warn("GPU加速请求但CuPy不可用，回退到CPU计算")
        
        if self.use_gpu:
            print(f"GPU计算已启用，使用设备: {cp.cuda.Device()}")
        else:
            print("使用CPU计算")
    
    def array(self, data, **kwargs):
        """创建数组，自动选择CPU或GPU"""
        return self.xp.array(data, **kwargs)
    
    def zeros(self, shape, **kwargs):
        """创建零数组"""
        return self.xp.zeros(shape, **kwargs)
    
    def ones(self, shape, **kwargs):
        """创建一数组"""
        return self.xp.ones(shape, **kwargs)
    
    def sqrt(self, x):
        """计算平方根"""
        return self.xp.sqrt(x)
    
    def exp(self, x):
        """计算指数"""
        return self.xp.exp(x)
    
    def mean(self, x, **kwargs):
        """计算均值"""
        return self.xp.mean(x, **kwargs)
    
    def std(self, x, **kwargs):
        """计算标准差"""
        return self.xp.std(x, **kwargs)
    
    def sum(self, x, **kwargs):
        """计算求和"""
        return self.xp.sum(x, **kwargs)
    
    def norm(self, x, axis=None, **kwargs):
        """计算范数"""
        return self.xp.linalg.norm(x, axis=axis, **kwargs)
    
    def to_cpu(self, x):
        """将GPU数组转换为CPU数组"""
        if self.use_gpu and hasattr(x, 'get'):
            return x.get()  # CuPy数组转NumPy
        return x
    
    def to_gpu(self, x):
        """将CPU数组转换为GPU数组"""
        if self.use_gpu and not hasattr(x, 'device'):
            return cp.asarray(x)  # NumPy数组转CuPy
        return x


def compute_distance_matrix_gpu(nodes: List, gpu_manager: GPUComputeManager) -> np.ndarray:
    """
    使用GPU加速计算节点间距离矩阵
    
    Args:
        nodes: 节点列表
        gpu_manager: GPU计算管理器
        
    Returns:
        距离矩阵 (numpy数组)
    """
    # 提取节点位置
    positions = gpu_manager.array([[n.position[0], n.position[1]] for n in nodes])
    
    # 计算所有节点间的距离
    # 使用广播计算: positions[i] - positions[j] 对所有i,j
    diff = positions[:, None, :] - positions[None, :, :]
    distances = gpu_manager.norm(diff, axis=2)
    
    # 转换为CPU数组返回
    return gpu_manager.to_cpu(distances)


def compute_energy_efficiency_batch_gpu(
    donors: List, 
    receivers: List, 
    distances: np.ndarray,
    gpu_manager: GPUComputeManager
) -> np.ndarray:
    """
    批量计算能量传输效率
    
    Args:
        donors: 发送节点列表
        receivers: 接收节点列表  
        distances: 距离矩阵
        gpu_manager: GPU计算管理器
        
    Returns:
        效率矩阵 (numpy数组)
    """
    # 将距离矩阵转换为GPU数组
    dist_gpu = gpu_manager.to_gpu(distances)
    
    # 批量计算效率 (使用指数衰减模型)
    efficiency = gpu_manager.exp(-dist_gpu / 2.0)
    
    # 转换为CPU数组返回
    return gpu_manager.to_cpu(efficiency)


def compute_statistics_gpu(
    energy_data: Union[List, np.ndarray],
    gpu_manager: GPUComputeManager
) -> Tuple[float, float, float]:
    """
    使用GPU加速计算能量统计信息
    
    Args:
        energy_data: 能量数据
        gpu_manager: GPU计算管理器
        
    Returns:
        (均值, 标准差, 总和)
    """
    # 转换为GPU数组
    data_gpu = gpu_manager.array(energy_data, dtype=float)
    
    # 计算统计信息
    mean_val = gpu_manager.mean(data_gpu)
    std_val = gpu_manager.std(data_gpu)
    sum_val = gpu_manager.sum(data_gpu)
    
    # 转换为CPU标量返回
    return (
        float(gpu_manager.to_cpu(mean_val)),
        float(gpu_manager.to_cpu(std_val)),
        float(gpu_manager.to_cpu(sum_val))
    )


def get_gpu_memory_info() -> dict:
    """获取GPU内存信息"""
    if not CUPY_AVAILABLE:
        return {"available": False, "message": "CuPy不可用"}
    
    try:
        mempool = cp.get_default_memory_pool()
        return {
            "available": True,
            "total_memory": cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'],
            "used_memory": mempool.used_bytes(),
            "free_memory": mempool.free_bytes()
        }
    except Exception as e:
        return {"available": True, "error": str(e)}


# 全局GPU管理器实例
_global_gpu_manager: Optional[GPUComputeManager] = None


def get_gpu_manager(use_gpu: bool = False) -> GPUComputeManager:
    """获取全局GPU管理器实例"""
    global _global_gpu_manager
    if _global_gpu_manager is None or _global_gpu_manager.use_gpu != use_gpu:
        _global_gpu_manager = GPUComputeManager(use_gpu)
    return _global_gpu_manager


def cleanup_gpu_memory():
    """清理GPU内存"""
    if CUPY_AVAILABLE:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            print("GPU内存已清理")
        except Exception as e:
            print(f"GPU内存清理失败: {e}")

