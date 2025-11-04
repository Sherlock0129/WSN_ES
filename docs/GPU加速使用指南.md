# GPU加速使用指南

## 📋 概述

本项目支持使用NVIDIA GPU加速计算，可以显著提升大规模网络仿真的性能。

## ✅ 安装完成

- **GPU型号**: NVIDIA GeForce RTX 3070 (8GB)
- **CUDA版本**: 12.9
- **CuPy版本**: 13.6.0 ✅ 已安装
- **测试状态**: ✅ 通过

## 🚀 如何启用GPU加速

### 方法1: 通过配置文件 (推荐)

创建或编辑配置JSON文件，例如 `config_gpu.json`:

```json
{
  "simulation": {
    "use_gpu_acceleration": true
  }
}
```

然后运行：

```bash
python src/sim/refactored_main.py --config config_gpu.json
```

### 方法2: 通过命令行参数

```bash
python src/sim/refactored_main.py --use-gpu
```

### 方法3: 在代码中直接设置

```python
from config.simulation_config import ConfigManager

config_manager = ConfigManager()
config_manager.simulation_config.use_gpu_acceleration = True

# 创建网络和仿真
network = config_manager.create_network()
simulation = config_manager.create_energy_simulation(network)
```

## 🔍 验证GPU加速是否生效

运行测试脚本：

```bash
python test_gpu_acceleration.py
```

预期输出：
```
GPU加速可用: CuPy已安装
GPU计算已启用，使用设备: <CUDA Device 0>
✅ 距离计算结果一致
```

## 📊 GPU加速的优势

GPU加速主要用于以下计算密集型操作：

1. **距离矩阵计算**: 节点间距离的批量计算
2. **能量效率计算**: 大规模能量传输效率评估
3. **统计分析**: 能量数据的均值、方差等统计量计算

### 实际性能测试结果 (RTX 3070)

**距离矩阵计算加速比**:
- 50 节点: 0.75x (CPU更快)
- 100 节点: 0.46x (CPU更快)
- 200 节点: 0.82x (CPU更快)
- 500 节点: 1.64x (GPU开始加速)

**统计计算加速比**:
- 1K 数据点: 0.10x (CPU更快)
- 10K 数据点: 0.09x (CPU更快)
- 100K 数据点: 0.36x (CPU更快)
- 1M 数据点: 2.25x (GPU加速显著)

**结论**: 
- GPU加速在**大规模计算**时有明显优势
- 小规模计算时，GPU数据传输开销大于计算收益
- **推荐节点数 > 500** 或 **数据规模 > 100万** 时使用GPU

## 🎯 GPU加速的应用场景

### ✅ 适合使用GPU的场景：

- **大规模网络**: 节点数量 > 500
- **超大规模数据**: 数据点 > 100万
- **批量计算**: 多次重复的矩阵运算
- **参数扫描**: 需要运行数百次仿真
- **长时间仿真**: time_steps > 10000

### ❌ 不建议使用GPU的场景：

- **小规模网络**: < 500节点 - CPU更快
- **小数据集**: < 10万数据点 - 数据传输开销大
- **单次快速仿真**: GPU初始化时间 > 仿真时间
- **GPU显存不足**: 大网络可能超出8GB显存限制
- **调试阶段**: CPU更容易调试和排错

## 🛠️ GPU计算API

项目提供了 `GPUComputeManager` 类，位于 `src/utils/gpu_compute.py`：

```python
from utils.gpu_compute import get_gpu_manager

# 创建GPU管理器
gpu_manager = get_gpu_manager(use_gpu=True)

# 创建数组（自动选择GPU或CPU）
data = gpu_manager.array([[1, 2], [3, 4]])

# 数学运算
result = gpu_manager.sqrt(data)
mean_val = gpu_manager.mean(data)

# GPU/CPU数据转换
cpu_data = gpu_manager.to_cpu(result)
gpu_data = gpu_manager.to_gpu(cpu_data)
```

### 专用加速函数

```python
from utils.gpu_compute import (
    compute_distance_matrix_gpu,
    compute_energy_efficiency_batch_gpu,
    compute_statistics_gpu,
    get_gpu_memory_info,
    cleanup_gpu_memory
)

# 批量距离计算
distance_matrix = compute_distance_matrix_gpu(nodes, gpu_manager)

# 批量效率计算
efficiency_matrix = compute_energy_efficiency_batch_gpu(
    donors, receivers, distances, gpu_manager
)

# 统计计算
mean, std, total = compute_statistics_gpu(energy_data, gpu_manager)

# 检查GPU内存
gpu_info = get_gpu_memory_info()
print(f"GPU内存使用: {gpu_info['used_memory'] / 1e9:.2f} GB")

# 清理GPU内存
cleanup_gpu_memory()
```

## 📈 性能监控

### 查看GPU使用情况

在仿真运行时，打开新终端：

```bash
nvidia-smi -l 1  # 每秒刷新一次
```

或使用Windows任务管理器的"性能"标签查看GPU利用率。

## ⚠️ 注意事项

1. **首次使用会较慢**: GPU初始化和数据传输需要时间
2. **内存管理**: 大规模计算后建议调用 `cleanup_gpu_memory()`
3. **混合计算**: CPU和GPU数据混用时注意类型转换
4. **错误回退**: 如果GPU不可用，会自动回退到CPU计算

## 🐛 故障排除

### 问题1: ImportError: No module named 'cupy'

**解决方案**:
```bash
pip install cupy-cuda12x  # CUDA 12.x
# 或
pip install cupy-cuda11x  # CUDA 11.x
```

### 问题2: GPU内存不足

**解决方案**:
```python
# 减小批处理大小
# 或在仿真循环中定期清理
from utils.gpu_compute import cleanup_gpu_memory
cleanup_gpu_memory()
```

### 问题3: GPU比CPU更慢

**可能原因**:
- 网络规模太小（建议 > 100节点）
- 数据传输开销大于计算收益
- GPU初始化时间

**解决方案**: 对小规模问题使用CPU模式

## 📝 配置示例

完整配置文件示例 `config_gpu_example.json`:

```json
{
  "simulation": {
    "use_gpu_acceleration": true,
    "time_steps": 2000,
    "random_seed": 42
  },
  "network": {
    "num_nodes": 200,
    "network_area_width": 10.0,
    "network_area_height": 10.0
  },
  "scheduler": {
    "scheduler_type": "lyapunov",
    "K": 5,
    "max_hops": 3
  }
}
```

## 🔗 相关文件

- `src/utils/gpu_compute.py` - GPU计算模块
- `src/core/network.py` - 网络类（支持GPU距离计算）
- `src/core/energy_simulation.py` - 能量仿真（支持GPU）
- `test_gpu_acceleration.py` - GPU功能测试

## 📚 参考资料

- [CuPy官方文档](https://docs.cupy.dev/)
- [CUDA编程指南](https://docs.nvidia.com/cuda/)
- [NumPy/CuPy接口对照](https://docs.cupy.dev/en/stable/user_guide/difference.html)

---

**最后更新**: 2025-11-04  
**GPU状态**: ✅ RTX 3070 正常工作

