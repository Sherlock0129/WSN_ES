# GPU加速快速开始 🚀

## ✅ 安装状态

- **CuPy**: ✅ 已安装 (v13.6.0)
- **GPU**: ✅ RTX 3070 (8GB, CUDA 12.9)
- **测试**: ✅ 通过

## 📦 已完成

1. ✅ 安装 `cupy-cuda12x`
2. ✅ 更新 `requirements.txt`
3. ✅ GPU功能测试通过
4. ✅ 性能基准测试完成
5. ✅ 创建配置示例和文档

## 🎯 何时使用GPU

### ✅ 推荐场景
- 节点数 > 500
- 数据点 > 100万
- 长时间仿真 (time_steps > 10000)

### ❌ 不推荐场景
- 节点数 < 200 (CPU更快)
- 小规模数据
- 快速单次测试

## 🚀 三种启动方式

### 方式1: 配置文件 (推荐)

```bash
# 使用GPU配置文件
python src/sim/refactored_main.py --config config_gpu_example.json
```

### 方式2: 命令行参数

```bash
# 大规模网络 + GPU
python src/sim/refactored_main.py --use-gpu --num-nodes 500 --time-steps 5000
```

### 方式3: 代码中设置

```python
from config.simulation_config import ConfigManager

config = ConfigManager()
config.simulation_config.use_gpu_acceleration = True
config.network_config.num_nodes = 500

network = config.create_network()
simulation = config.create_energy_simulation(network)
simulation.run()
```

## 🧪 测试命令

```bash
# 1. 基础功能测试
python test_gpu_acceleration.py

# 2. 性能对比测试
python benchmark_gpu_performance.py

# 3. 查看GPU状态
nvidia-smi
```

## 📊 性能测试结果

| 节点数 | CPU时间 | GPU时间 | 加速比 |
|--------|---------|---------|--------|
| 50     | 0.16ms  | 0.21ms  | 0.75x  |
| 100    | 0.22ms  | 0.47ms  | 0.46x  |
| 200    | 0.65ms  | 0.80ms  | 0.82x  |
| 500    | 6.48ms  | 3.96ms  | **1.64x** ✅ |

**结论**: 500+节点时GPU开始显示加速效果

## 📁 相关文件

- `requirements.txt` - 依赖配置（已添加cupy）
- `config_gpu_example.json` - GPU配置示例
- `test_gpu_acceleration.py` - GPU功能测试
- `benchmark_gpu_performance.py` - 性能基准测试
- `GPU加速使用指南.md` - 完整使用文档
- `src/utils/gpu_compute.py` - GPU计算模块

## ⚡ 快速示例

### 运行一个大规模GPU加速仿真

```bash
# 创建大规模网络配置
cat > config_large_gpu.json << EOF
{
  "simulation": {
    "use_gpu_acceleration": true,
    "time_steps": 5000
  },
  "network": {
    "num_nodes": 500,
    "network_area_width": 20.0,
    "network_area_height": 20.0
  },
  "scheduler": {
    "scheduler_type": "lyapunov",
    "K": 10
  }
}
EOF

# 运行仿真
python src/sim/refactored_main.py --config config_large_gpu.json
```

## 🔍 监控GPU使用

在仿真运行时，打开新终端：

```bash
# 实时监控GPU
nvidia-smi -l 1

# 或使用Windows任务管理器
# 按 Ctrl+Shift+Esc -> 性能 -> GPU
```

## 💡 使用建议

1. **开发/调试阶段**: 使用CPU模式（默认），更容易调试
2. **小规模测试**: 使用CPU模式（< 200节点）
3. **大规模实验**: 启用GPU模式（> 500节点）
4. **参数扫描**: 启用GPU模式，批量计算效果更好

## ⚠️ 注意事项

1. 首次运行GPU模式会有初始化延迟
2. 数据在CPU/GPU间传输有开销
3. 定期清理GPU内存（代码已自动处理）
4. 如果GPU不可用，会自动回退到CPU

## 🐛 故障排除

### GPU内存不足

```python
# 在仿真循环中定期清理
from utils.gpu_compute import cleanup_gpu_memory
cleanup_gpu_memory()
```

### GPU没有加速

- 检查节点数是否 > 500
- 检查是否真的启用了GPU
- 查看 `nvidia-smi` 确认GPU在运行

### CuPy导入错误

```bash
# 重新安装
pip uninstall cupy-cuda12x
pip install cupy-cuda12x
```

## 📚 更多信息

详细文档请查看: `GPU加速使用指南.md`

---

**状态**: ✅ 已完成安装和测试  
**GPU**: RTX 3070 (8GB)  
**CUDA**: 12.9  
**CuPy**: 13.6.0  
**最后测试**: 2025-11-04


