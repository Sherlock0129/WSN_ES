# Utils 模块

## 概述

`utils` 模块提供各种工具函数和辅助类，包括日志、错误处理、输出管理、GPU加速等。

## 主要组件

### 1. Logger (logger.py)

统一日志管理系统。

**核心功能：**
- 统一日志记录接口
- 按日期自动创建日志目录
- 支持多种日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）
- 支持专门的统计日志器

**关键特性：**
- 单例模式
- 自动创建日志目录（logs/YYYYMMDD/）
- 支持控制台和文件输出
- 支持专门的统计日志器

**使用方法：**
```python
from utils.logger import logger, get_statistics_logger

# 使用主日志器
logger.info("信息日志")
logger.warning("警告日志")
logger.error("错误日志")

# 使用统计日志器
stats_logger = get_statistics_logger()
stats_logger.info("统计信息")
```

### 2. ErrorHandling (error_handling.py)

错误处理工具。

**核心功能：**
- 统一错误处理接口
- 错误日志记录
- 异常捕获和恢复

**关键函数：**
- `error_handler()`: 错误处理装饰器
- `handle_exceptions()`: 异常处理上下文管理器

**使用方法：**
```python
from utils.error_handling import error_handler, handle_exceptions

# 使用装饰器
@error_handler
def my_function():
    # 可能出错的代码
    pass

# 使用上下文管理器
with handle_exceptions():
    # 可能出错的代码
    pass
```

### 3. OutputManager (output_manager.py)

输出管理工具类。

**核心功能：**
- 统一管理文件输出路径
- 自动创建会话目录
- 支持并行运行目录
- 支持ADCR子目录

**关键方法：**
- `get_session_dir()`: 获取按日期+时间命名的会话目录
- `get_file_path()`: 获取文件完整路径
- `get_adcr_dir()`: 获取ADCR子目录
- `get_timestamped_filename()`: 获取带时间戳的文件名

**使用方法：**
```python
from utils.output_manager import OutputManager

# 获取会话目录
session_dir = OutputManager.get_session_dir("data")
# 返回: data/20251023_143022

# 获取文件路径
file_path = OutputManager.get_file_path(session_dir, "results.csv")
# 返回: data/20251023_143022/results.csv

# 获取ADCR目录
adcr_dir = OutputManager.get_adcr_dir(session_dir)
# 返回: data/20251023_143022/adcr
```

### 4. GPUCompute (gpu_compute.py)

GPU加速计算工具类。

**核心功能：**
- 提供可选的GPU加速功能
- 支持CuPy和NumPy之间的无缝切换
- 距离矩阵计算加速
- 统计计算加速

**关键特性：**
- 自动检测CuPy可用性
- CPU/GPU无缝切换
- GPU数组自动转换

**关键类：**
- `GPUComputeManager`: GPU计算管理器

**使用方法：**
```python
from utils.gpu_compute import get_gpu_manager, compute_distance_matrix_gpu

# 创建GPU管理器
gpu_manager = get_gpu_manager(use_gpu=True)

# 计算距离矩阵（自动使用GPU或CPU）
distance_matrix = compute_distance_matrix_gpu(nodes, gpu_manager)

# 使用GPU数组
x = gpu_manager.array([1, 2, 3])
y = gpu_manager.mean(x)
```

### 5. Parameters (parameters.py)

参数管理工具类（旧版参数管理，已被ConfigManager替代）。

**核心功能：**
- 参数定义和默认值管理
- 从JSON文件加载参数
- 保存参数到JSON文件
- 参数访问和设置

**注意：** 该模块为旧版参数管理系统，建议使用`ConfigManager`进行配置管理。保留此模块用于向后兼容。

**主要方法：**
- `load_from_file()`: 从JSON文件加载参数
- `save_to_file()`: 保存参数到JSON文件
- `get()`: 获取参数值
- `set()`: 设置参数值

### 6. Utils (utils.py)

通用工具函数集合。

**核心功能：**
- **文件操作**：
  - `read_json()`: 读取JSON文件
  - `write_json()`: 写入JSON文件
  - `save_to_csv()`: 保存数据到CSV文件
  
- **网络参数**：
  - `load_network_parameters()`: 从JSON配置文件加载网络参数
  - `load_initial_energy_distribution()`: 生成初始能量分布
  
- **能量处理**：
  - `normalize_energy()`: 归一化能量值到指定范围
  - 能量相关的辅助计算函数

**主要函数：**
- `read_json(file_path)`: 读取JSON文件并返回解析数据
- `write_json(data, file_path)`: 将数据写入JSON文件
- `save_to_csv(data, file_path)`: 将数据保存到CSV文件
- `load_network_parameters(file_path)`: 从JSON配置文件加载网络参数
- `load_initial_energy_distribution(num_nodes, min_energy, max_energy)`: 生成初始能量分布（均匀分布）
- `normalize_energy(energy, min_energy, max_energy)`: 将能量值归一化到指定范围

**使用方法：**
```python
from utils.utils import read_json, write_json, save_to_csv

# 读取JSON文件
config = read_json("config.json")

# 写入JSON文件
write_json(data, "output.json")

# 保存到CSV
save_to_csv(data, "results.csv")
```

## 文件结构

```
utils/
├── __pycache__/
├── error_handling.py    # 错误处理工具
├── gpu_compute.py       # GPU加速计算工具
├── logger.py            # 日志管理系统
├── output_manager.py    # 输出管理工具
├── parameters.py        # 参数管理工具
└── utils.py             # 通用工具函数
```

## 使用示例

### 示例1：日志记录

```python
from utils.logger import logger

logger.info("开始仿真")
logger.warning("能量不足")
logger.error("节点失效")
```

### 示例2：输出管理

```python
from utils.output_manager import OutputManager

# 创建会话目录
session_dir = OutputManager.get_session_dir()

# 保存文件
file_path = OutputManager.get_file_path(session_dir, "results.csv")
with open(file_path, 'w') as f:
    f.write("results")
```

### 示例3：GPU加速

```python
from utils.gpu_compute import get_gpu_manager

# 创建GPU管理器
gpu_manager = get_gpu_manager(use_gpu=True)

# 使用GPU数组
x = gpu_manager.array([1, 2, 3, 4, 5])
mean = gpu_manager.mean(x)
std = gpu_manager.std(x)
```

### 示例4：错误处理

```python
from utils.error_handling import error_handler

@error_handler
def risky_function():
    # 可能出错的代码
    result = 1 / 0
    return result
```

## 相关文档

- [GPU加速使用指南](../../GPU加速使用指南.md)
- [GPU加速快速开始](../../GPU加速快速开始.md)

