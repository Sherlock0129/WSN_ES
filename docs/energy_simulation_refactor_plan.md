# EnergySimulation构造函数重构方案

## 当前问题
```python
# 当前方式：硬编码 + 手动设置
simulation = EnergySimulation(network, time_steps, scheduler)
simulation.K_max = adaptive_params["K_max"]
simulation.hysteresis = adaptive_params["hysteresis"]
simulation.w_b = adaptive_params["w_b"]
simulation.w_d = adaptive_params["w_d"]
simulation.w_l = adaptive_params["w_l"]
```

## 重构方案

### 方案1：直接参数传入
```python
def __init__(self, network, time_steps, scheduler=None, 
             # 自适应K值参数
             initial_K=1, K_max=24, hysteresis=0.2, 
             w_b=0.8, w_d=0.8, w_l=1.5,
             # 其他参数
             use_lookahead=False):
```

### 方案2：配置对象传入
```python
def __init__(self, network, time_steps, scheduler=None, 
             simulation_config=None):
    # 从配置对象获取参数
    if simulation_config:
        self.K = simulation_config.initial_K
        self.K_max = simulation_config.K_max
        # ...
```

### 方案3：混合方式（推荐）
```python
def __init__(self, network, time_steps, scheduler=None, 
             # 核心参数
             initial_K=1, K_max=24, hysteresis=0.2, 
             w_b=0.8, w_d=0.8, w_l=1.5,
             use_lookahead=False,
             # 可选：完整配置对象
             simulation_config=None):
```

## 优势分析

### ✅ 优势
1. **参数集中**：所有参数在构造函数中明确定义
2. **类型安全**：参数类型和默认值清晰可见
3. **向后兼容**：可以保持默认值，不影响现有代码
4. **配置灵活**：既支持直接传参，也支持配置对象
5. **维护性强**：参数变更只需要修改构造函数

### ⚠️ 注意事项
1. **参数过多**：构造函数参数可能变得很长
2. **默认值管理**：需要确保默认值与配置一致
3. **向后兼容**：需要保持现有调用方式可用

## 实现建议

### 推荐方案：混合方式
- 核心参数直接传入（有默认值）
- 支持配置对象覆盖
- 保持向后兼容性
- 参数按功能分组
