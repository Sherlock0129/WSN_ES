"""
分析DurationAwareLyapunovScheduler的传输时长分布和节点锁定情况
"""
import sys
import os

# 添加src目录到路径
src_dir = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_dir)

def analyze_problem():
    """
    分析DurationAwareLyapunovScheduler可能导致传输次数大幅减少的原因
    """
    print("=" * 70)
    print("DurationAwareLyapunovScheduler 潜在问题分析")
    print("=" * 70)
    
    print("\n【问题1：节点锁定导致可用节点急剧减少】")
    print("-" * 70)
    print("""
    当DurationAwareLyapunovScheduler选择传输时长>1时：
    1. 路径中的所有节点（donor、receiver、中继）会被锁定
    2. 锁定时间 = 当前时间 + 传输时长（duration）
    3. 在锁定期间，这些节点无法参与新的能量传输
    
    潜在问题：
    - 如果大部分传输选择了duration=3或5分钟，节点会被锁定3-5分钟
    - 在小型网络中（如30个节点），如果每次传输锁定3-5个节点
    - 连续几个时间步后，可用节点可能急剧减少
    - 调度器可能因为找不到足够的可用donor/receiver而减少传输次数
    
    例如：
    - 30个节点的网络，每次传输锁定3个节点（单跳）
    - 如果每次传输duration=3分钟，连续10个时间步传输
    - 理论上最多可能有30个节点被锁定（虽然有些会解锁）
    - 实际可用节点可能大幅减少，导致后续无法生成传输计划
    """)
    
    print("\n【问题2：评分函数可能偏向长传输时长】")
    print("-" * 70)
    print("""
    DurationAwareLyapunovScheduler的评分函数：
    - energy_benefit_score = energy_delivered × Q_normalized
    - energy_delivered = duration × E_char × eta（传输时长越长，传输能量越多）
    - energy_loss_penalty = V × energy_loss
    - aoi_penalty = w_aoi × duration × Q_normalized（传输时长越长，惩罚越大）
    - info_bonus = w_info × info_gain（传输时长越长，奖励越大，如果有信息）
    
    如果：
    - w_aoi较小（如0.1），AoI惩罚较小
    - w_info较大或receiver有未上报信息，info_bonus为正
    - energy_benefit_score随duration线性增长
    
    那么评分函数可能更倾向于选择较长的传输时长（3-5分钟），因为：
    - 能量传输量增加（duration倍）
    - AoI惩罚相对较小
    - 信息奖励可能补偿AoI惩罚
    
    这会导致大部分传输选择duration>1，从而触发大量节点锁定
    """)
    
    print("\n【问题3：路径收集次数依赖于能量传输次数】")
    print("-" * 70)
    print("""
    路径收集的工作机制：
    1. 每次能量传输后，会调用path_info_collector.collect_and_report()
    2. 每次调用会记录一次path_collector_transmission_count
    3. 路径收集的能量消耗 = 路径跳能量 + 上报跳能量
    
    如果能量传输次数减少（如从2271次降到411次）：
    - 路径收集次数也会相应减少（411次）
    - 总能量消耗 = 411次 × 每次路径收集的能量
    - 即使每次路径收集的能量相同，总消耗也会大幅减少
    
    但这不一定是"优化效果"，而是因为：
    - 节点锁定导致可用节点减少
    - 调度器无法生成足够的传输计划
    - 传输次数被动减少
    """)
    
    print("\n【建议检查项】")
    print("-" * 70)
    print("""
    1. 检查duration分布：
       - 查看实际生成的传输计划中，duration=1的占多少？
       - duration=2, 3, 4, 5的分别占多少？
       - 如果大部分都是duration>1，说明评分函数可能偏向长传输
    
    2. 检查节点锁定情况：
       - 在每个时间步，有多少节点被锁定？
       - 可用节点数（未锁定）是多少？
       - 是否经常出现"找不到可用donor/receiver"的情况？
    
    3. 检查调度器输出：
       - 查看plans的数量变化趋势
       - 是否有时间步生成0个plan？
       - 是否有时间步因为找不到可用节点而跳过？
    
    4. 对比测试：
       - 设置duration_min=duration_max=1，禁用传输时长优化
       - 看看传输次数是否恢复正常
       - 如果恢复正常，说明问题确实是节点锁定导致的
    
    5. 检查评分函数权重：
       - w_aoi=0.1可能太小，无法有效惩罚长传输
       - w_info=0.05如果有信息奖励，可能补偿了AoI惩罚
       - 可能需要调整权重平衡
    """)
    
    print("\n【可能的解决方案】")
    print("-" * 70)
    print("""
    1. 调整评分函数权重：
       - 增加w_aoi，加大对长传输的惩罚
       - 降低w_info，或者只在特定条件下给予信息奖励
       - 确保AoI惩罚能有效抑制过长的传输时长
    
    2. 优化节点锁定策略：
       - 考虑部分锁定（只锁定donor和receiver，不锁定中继节点）
       - 或者缩短锁定时间（例如锁定时间 = duration - 1）
       - 或者允许锁定的节点作为中继，但不能作为donor/receiver
    
    3. 添加约束条件：
       - 限制每个时间步锁定的节点数量上限
       - 如果可用节点过少，强制选择duration=1
       - 平衡传输时长优化和网络可用性
    """)
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    analyze_problem()
