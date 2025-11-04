"""
测试和分析DurationAwareLyapunovScheduler的duration分布和节点锁定情况
"""
import sys
import os
from collections import defaultdict

# 添加src目录到路径
src_dir = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_dir)

from config.simulation_config import ConfigManager
from core.network import Network
from core.energy_simulation import EnergySimulation
from sim.refactored_main import create_scheduler

class PlanCollector:
    """收集传输计划统计信息的收集器"""
    
    def __init__(self):
        self.duration_distribution = defaultdict(int)  # {duration: count}
        self.locked_nodes_history = []  # [(time, locked_count, available_count, plan_count)]
        self.total_plans = 0
        self.zero_plan_steps = []
        
    def collect_step(self, plans, current_time, nim):
        """收集一个时间步的统计信息"""
        plan_count = len(plans) if plans else 0
        
        # 只统计一次，避免重复计数（execute_energy_transfer调用时统计）
        # 但需要确保在传输后统计（此时已应用锁定状态）
        if current_time not in [t for t, _, _, _ in self.locked_nodes_history]:
            self.total_plans += plan_count
            
            if plan_count == 0:
                self.zero_plan_steps.append(current_time)
            
            # 统计duration分布
            for plan in plans:
                duration = plan.get("duration", 1)
                self.duration_distribution[duration] += 1
            
            # 统计锁定节点数
            locked_count = 0
            available_count = 0
            if nim:
                for node_id, info in nim.latest_info.items():
                    if info.get('is_locked', False):
                        lock_until = info.get('lock_until', -1)
                        if lock_until >= current_time:
                            locked_count += 1
                        else:
                            available_count += 1
                    else:
                        available_count += 1
            
            self.locked_nodes_history.append((current_time, locked_count, available_count, plan_count))
    
    def print_report(self, label=""):
        """打印分析报告"""
        print("\n" + "=" * 70)
        if label:
            print(f"{label} - 分析报告")
        else:
            print("分析报告")
        print("=" * 70)
        
        # Duration分布
        print("\n【Duration分布统计】")
        print("-" * 70)
        if self.total_plans > 0:
            print(f"总传输计划数: {self.total_plans}")
            print("\n各Duration的分布:")
            for duration in sorted(self.duration_distribution.keys()):
                count = self.duration_distribution[duration]
                percentage = (count / self.total_plans) * 100
                print(f"  duration={duration}分钟: {count}次 ({percentage:.1f}%)")
            
            # 统计duration>1的比例
            duration_gt1 = sum(count for d, count in self.duration_distribution.items() if d > 1)
            duration_gt1_percentage = (duration_gt1 / self.total_plans) * 100 if self.total_plans > 0 else 0
            print(f"\nduration>1的传输: {duration_gt1}次 ({duration_gt1_percentage:.1f}%)")
            if duration_gt1_percentage > 50:
                print("  [警告] 超过50%的传输选择了duration>1，可能导致大量节点锁定")
        else:
            print("没有生成任何传输计划")
        
        # 锁定节点统计
        print("\n【节点锁定统计】")
        print("-" * 70)
        if self.locked_nodes_history:
            locked_counts = [count for _, count, _, _ in self.locked_nodes_history]
            available_counts = [count for _, _, count, _ in self.locked_nodes_history]
            
            if locked_counts:
                max_locked = max(locked_counts)
                avg_locked = sum(locked_counts) / len(locked_counts)
                print(f"平均锁定节点数: {avg_locked:.1f}")
                print(f"最大锁定节点数: {max_locked}")
            
            if available_counts:
                min_available = min(available_counts)
                avg_available = sum(available_counts) / len(available_counts)
                print(f"\n平均可用节点数: {avg_available:.1f}")
                print(f"最少可用节点数: {min_available}")
                
            if min_available < 5:
                print(f"  [警告] 可用节点过少，可能导致调度器无法生成足够的传输计划")
                
                # 找到可用节点数较少的时间步
                low_available = [(t, count) for t, _, count, _ in self.locked_nodes_history if count < min_available + 2]
                if low_available:
                    print(f"\n可用节点数较少(<{min_available + 2})的时间步示例:")
                    for t, count in low_available[:5]:
                        matched = next(((l, a, p) for time, l, a, p in self.locked_nodes_history if time == t), None)
                        if matched:
                            locked, available, plans = matched
                            print(f"  时间步 {t}: {count}个可用, {locked}个锁定, {plans}个计划")
        else:
            print("没有锁定节点数据")
        
        # 传输计划统计
        print("\n【传输计划生成统计】")
        print("-" * 70)
        print(f"总传输计划数: {self.total_plans}")
        print(f"生成0个计划的时间步数: {len(self.zero_plan_steps)}")
        
        if self.locked_nodes_history:
            total_steps = len(self.locked_nodes_history)
            avg_plans_per_step = self.total_plans / total_steps if total_steps > 0 else 0
            print(f"平均每步计划数: {avg_plans_per_step:.2f}")
        
        if self.zero_plan_steps:
            zero_percentage = (len(self.zero_plan_steps) / len(self.locked_nodes_history)) * 100 if self.locked_nodes_history else 0
            print(f"生成0个计划的比例: {zero_percentage:.1f}%")
            if zero_percentage > 20:
                print("  [警告] 超过20%的时间步生成了0个计划，可能是节点锁定导致的")
            print(f"\n生成0个计划的时间步示例（前10个）:")
            for t in self.zero_plan_steps[:10]:
                matched = next(((l, a, p) for time, l, a, p in self.locked_nodes_history if time == t), None)
                if matched:
                    locked, available, plans = matched
                    print(f"  时间步 {t}: {available}个可用节点, {locked}个锁定节点")
                else:
                    print(f"  时间步 {t}: 数据不可用")


def run_test_with_config(config_manager, test_label):
    """运行测试并收集统计信息"""
    print(f"\n{'='*70}")
    print(f"测试: {test_label}")
    print('='*70)
    
    collector = PlanCollector()
    
    # 创建网络和调度器
    network = config_manager.create_network()
    scheduler = create_scheduler(config_manager, network)
    
    # 获取NodeInfoManager
    nim = None
    if hasattr(scheduler, 'nim') and scheduler.nim is not None:
        nim = scheduler.nim
    
    # 创建仿真
    simulation = config_manager.create_energy_simulation(network, scheduler=scheduler)
    
    # 拦截scheduler.plan的调用（收集计划信息）
    original_plan = scheduler.plan
    plan_cache = {}  # {time: plans}
    
    def wrapped_plan(network, t):
        result = original_plan(network, t)
        if isinstance(result, tuple):
            plans, _ = result
        else:
            plans = result
        
        # 缓存计划（传输前）
        plan_cache[t] = plans
        
        return result
    
    scheduler.plan = wrapped_plan
    
    # 拦截execute_energy_transfer后的状态（收集传输后的统计）
    original_execute = network.execute_energy_transfer
    def wrapped_execute(plans, current_time=None):
        result = original_execute(plans, current_time)
        
        # 在传输后收集统计（此时已经应用了锁定状态）
        if current_time is not None and nim:
            collector.collect_step(plans, current_time, nim)
        
        return result
    
    network.execute_energy_transfer = wrapped_execute
    
    # 运行仿真
    try:
        simulation.simulate()
        print(f"\n[测试完成] {test_label}")
        return collector
    except Exception as e:
        print(f"\n[测试失败] {test_label}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主测试函数"""
    print("=" * 70)
    print("DurationAwareLyapunovScheduler 深度分析测试")
    print("=" * 70)
    
    # ========== 测试1: 带传输时长优化 ==========
    config1 = ConfigManager()
    config1.scheduler_config.scheduler_type = "DurationAwareLyapunovScheduler"
    config1.scheduler_config.duration_min = 1
    config1.scheduler_config.duration_max = 5
    config1.scheduler_config.duration_w_aoi = 0.1
    config1.scheduler_config.duration_w_info = 0.05
    config1.simulation_config.time_steps = 1440  # 24小时，确保有足够触发次数
    config1.simulation_config.passive_mode = False  # 使用定时触发，确保能触发传输
    config1.simulation_config.energy_transfer_interval = 60  # 每60分钟触发一次
    
    collector1 = run_test_with_config(config1, "测试1: DurationAwareLyapunovScheduler (传输时长优化)")
    
    # ========== 测试2: duration固定为1（禁用优化）==========
    config2 = ConfigManager()
    config2.scheduler_config.scheduler_type = "DurationAwareLyapunovScheduler"
    config2.scheduler_config.duration_min = 1
    config2.scheduler_config.duration_max = 1  # 固定为1，禁用优化
    config2.scheduler_config.duration_w_aoi = 0.1
    config2.scheduler_config.duration_w_info = 0.05
    config2.simulation_config.time_steps = 360  # 6小时
    config2.simulation_config.passive_mode = False  # 使用定时触发（每60分钟）
    config2.simulation_config.check_interval = 60
    
    collector2 = run_test_with_config(config2, "测试2: DurationAwareLyapunovScheduler (duration=1固定)")
    
    # ========== 打印对比报告 ==========
    if collector1 and collector2:
        print("\n" + "=" * 70)
        print("详细对比分析")
        print("=" * 70)
        
        collector1.print_report("测试1 (duration优化)")
        collector2.print_report("测试2 (duration=1固定)")
        
        print("\n" + "=" * 70)
        print("关键指标对比")
        print("=" * 70)
        
        total1 = collector1.total_plans
        total2 = collector2.total_plans
        
        print(f"测试1总传输计划数: {total1}")
        print(f"测试2总传输计划数: {total2}")
        
        if total2 > 0:
            ratio = (total1 / total2) * 100
            reduction = total2 - total1
            print(f"\n测试1相对于测试2:")
            print(f"  传输计划数: {ratio:.1f}% ({reduction}次减少)")
            if ratio < 60:
                print(f"  [警告] 传输次数大幅减少超过40%，可能是节点锁定导致的副作用")
        
        # Duration分布对比
        duration_gt1_1 = sum(count for d, count in collector1.duration_distribution.items() if d > 1)
        duration_gt1_1_percentage = (duration_gt1_1 / total1 * 100) if total1 > 0 else 0
        
        print(f"\n测试1中duration>1的传输:")
        print(f"  数量: {duration_gt1_1}次 ({duration_gt1_1_percentage:.1f}%)")
        if duration_gt1_1_percentage > 50:
            print(f"  ⚠️  超过50%的传输选择了duration>1，这会导致大量节点锁定")
        
        # 零计划时间步对比
        zero1 = len(collector1.zero_plan_steps)
        zero2 = len(collector2.zero_plan_steps)
        
        print(f"\n生成0个计划的时间步数:")
        print(f"  测试1: {zero1}")
        print(f"  测试2: {zero2}")
        if zero1 > zero2 * 2:
            print(f"  [警告] 测试1生成0个计划的时间步明显更多，可能是节点锁定导致的")
        
        # 可用节点对比
        if collector1.locked_nodes_history and collector2.locked_nodes_history:
            min_avail1 = min(count for _, _, count, _ in collector1.locked_nodes_history)
            min_avail2 = min(count for _, _, count, _ in collector2.locked_nodes_history)
            
            print(f"\n最少可用节点数:")
            print(f"  测试1: {min_avail1}")
            print(f"  测试2: {min_avail2}")
            if min_avail1 < min_avail2:
                print(f"  [警告] 测试1的可用节点更少，可能导致调度失败")
        
        # 诊断结论
        print("\n" + "=" * 70)
        print("诊断结论")
        print("=" * 70)
        
        ratio = (total1 / total2 * 100) if total2 > 0 else 0
        
        if total2 == 0:
            print("\n[警告] 测试2没有生成任何传输计划，无法对比")
        elif total1 == 0:
            print("\n[警告] 测试1没有生成任何传输计划，可能是被动传输未触发")
        elif ratio < 60 and duration_gt1_1_percentage > 50:
            print("\n[问题确认] 节点锁定机制导致传输次数大幅减少")
            print("\n原因分析:")
            print("  1. 超过50%的传输选择了duration>1")
            print("  2. 这些传输锁定了大量节点（3-5分钟）")
            print("  3. 可用节点急剧减少，导致调度器无法生成足够的传输计划")
            print("  4. 传输次数被动减少，并非真正的优化效果")
            print("\n建议解决方案:")
            print("  1. 增加w_aoi权重（如0.5或1.0），加大对长传输的惩罚")
            print("  2. 降低w_info权重，或只在特定条件下给予信息奖励")
            print("  3. 添加约束：如果可用节点过少，强制选择duration=1")
            print("  4. 优化锁定策略：缩短锁定时间或允许部分节点复用")
        elif ratio >= 80:
            print("\n[正常] 传输次数减少在合理范围内")
            print("节点锁定机制影响较小，可能是真正的优化效果")
        else:
            print("\n[需要分析] 需要进一步分析")
            print("传输次数有一定减少，但可能同时存在优化和副作用")
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
