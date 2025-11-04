#!/usr/bin/env python3
"""
验证动态K是否已关闭
快速检查配置文件和默认设置
"""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from config.simulation_config import ConfigManager, load_config

def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def check_default_config():
    """检查默认配置"""
    print_section("1. 检查默认配置")
    config = ConfigManager()
    
    print(f"enable_k_adaptation: {config.simulation_config.enable_k_adaptation}")
    print(f"fixed_k: {config.simulation_config.fixed_k}")
    print(f"initial_K: {config.simulation_config.initial_K}")
    print(f"K_max: {config.simulation_config.K_max}")
    
    if not config.simulation_config.enable_k_adaptation:
        print("\n✅ 动态K已关闭（默认配置）")
        print(f"✅ 使用固定K值: {config.simulation_config.fixed_k}")
    else:
        print("\n⚠️ 动态K已启用（默认配置）")
        print("   建议设置 enable_k_adaptation: false")
    
    return config

def check_config_file(config_file):
    """检查配置文件"""
    print_section(f"检查配置文件: {config_file}")
    
    if not os.path.exists(config_file):
        print(f"❌ 文件不存在: {config_file}")
        return None
    
    try:
        # 加载配置
        if config_file.endswith('.json'):
            with open(config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        elif config_file.endswith(('.yaml', '.yml')):
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        else:
            print(f"❌ 不支持的文件格式: {config_file}")
            return None
        
        # 检查simulation配置
        sim_config = config_dict.get('simulation', {})
        enable_k_adaptation = sim_config.get('enable_k_adaptation', None)
        fixed_k = sim_config.get('fixed_k', None)
        
        print(f"enable_k_adaptation: {enable_k_adaptation}")
        print(f"fixed_k: {fixed_k}")
        
        if enable_k_adaptation is False:
            print(f"\n✅ 动态K已关闭")
            print(f"✅ 使用固定K值: {fixed_k if fixed_k else '默认值'}")
        elif enable_k_adaptation is True:
            print(f"\n⚠️ 动态K已启用")
            print("   建议修改为: enable_k_adaptation: false")
        else:
            print(f"\n⚠️ 未设置 enable_k_adaptation")
            print("   将使用默认值（False）")
        
        return config_dict
        
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        return None

def check_all_configs():
    """检查所有配置文件"""
    config_files = [
        'config_fixed_k.json',
        'config_gpu_example.json',
        'config_dqn_example.yaml',
        'test_config.json'
    ]
    
    print_section("2. 检查所有配置文件")
    
    results = {}
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"\n📄 {config_file}:")
            try:
                if config_file.endswith('.json'):
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_dict = json.load(f)
                else:
                    import yaml
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_dict = yaml.safe_load(f)
                
                sim_config = config_dict.get('simulation', {})
                enable_k = sim_config.get('enable_k_adaptation', 'not set')
                fixed_k = sim_config.get('fixed_k', 'not set')
                
                print(f"   enable_k_adaptation: {enable_k}")
                print(f"   fixed_k: {fixed_k}")
                
                if enable_k is False or enable_k == 'not set':
                    print(f"   ✅ 动态K关闭")
                    results[config_file] = 'OK'
                else:
                    print(f"   ⚠️ 动态K启用")
                    results[config_file] = 'WARN'
            except:
                print(f"   ❌ 读取失败")
                results[config_file] = 'ERROR'
    
    return results

def print_summary(results):
    """打印总结"""
    print_section("总结")
    
    print("\n配置文件检查结果:")
    for config_file, status in results.items():
        if status == 'OK':
            print(f"  ✅ {config_file}")
        elif status == 'WARN':
            print(f"  ⚠️ {config_file}")
        else:
            print(f"  ❌ {config_file}")
    
    print("\n💡 使用建议:")
    print("  1. 默认配置已关闭动态K（enable_k_adaptation: False）")
    print("  2. 推荐使用配置文件明确设置 enable_k_adaptation: false")
    print("  3. 根据网络规模选择合适的 fixed_k 值（推荐2-5）")
    
    print("\n🚀 快速开始（使用固定K）:")
    print("  python src/sim/refactored_main.py --config config_fixed_k.json")
    print("  python src/sim/refactored_main.py --config config_gpu_example.json")
    
    print("\n📖 详细文档:")
    print("  关闭动态K配置说明.md")

def main():
    """主函数"""
    print("\n" + "=" * 70)
    print(" " * 20 + "动态K配置验证工具")
    print("=" * 70)
    
    # 1. 检查默认配置
    check_default_config()
    
    # 2. 检查所有配置文件
    results = check_all_configs()
    
    # 3. 打印总结
    print_summary(results)
    
    print("\n" + "=" * 70)
    print("验证完成！")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()


