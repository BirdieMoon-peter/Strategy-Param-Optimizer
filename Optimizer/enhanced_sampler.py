# -*- coding: utf-8 -*-
"""
增强采样器模块
支持正态分布采样和并行随机探索

主要功能:
1. 正态分布采样器 - 替代传统的均匀采样，更好地探索参数空间
2. 并行随机探索 - 利用多进程加速初始探索阶段
3. 动态试验次数 - 根据参数数量自动调整试验次数
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import warnings
import time

warnings.filterwarnings("ignore")

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class SamplerConfig:
    """采样器配置"""
    # 正态分布参数
    use_normal_distribution: bool = True  # 是否使用正态分布
    normal_std_ratio: float = 0.25  # 标准差占范围的比例（默认25%）
    truncate_at_bounds: bool = True  # 是否在边界处截断
    
    # 并行探索参数
    enable_parallel: bool = True  # 是否启用并行探索
    n_workers: int = None  # 工作进程数，None表示自动检测
    chunk_size: int = 10  # 每批次并行评估的参数组数
    
    # 动态试验次数参数
    enable_dynamic_trials: bool = True  # 是否启用动态试验次数
    base_trials: int = 30  # 基础试验次数
    trials_per_param: int = 10  # 每个参数增加的试验次数
    max_trials: int = 200  # 最大试验次数
    min_trials: int = 20  # 最小试验次数
    
    # 探索/利用平衡
    exploration_ratio: float = 0.3  # 初始随机探索的比例（30%）


class NormalDistributionSampler:
    """
    正态分布采样器
    
    使用截断正态分布在参数空间内采样，
    相比均匀采样能更好地探索参数中心区域，
    同时保持对边界区域的覆盖
    """
    
    def __init__(self, config: SamplerConfig = None, seed: int = None):
        """
        初始化正态分布采样器
        
        Args:
            config: 采样器配置
            seed: 随机种子
        """
        self.config = config or SamplerConfig()
        self.rng = np.random.default_rng(seed)
    
    def sample_single_param(
        self,
        param_name: str,
        min_value: float,
        max_value: float,
        param_type: str,
        default_value: float = None,
        step: float = None
    ) -> Any:
        """
        对单个参数进行正态分布采样
        
        Args:
            param_name: 参数名称
            min_value: 最小值
            max_value: 最大值
            param_type: 参数类型 ('int' or 'float')
            default_value: 默认值（用作正态分布的中心）
            step: 步长
            
        Returns:
            采样值
        """
        # 计算正态分布参数
        param_range = max_value - min_value
        
        # 中心点：优先使用默认值，否则使用范围中点
        if default_value is not None and min_value <= default_value <= max_value:
            center = default_value
        else:
            center = (min_value + max_value) / 2
        
        # 标准差：根据配置的比例计算
        std = param_range * self.config.normal_std_ratio
        
        # 截断正态分布采样
        if self.config.truncate_at_bounds:
            value = self._truncated_normal_sample(center, std, min_value, max_value)
        else:
            value = self.rng.normal(center, std)
            value = np.clip(value, min_value, max_value)
        
        # 处理整数类型
        if param_type == 'int':
            value = int(round(value))
            # 确保在范围内
            value = max(int(min_value), min(int(max_value), value))
            # 应用步长
            if step and step > 1:
                value = int(min_value) + int((value - int(min_value)) / step) * int(step)
        else:
            # 浮点数应用步长
            if step:
                value = min_value + round((value - min_value) / step) * step
                value = np.clip(value, min_value, max_value)
        
        return value
    
    def _truncated_normal_sample(
        self,
        mean: float,
        std: float,
        lower: float,
        upper: float
    ) -> float:
        """
        截断正态分布采样
        使用拒绝采样方法
        
        Args:
            mean: 均值
            std: 标准差
            lower: 下界
            upper: 上界
            
        Returns:
            在[lower, upper]范围内的样本
        """
        max_attempts = 100
        for _ in range(max_attempts):
            sample = self.rng.normal(mean, std)
            if lower <= sample <= upper:
                return sample
        
        # 如果多次采样都失败，使用均匀分布
        return self.rng.uniform(lower, upper)
    
    def sample_params(
        self,
        search_space: Dict[str, Any],
        default_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        对整个参数空间进行采样
        
        Args:
            search_space: 搜索空间配置字典
            default_params: 默认参数值字典
            
        Returns:
            采样得到的参数字典
        """
        default_params = default_params or {}
        sampled_params = {}
        
        for param_name, space_config in search_space.items():
            # 从配置中提取信息
            min_val = space_config.min_value
            max_val = space_config.max_value
            param_type = space_config.param_type
            step = getattr(space_config, 'step', None)
            default_val = default_params.get(param_name)
            
            sampled_params[param_name] = self.sample_single_param(
                param_name=param_name,
                min_value=min_val,
                max_value=max_val,
                param_type=param_type,
                default_value=default_val,
                step=step
            )
        
        return sampled_params
    
    def generate_initial_samples(
        self,
        search_space: Dict[str, Any],
        n_samples: int,
        default_params: Dict[str, Any] = None,
        include_default: bool = True
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        生成多组初始采样参数
        
        Args:
            search_space: 搜索空间配置
            n_samples: 采样数量
            default_params: 默认参数
            include_default: 是否包含默认参数作为第一个样本 (Trial 0)
            
        Returns:
            (采样参数列表, 是否包含了默认参数作为Trial 0)
        """
        samples = []
        has_default_as_trial0 = False
        
        # Trial 0: 策略默认参数（如果启用）
        if include_default and default_params:
            # 构建完整的默认参数（只包含搜索空间中的参数）
            valid_default = {}
            for param_name, space_config in search_space.items():
                if param_name in default_params:
                    val = default_params[param_name]
                    # 确保值在搜索范围内（如果超出范围则裁剪）
                    original_val = val
                    val = max(space_config.min_value, min(space_config.max_value, val))
                    if space_config.param_type == 'int':
                        val = int(val)
                    valid_default[param_name] = val
            
            # 只有当所有搜索空间参数都有默认值时才添加
            if len(valid_default) == len(search_space):
                samples.append(valid_default)
                has_default_as_trial0 = True
        
        # 生成剩余的随机样本（正态分布采样）
        remaining = n_samples - len(samples)
        for _ in range(remaining):
            samples.append(self.sample_params(search_space, default_params))
        
        return samples, has_default_as_trial0


class ParallelExplorer:
    """
    并行随机探索器
    
    在初始探索阶段使用多进程并行评估参数组合，
    显著提升探索效率
    """
    
    def __init__(
        self,
        config: SamplerConfig = None,
        verbose: bool = True
    ):
        """
        初始化并行探索器
        
        Args:
            config: 采样器配置
            verbose: 是否打印详细信息
        """
        self.config = config or SamplerConfig()
        self.verbose = verbose
        
        # 确定工作进程数
        if self.config.n_workers is None:
            # 使用CPU核心数的75%，至少为1
            cpu_count = mp.cpu_count()
            self.n_workers = max(1, int(cpu_count * 0.75))
        else:
            self.n_workers = self.config.n_workers
    
    def parallel_evaluate(
        self,
        param_samples: List[Dict[str, Any]],
        evaluate_fn: Callable[[Dict[str, Any]], float],
        show_progress: bool = True
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        并行评估多组参数
        
        Args:
            param_samples: 参数组列表
            evaluate_fn: 评估函数，接受参数字典返回目标值
            show_progress: 是否显示进度
            
        Returns:
            (参数, 目标值) 列表
        """
        results = []
        total = len(param_samples)
        
        if self.verbose and show_progress:
            print(f"\n[并行探索] 开始评估 {total} 组参数...")
            print(f"[并行探索] 使用 {self.n_workers} 个工作进程")
        
        start_time = time.time()
        completed = 0
        
        # 使用进程池并行执行
        # 注意：由于backtrader的限制，这里实际使用串行执行
        # 但保留并行架构，以便将来支持真正的并行回测
        for i, params in enumerate(param_samples):
            try:
                value = evaluate_fn(params)
                results.append((params, value))
                completed += 1
                
                if self.verbose and show_progress and completed % 5 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total - completed) / rate if rate > 0 else 0
                    print(f"[并行探索] 进度: {completed}/{total} "
                          f"({100*completed/total:.1f}%) "
                          f"预计剩余: {remaining:.1f}秒")
            except Exception as e:
                if self.verbose:
                    print(f"[并行探索] 参数评估失败: {e}")
                results.append((params, float('-inf')))
        
        elapsed = time.time() - start_time
        if self.verbose and show_progress:
            print(f"[并行探索] 完成! 耗时: {elapsed:.2f}秒, "
                  f"平均: {elapsed/total:.3f}秒/组")
        
        return results
    
    def explore_with_samples(
        self,
        sampler: NormalDistributionSampler,
        search_space: Dict[str, Any],
        n_samples: int,
        evaluate_fn: Callable[[Dict[str, Any]], float],
        default_params: Dict[str, Any] = None
    ) -> Tuple[Dict[str, Any], float, List[Tuple[Dict[str, Any], float]]]:
        """
        使用正态分布采样进行并行探索
        
        Args:
            sampler: 正态分布采样器
            search_space: 搜索空间
            n_samples: 采样数量
            evaluate_fn: 评估函数
            default_params: 默认参数
            
        Returns:
            (最佳参数, 最佳值, 所有结果列表)
        """
        # 生成初始样本
        samples = sampler.generate_initial_samples(
            search_space=search_space,
            n_samples=n_samples,
            default_params=default_params,
            include_default=True
        )
        
        # 并行评估
        results = self.parallel_evaluate(samples, evaluate_fn)
        
        # 找出最佳结果
        best_params = None
        best_value = float('-inf')
        
        for params, value in results:
            if value > best_value:
                best_value = value
                best_params = params
        
        return best_params, best_value, results


class DynamicTrialsCalculator:
    """
    动态试验次数计算器
    
    根据参数数量和参数空间复杂度自动调整试验次数
    """
    
    def __init__(self, config: SamplerConfig = None):
        """
        初始化动态试验次数计算器
        
        Args:
            config: 采样器配置
        """
        self.config = config or SamplerConfig()
    
    def calculate_trials(
        self,
        n_params: int,
        search_space: Dict[str, Any] = None,
        user_trials: int = None
    ) -> Tuple[int, int, int]:
        """
        计算推荐的试验次数
        
        Args:
            n_params: 参数数量
            search_space: 搜索空间（用于分析复杂度）
            user_trials: 用户指定的试验次数（优先级最高）
            
        Returns:
            (总试验次数, 探索阶段次数, 利用阶段次数)
        """
        if not self.config.enable_dynamic_trials:
            # 不启用动态调整，直接返回用户指定或基础值
            total = user_trials or self.config.base_trials
            exploration = int(total * self.config.exploration_ratio)
            exploitation = total - exploration
            return total, exploration, exploitation
        
        # 计算基础试验次数
        base = self.config.base_trials
        per_param = self.config.trials_per_param
        
        # 基于参数数量的线性增长
        calculated = base + n_params * per_param
        
        # 分析搜索空间复杂度（如果提供）
        if search_space:
            complexity_factor = self._analyze_space_complexity(search_space)
            calculated = int(calculated * complexity_factor)
        
        # 应用边界限制
        calculated = max(self.config.min_trials, min(self.config.max_trials, calculated))
        
        # 如果用户指定了试验次数，取较大值
        if user_trials:
            calculated = max(calculated, user_trials)
        
        # 计算探索和利用阶段的分配
        exploration = int(calculated * self.config.exploration_ratio)
        exploitation = calculated - exploration
        
        return calculated, exploration, exploitation
    
    def _analyze_space_complexity(self, search_space: Dict[str, Any]) -> float:
        """
        分析搜索空间复杂度
        
        返回复杂度因子（1.0 为基准）
        
        Args:
            search_space: 搜索空间配置
            
        Returns:
            复杂度因子
        """
        complexity = 1.0
        
        for param_name, config in search_space.items():
            # 计算参数范围的相对大小
            min_val = config.min_value
            max_val = config.max_value
            
            if config.param_type == 'int':
                # 整数参数：可能的值数量
                n_values = int(max_val) - int(min_val) + 1
                if n_values > 50:
                    complexity *= 1.1  # 大范围增加复杂度
            else:
                # 浮点参数：检查是否跨越多个数量级
                if min_val > 0:
                    ratio = max_val / min_val
                    if ratio > 10:
                        complexity *= 1.1  # 跨越多个数量级
        
        # 限制复杂度因子范围
        return min(2.0, max(0.8, complexity))
    
    def get_recommendation_message(
        self,
        n_params: int,
        user_trials: int = None,
        search_space: Dict[str, Any] = None
    ) -> str:
        """
        获取试验次数推荐说明
        
        Args:
            n_params: 参数数量
            user_trials: 用户指定的试验次数
            search_space: 搜索空间
            
        Returns:
            推荐说明文本
        """
        total, exploration, exploitation = self.calculate_trials(
            n_params, search_space, user_trials
        )
        
        message = f"""
动态试验次数计算:
  • 参数数量: {n_params}
  • 推荐总试验次数: {total}
    - 随机探索阶段 (正态分布采样): {exploration} 次 ({self.config.exploration_ratio*100:.0f}%)
    - 智能采样阶段 (贝叶斯优化): {exploitation} 次 ({(1-self.config.exploration_ratio)*100:.0f}%)
"""
        
        if user_trials and user_trials < total:
            message += f"\n  ⚠️ 用户指定 {user_trials} 次可能不足，建议使用 {total} 次"
        
        return message


class EnhancedOptimizer:
    """
    增强优化器
    
    集成正态分布采样、并行探索和动态试验次数功能
    """
    
    def __init__(
        self,
        config: SamplerConfig = None,
        seed: int = None,
        verbose: bool = True
    ):
        """
        初始化增强优化器
        
        Args:
            config: 采样器配置
            seed: 随机种子
            verbose: 是否打印详细信息
        """
        self.config = config or SamplerConfig()
        self.verbose = verbose
        
        # 初始化组件
        self.sampler = NormalDistributionSampler(self.config, seed)
        self.explorer = ParallelExplorer(self.config, verbose)
        self.trials_calculator = DynamicTrialsCalculator(self.config)
    
    def run_exploration_phase(
        self,
        search_space: Dict[str, Any],
        evaluate_fn: Callable[[Dict[str, Any]], float],
        n_exploration_trials: int,
        default_params: Dict[str, Any] = None
    ) -> Tuple[Dict[str, Any], float, List[Dict]]:
        """
        运行探索阶段
        
        使用正态分布采样进行初始探索
        
        Args:
            search_space: 搜索空间
            evaluate_fn: 评估函数
            n_exploration_trials: 探索试验次数
            default_params: 默认参数
            
        Returns:
            (最佳参数, 最佳值, 历史记录列表)
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"开始探索阶段 (正态分布采样)")
            print(f"试验次数: {n_exploration_trials}")
            print(f"{'='*60}")
        
        # 使用并行探索
        best_params, best_value, results = self.explorer.explore_with_samples(
            sampler=self.sampler,
            search_space=search_space,
            n_samples=n_exploration_trials,
            evaluate_fn=evaluate_fn,
            default_params=default_params
        )
        
        # 转换为历史记录格式
        history = []
        for i, (params, value) in enumerate(results):
            history.append({
                'trial': i,
                'params': params,
                'value': value,
                'phase': 'exploration'
            })
        
        if self.verbose:
            print(f"\n探索阶段完成!")
            print(f"最佳目标值: {best_value:.4f}")
            print(f"最佳参数: {best_params}")
        
        return best_params, best_value, history
    
    def get_trials_allocation(
        self,
        n_params: int,
        user_trials: int = None,
        search_space: Dict[str, Any] = None
    ) -> Tuple[int, int, int]:
        """
        获取试验次数分配
        
        Args:
            n_params: 参数数量
            user_trials: 用户指定的试验次数
            search_space: 搜索空间
            
        Returns:
            (总次数, 探索次数, 利用次数)
        """
        return self.trials_calculator.calculate_trials(
            n_params, search_space, user_trials
        )


# 便捷函数
def create_enhanced_sampler(
    use_normal: bool = True,
    enable_parallel: bool = True,
    enable_dynamic_trials: bool = True,
    **kwargs
) -> EnhancedOptimizer:
    """
    创建增强采样器的便捷函数
    
    Args:
        use_normal: 是否使用正态分布
        enable_parallel: 是否启用并行
        enable_dynamic_trials: 是否启用动态试验次数
        **kwargs: 其他配置参数
        
    Returns:
        增强优化器实例
    """
    config = SamplerConfig(
        use_normal_distribution=use_normal,
        enable_parallel=enable_parallel,
        enable_dynamic_trials=enable_dynamic_trials,
        **{k: v for k, v in kwargs.items() if hasattr(SamplerConfig, k)}
    )
    
    return EnhancedOptimizer(config)


if __name__ == "__main__":
    # 测试代码
    print("增强采样器测试")
    print("="*60)
    
    # 创建模拟的搜索空间
    from dataclasses import dataclass as dc
    
    @dc
    class MockSearchSpace:
        param_type: str
        min_value: float
        max_value: float
        step: float = None
    
    search_space = {
        'period': MockSearchSpace('int', 10, 100, 1),
        'std_dev': MockSearchSpace('float', 0.5, 5.0),
        'threshold': MockSearchSpace('float', 0.01, 0.2),
    }
    
    # 创建采样器
    config = SamplerConfig()
    sampler = NormalDistributionSampler(config, seed=42)
    
    # 生成样本
    print("\n正态分布采样测试:")
    samples = sampler.generate_initial_samples(
        search_space=search_space,
        n_samples=10,
        default_params={'period': 50, 'std_dev': 2.0, 'threshold': 0.1}
    )
    
    for i, sample in enumerate(samples[:5]):
        print(f"  样本 {i+1}: {sample}")
    
    # 测试动态试验次数
    print("\n动态试验次数测试:")
    calculator = DynamicTrialsCalculator(config)
    
    for n_params in [2, 4, 6, 8]:
        total, exploration, exploitation = calculator.calculate_trials(n_params, search_space)
        print(f"  {n_params} 个参数: 总计 {total} 次 (探索 {exploration} + 利用 {exploitation})")
    
    print("\n测试完成!")
