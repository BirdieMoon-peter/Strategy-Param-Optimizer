# -*- coding: utf-8 -*-
"""
报告生成模块
使用LLM生成自然语言分析报告
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import REPORTS_DIR, STRATEGY_PARAMS, OPTIMIZATION_OBJECTIVES
from llm_client import LLMClient, get_llm_client
from bayesian_optimizer import OptimizationResult
from backtest_engine import BacktestResult


class ReportGenerator:
    """
    报告生成器
    将技术性优化结果转化为易于理解的自然语言报告
    """
    
    def __init__(self, llm_client: LLMClient = None, use_llm: bool = True):
        """
        初始化报告生成器
        
        Args:
            llm_client: LLM客户端
            use_llm: 是否使用LLM生成报告
        """
        self.llm_client = llm_client or get_llm_client()
        self.use_llm = use_llm and self.llm_client.check_connection()
        
        os.makedirs(REPORTS_DIR, exist_ok=True)
    
    def generate_optimization_report(
        self,
        strategy_name: str,
        results: Dict[str, OptimizationResult],
        asset_name: str = None,
        optimization_history: Dict = None
    ) -> str:
        """生成优化报告(强制模板化渲染)。"""
        # 汇总最优参数与回测结果
        best_params: Dict[str, Dict] = {}
        backtest_results: Dict[str, Dict] = {}

        for objective, result in results.items():
            best_params[objective] = {
                "params": result.best_params,
                "value": result.best_value
            }
            if result.backtest_result:
                backtest_results[objective] = {
                    "total_return": result.backtest_result.total_return,
                    "annual_return": result.backtest_result.annual_return,
                    "max_drawdown": result.backtest_result.max_drawdown,
                    "sharpe_ratio": result.backtest_result.sharpe_ratio,
                    "trades_count": result.backtest_result.trades_count,
                    "win_rate": result.backtest_result.win_rate
                }

        # 首选：LLM输出结构化JSON，由我们渲染固定模板
        if self.use_llm:
            sections = self.llm_client.generate_report_sections(
                strategy_name,
                best_params,
                optimization_history or {},
                backtest_results
            )
            if sections:
                body = self._render_sections(sections, strategy_name, best_params, backtest_results)
                return self._format_report(body, strategy_name, asset_name)

        # 回退：使用本地模板
        return self._generate_template_report(
            strategy_name,
            best_params,
            backtest_results,
            asset_name
        )
    
    def _format_report(
        self,
        llm_report: str,
        strategy_name: str,
        asset_name: str = None
    ) -> str:
        """格式化LLM生成的报告"""
        header = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        量化策略优化分析报告                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  策略名称: {strategy_name:<66}║
║  资产标的: {(asset_name or '多资产'):<66}║
║  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<66}║
╚══════════════════════════════════════════════════════════════════════════════╝

"""
        return header + llm_report

    def _render_sections(
        self,
        sections: Dict[str, Any],
        strategy_name: str,
        best_params: Dict,
        backtest_results: Dict
    ) -> str:
        """将LLM返回的JSON片段渲染为固定模板正文。"""
        # 第一部分：执行摘要
        text = (
            "\n" +
            "="*80 + "\n" +
            "                                一、执行摘要\n" +
            "="*80 + "\n\n" +
            str(sections.get("executive_summary", "")) + "\n\n"
        )

        # 第二部分：优化过程分析
        text += (
            "="*80 + "\n" +
            "                              二、优化过程分析\n" +
            "="*80 + "\n\n" +
            str(sections.get("process_analysis", "")) + "\n\n"
        )

        # 第三部分：最优参数解读
        text += (
            "="*80 + "\n" +
            "                              三、最优参数解读\n" +
            "="*80 + "\n\n"
        )
        for objective, obj in best_params.items():
            text += f"【{objective}】\n"
            for p, v in obj.get("params", {}).items():
                text += f"  • {p}: {v}\n"
            text += "\n"
        param_expl = sections.get("parameters_explained", [])
        if param_expl:
            text += "关键参数说明:\n"
            for item in param_expl:
                text += f"  • {item}\n"
        text += "\n"

        # 第四部分：风险提示与建议
        text += (
            "="*80 + "\n" +
            "                              四、风险提示与建议\n" +
            "="*80 + "\n\n"
        )
        risks = sections.get("risks", [])
        if risks:
            text += "风险提示:\n"
            for r in risks:
                text += f"  • {r}\n"
            text += "\n"
        recs = sections.get("recommendations", [])
        if recs:
            text += "建议：\n"
            for r in recs:
                text += f"  • {r}\n"
            text += "\n"

        # 第五部分：结论
        text += (
            "="*80 + "\n" +
            "                                五、结论\n" +
            "="*80 + "\n\n" +
            str(sections.get("conclusion", "")) + "\n"
        )

        return text
    
    def _generate_template_report(
        self,
        strategy_name: str,
        best_params: Dict,
        backtest_results: Dict,
        asset_name: str = None
    ) -> str:
        """使用模板生成报告（当LLM不可用时）"""
        
        strategy_info = STRATEGY_PARAMS.get(strategy_name, {})
        strategy_desc = strategy_info.get('description', '量化交易策略')
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        量化策略优化分析报告                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  策略名称: {strategy_name:<66}║
║  资产标的: {(asset_name or '多资产'):<66}║
║  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<66}║
╚══════════════════════════════════════════════════════════════════════════════╝

================================================================================
                                一、执行摘要
================================================================================

本次优化针对 {strategy_name} 策略进行了多目标贝叶斯优化。

策略简介：{strategy_desc}

本次优化分别以以下三个目标进行参数搜索：
  • 最大化夏普比率：寻找风险调整后收益最优的参数组合
  • 最大化年化收益率：寻找收益最大化的参数组合
  • 最小化最大回撤：寻找风险最小化的参数组合

================================================================================
                              二、优化结果详情
================================================================================

"""
        # 各目标的结果
        for objective, obj_info in OPTIMIZATION_OBJECTIVES.items():
            if objective in best_params:
                params = best_params[objective]
                bt_result = backtest_results.get(objective, {})
                
                report += f"""
--------------------------------------------------------------------------------
【{obj_info.description}】
--------------------------------------------------------------------------------

最优参数配置：
"""
                for param_name, param_value in params['params'].items():
                    # 获取参数描述
                    param_desc = ""
                    for p in strategy_info.get('params', []):
                        if p.name == param_name:
                            param_desc = p.description
                            break
                    
                    report += f"  • {param_name}: {param_value}"
                    if param_desc:
                        report += f"  ({param_desc})"
                    report += "\n"
                
                report += f"""
回测性能指标：
  • 夏普比率: {bt_result.get('sharpe_ratio', 'N/A'):.4f}
  • 年化收益率: {bt_result.get('annual_return', 'N/A'):.2f}%
  • 最大回撤: {bt_result.get('max_drawdown', 'N/A'):.2f}%
  • 总收益率: {bt_result.get('total_return', 'N/A'):.2f}%
  • 交易次数: {bt_result.get('trades_count', 'N/A')}
  • 胜率: {bt_result.get('win_rate', 'N/A'):.1f}%

"""
        
        # 参数对比分析
        report += """
================================================================================
                              三、参数对比分析
================================================================================

下表展示了不同优化目标下的最优参数差异：

"""
        # 创建参数对比表
        all_params = set()
        for obj_data in best_params.values():
            all_params.update(obj_data['params'].keys())
        
        report += f"{'参数名':<20} {'夏普比率':<15} {'年化收益率':<15} {'最小回撤':<15}\n"
        report += "-" * 65 + "\n"
        
        for param in sorted(all_params):
            sharpe_val = best_params.get('sharpe_ratio', {}).get('params', {}).get(param, 'N/A')
            return_val = best_params.get('annual_return', {}).get('params', {}).get(param, 'N/A')
            dd_val = best_params.get('max_drawdown', {}).get('params', {}).get(param, 'N/A')
            report += f"{param:<20} {str(sharpe_val):<15} {str(return_val):<15} {str(dd_val):<15}\n"
        
        # 风险提示和建议
        report += """

================================================================================
                              四、风险提示与建议
================================================================================

⚠️  重要提示：

1. 过拟合风险
   历史回测结果是基于过去的市场数据进行的优化，参数可能过度拟合历史数据，
   在未来市场中的表现可能会有所不同。

2. 市场环境变化
   金融市场具有动态性，当前最优参数可能在市场结构发生变化时失效。
   建议定期重新评估和调整参数。

3. 样本外测试
   强烈建议在未参与优化的数据集上进行样本外测试，
   以验证参数的稳健性。

4. 风险管理
   无论选择哪组参数，都应该配合适当的仓位管理和风险控制措施。

💡 后续建议：

1. 在不同时间段和市场环境下进行稳健性测试
2. 考虑使用滚动优化方法，定期更新参数
3. 可以尝试结合多个目标的参数，寻找风险收益的平衡点
4. 在实盘交易前，先进行充分的模拟交易验证

================================================================================
                                五、结论
================================================================================

"""
        # 生成结论
        if 'sharpe_ratio' in backtest_results:
            sharpe = backtest_results['sharpe_ratio'].get('sharpe_ratio', 0)
            if sharpe > 1.5:
                conclusion = "优化后的策略表现出色，夏普比率超过1.5，具有较好的风险调整收益特征。"
            elif sharpe > 1.0:
                conclusion = "优化后的策略表现良好，夏普比率超过1.0，具有一定的投资价值。"
            elif sharpe > 0.5:
                conclusion = "优化后的策略表现一般，夏普比率在0.5-1.0之间，建议进一步优化或调整策略逻辑。"
            else:
                conclusion = "优化后的策略夏普比率较低，建议重新评估策略的有效性。"
        else:
            conclusion = "优化过程已完成，请根据具体需求选择合适的参数组合。"
        
        report += f"""
{conclusion}

建议根据自身的风险偏好，在三组最优参数中进行选择：
  • 追求高收益：选择年化收益率最大化的参数组合
  • 注重风险控制：选择最大回撤最小化的参数组合
  • 平衡风险收益：选择夏普比率最大化的参数组合

--------------------------------------------------------------------------------
                              报告生成完毕
--------------------------------------------------------------------------------
"""
        
        return report
    
    def save_report(
        self,
        report: str,
        strategy_name: str,
        asset_name: str = None
    ) -> str:
        """
        保存报告到文件
        
        Args:
            report: 报告文本
            strategy_name: 策略名称
            asset_name: 资产名称
            
        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if asset_name:
            filename = f"{strategy_name}_{asset_name}_report_{timestamp}.txt"
        else:
            filename = f"{strategy_name}_report_{timestamp}.txt"
        
        filepath = os.path.join(REPORTS_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"报告已保存至: {filepath}")
        return filepath
    
    def generate_summary_table(
        self,
        all_results: Dict[str, Dict[str, OptimizationResult]]
    ) -> pd.DataFrame:
        """
        生成多策略优化结果汇总表（旧格式，保留兼容性）
        
        Args:
            all_results: 所有策略的优化结果
            
        Returns:
            汇总DataFrame
        """
        rows = []
        
        for strategy_name, objectives in all_results.items():
            for objective, result in objectives.items():
                row = {
                    "策略": strategy_name,
                    "优化目标": objective,
                    "最佳值": result.best_value,
                    "试验次数": result.n_trials,
                    "优化时间(秒)": round(result.optimization_time, 1)
                }
                
                if result.backtest_result:
                    row["夏普比率"] = round(result.backtest_result.sharpe_ratio, 4)
                    row["年化收益率(%)"] = round(result.backtest_result.annual_return, 2)
                    row["最大回撤(%)"] = round(result.backtest_result.max_drawdown, 2)
                    row["胜率(%)"] = round(result.backtest_result.win_rate, 1)
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_detailed_csv(
        self,
        all_results: Dict[str, Dict[str, OptimizationResult]],
        objective_focus: str = "sharpe_ratio"
    ) -> pd.DataFrame:
        """
        生成详细的策略性能CSV表格，包含总体指标和每年的指标
        转置格式：行为指标，列为策略
        
        Args:
            all_results: 所有策略的优化结果
            objective_focus: 选择哪个优化目标的结果（默认为夏普比率）
            
        Returns:
            详细的DataFrame，转置后：行为指标，列为策略
        """
        # 先收集所有数据
        strategies_data = {}
        
        # 收集所有年份
        all_years = set()
        for strategy_name, objectives in all_results.items():
            if objective_focus in objectives:
                result = objectives[objective_focus]
                if result.backtest_result and result.backtest_result.yearly_returns:
                    all_years.update(result.backtest_result.yearly_returns.keys())
        
        all_years = sorted(all_years)
        
        # 收集每个策略的数据
        for strategy_name, objectives in all_results.items():
            if objective_focus not in objectives:
                continue
                
            result = objectives[objective_focus]
            bt_result = result.backtest_result
            
            if not bt_result:
                continue
            
            strategy_data = {
                "总夏普比率": round(bt_result.sharpe_ratio, 4),
                "总年化收益率(%)": round(bt_result.annual_return, 2),
                "总最大回撤(%)": round(bt_result.max_drawdown, 2)
            }
            
            # 添加每年的收益率
            if bt_result.yearly_returns:
                for year in all_years:
                    year_return = bt_result.yearly_returns.get(year, 0)
                    strategy_data[f"{year}年收益率(%)"] = round(year_return, 2)
            
            # 添加每年的回撤
            if bt_result.yearly_drawdowns:
                for year in all_years:
                    year_dd = bt_result.yearly_drawdowns.get(year, 0)
                    strategy_data[f"{year}年回撤(%)"] = round(year_dd, 2)
            
            # 添加每年的夏普比率
            if bt_result.yearly_sharpe:
                for year in all_years:
                    year_sharpe = bt_result.yearly_sharpe.get(year, 0)
                    strategy_data[f"{year}年夏普比率"] = round(year_sharpe, 4)
            
            strategies_data[strategy_name] = strategy_data
        
        # 构建转置的DataFrame：行为指标，列为策略
        # 先确定所有指标名称（按顺序）
        metric_names = ["总夏普比率", "总年化收益率(%)", "总最大回撤(%)"]
        
        # 添加每年的指标（按年份和类型排序）
        for year in all_years:
            metric_names.append(f"{year}年收益率(%)")
        for year in all_years:
            metric_names.append(f"{year}年回撤(%)")
        for year in all_years:
            metric_names.append(f"{year}年夏普比率")
        
        # 构建转置数据
        transposed_data = {}
        for metric in metric_names:
            transposed_data[metric] = {}
            for strategy_name in strategies_data.keys():
                transposed_data[metric][strategy_name] = strategies_data[strategy_name].get(metric, 0)
        
        # 创建DataFrame并转置（指标为行，策略为列）
        df = pd.DataFrame(transposed_data)
        df_transposed = df.T  # 转置：行为指标，列为策略
        
        # 确保索引名称正确（用于CSV的行名）
        df_transposed.index.name = '指标'
        
        return df_transposed
    
    def save_detailed_csv(
        self,
        all_results: Dict[str, Dict[str, OptimizationResult]],
        asset_name: str = None,
        objective_focus: str = "sharpe_ratio"
    ) -> str:
        """
        保存详细的CSV文件
        
        Args:
            all_results: 所有策略的优化结果
            asset_name: 资产名称
            objective_focus: 选择哪个优化目标的结果
            
        Returns:
            CSV文件路径
        """
        df = self.generate_detailed_csv(all_results, objective_focus)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if asset_name:
            filename = f"策略性能汇总_{asset_name}_{timestamp}.csv"
        else:
            filename = f"策略性能汇总_{timestamp}.csv"
        
        filepath = os.path.join(REPORTS_DIR, filename)
        # 保存时包含索引（指标名称作为第一列）
        df.to_csv(filepath, index=True, encoding='utf-8-sig')
        
        print(f"\n详细CSV已保存至: {filepath}")
        return filepath
    
    def print_quick_summary(
        self,
        strategy_name: str,
        results: Dict[str, OptimizationResult]
    ):
        """打印快速摘要"""
        print(f"\n{'='*60}")
        print(f"策略优化摘要: {strategy_name}")
        print(f"{'='*60}")
        
        for objective, result in results.items():
            obj_info = OPTIMIZATION_OBJECTIVES.get(objective, {})
            obj_desc = getattr(obj_info, 'description', objective)
            
            print(f"\n【{obj_desc}】")
            print(f"  最优值: {result.best_value:.4f}")
            print(f"  最优参数:")
            for param, value in result.best_params.items():
                print(f"    - {param}: {value}")
            
            if result.backtest_result:
                print(f"  回测结果:")
                print(f"    - 夏普比率: {result.backtest_result.sharpe_ratio:.4f}")
                print(f"    - 年化收益率: {result.backtest_result.annual_return:.2f}%")
                print(f"    - 最大回撤: {result.backtest_result.max_drawdown:.2f}%")
        
        print(f"\n{'='*60}")


if __name__ == "__main__":
    # 测试代码
    generator = ReportGenerator(use_llm=False)
    
    # 模拟数据
    from backtest_engine import BacktestResult
    
    mock_results = {
        "sharpe_ratio": OptimizationResult(
            objective="sharpe_ratio",
            best_params={"period": 40, "std_dev_upper": 1.8, "std_dev_lower": 2.2},
            best_value=1.45,
            backtest_result=BacktestResult(
                total_return=85.5,
                annual_return=28.5,
                max_drawdown=15.2,
                sharpe_ratio=1.45,
                final_value=185500,
                trades_count=42,
                win_rate=55.0,
                params={}
            ),
            n_trials=100,
            optimization_time=120.5
        ),
        "annual_return": OptimizationResult(
            objective="annual_return",
            best_params={"period": 25, "std_dev_upper": 1.5, "std_dev_lower": 1.5},
            best_value=35.2,
            backtest_result=BacktestResult(
                total_return=105.6,
                annual_return=35.2,
                max_drawdown=22.5,
                sharpe_ratio=1.15,
                final_value=205600,
                trades_count=68,
                win_rate=48.0,
                params={}
            ),
            n_trials=100,
            optimization_time=118.3
        ),
        "max_drawdown": OptimizationResult(
            objective="max_drawdown",
            best_params={"period": 50, "std_dev_upper": 2.5, "std_dev_lower": 2.5},
            best_value=8.5,
            backtest_result=BacktestResult(
                total_return=45.2,
                annual_return=15.1,
                max_drawdown=8.5,
                sharpe_ratio=1.05,
                final_value=145200,
                trades_count=25,
                win_rate=60.0,
                params={}
            ),
            n_trials=100,
            optimization_time=115.8
        )
    }
    
    # 生成报告
    report = generator.generate_optimization_report(
        "AberrationStrategy",
        mock_results,
        "BTC"
    )
    
    print(report)
    
    # 打印快速摘要
    generator.print_quick_summary("AberrationStrategy", mock_results)
