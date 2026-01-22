# -*- coding: utf-8 -*-
"""
LLM客户端模块
封装与Ollama轩辕大模型的通信接口
"""

import json
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from config import LLMConfig, DEFAULT_LLM_CONFIG


class LLMClient:
    """
    Ollama轩辕大模型客户端
    用于分析策略参数、推荐搜索空间、动态调整优化过程
    """
    
    def __init__(self, config: LLMConfig = None):
        """
        初始化LLM客户端
        
        Args:
            config: LLM配置对象，默认使用DEFAULT_LLM_CONFIG
        """
        self.config = config or DEFAULT_LLM_CONFIG
        self.api_url = f"{self.config.base_url}/api/generate"
        self.chat_url = f"{self.config.base_url}/api/chat"
        
    def _make_request(self, prompt: str, system_prompt: str = None) -> str:
        """
        向Ollama发送请求
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            
        Returns:
            模型回复文本
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        }
        
        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            print(f"[LLM错误] 请求失败: {e}")
            return ""
    
    def analyze_strategy_params(self, strategy_info: Dict) -> Dict:
        """
        分析策略参数并推荐初始搜索空间
        
        Args:
            strategy_info: 策略信息字典，包含名称、描述、参数列表
            
        Returns:
            推荐的搜索空间配置
        """
        system_prompt = """你是一位资深的量化交易策略专家和机器学习工程师。
你的任务是分析量化交易策略的超参数，并为贝叶斯优化推荐合适的搜索空间。

请根据策略类型和参数含义，给出合理的：
1. 参数搜索范围（min, max）
2. 参数分布类型（uniform, log_uniform, int_uniform）
3. 参数之间的约束关系
4. 优先级建议（哪些参数更重要）

你的回复必须是有效的JSON格式。"""

        prompt = f"""请分析以下量化交易策略的参数，并推荐贝叶斯优化的搜索空间：

策略名称: {strategy_info['class_name']}
策略描述: {strategy_info['description']}

参数列表:
"""
        for param in strategy_info['params']:
            prompt += f"""
- 参数名: {param.name}
  类型: {param.param_type}
  默认值: {param.default_value}
  描述: {param.description}
  当前范围: [{param.min_value}, {param.max_value}]
  步长: {param.step}
"""
        
        prompt += """
请以JSON格式返回推荐的搜索空间配置，格式如下:
{
    "search_space": {
        "参数名": {
            "type": "int/float",
            "distribution": "uniform/log_uniform/int_uniform",
            "min": 最小值,
            "max": 最大值,
            "step": 步长（可选）,
            "priority": "high/medium/low",
            "reason": "推荐理由"
        }
    },
    "constraints": ["约束条件列表"],
    "recommendations": "优化建议"
}"""

        response = self._make_request(prompt, system_prompt)
        
        try:
            # 尝试从回复中提取JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # 如果解析失败，返回默认配置
        return self._get_default_search_space(strategy_info)
    
    def _get_default_search_space(self, strategy_info: Dict) -> Dict:
        """生成默认搜索空间配置"""
        search_space = {}
        for param in strategy_info['params']:
            search_space[param.name] = {
                "type": param.param_type,
                "distribution": "int_uniform" if param.param_type == "int" else "uniform",
                "min": param.min_value,
                "max": param.max_value,
                "step": param.step,
                "priority": "medium",
                "reason": "使用默认配置"
            }
        return {
            "search_space": search_space,
            "constraints": [],
            "recommendations": "使用默认搜索空间配置"
        }
    
    def analyze_optimization_history(
        self, 
        history: List[Dict],
        current_space: Dict,
        objective: str
    ) -> Dict:
        """
        分析优化历史并建议调整搜索空间
        
        Args:
            history: 优化历史记录，每条包含参数和目标值
            current_space: 当前搜索空间配置
            objective: 优化目标名称
            
        Returns:
            调整后的搜索空间配置
        """
        system_prompt = """你是一位资深的量化交易策略专家和贝叶斯优化专家。
你的任务是根据历史优化结果，动态调整搜索空间以提高优化效率。

分析要点：
1. 识别表现好的参数区间，建议收窄搜索范围
2. 识别表现差的参数区间，建议排除或扩展
3. 发现参数之间的相关性
4. 识别对结果影响较大的关键参数

你的回复必须是有效的JSON格式。"""

        # 统计历史数据
        n_trials = len(history)
        if n_trials == 0:
            return current_space
        
        # 找出最佳和最差的结果
        sorted_history = sorted(history, key=lambda x: x.get('value', 0), reverse=True)
        best_trials = sorted_history[:min(10, n_trials//5 + 1)]
        worst_trials = sorted_history[-min(10, n_trials//5 + 1):]
        
        prompt = f"""请分析以下贝叶斯优化的历史结果，并建议如何调整搜索空间：

优化目标: {objective}
总试验次数: {n_trials}

当前搜索空间:
{json.dumps(current_space, indent=2, ensure_ascii=False)}

最佳结果（前{len(best_trials)}个）:
"""
        for i, trial in enumerate(best_trials):
            prompt += f"\n{i+1}. 参数: {trial.get('params', {})}, 目标值: {trial.get('value', 'N/A'):.4f}"
        
        prompt += f"\n\n最差结果（后{len(worst_trials)}个）:"
        for i, trial in enumerate(worst_trials):
            prompt += f"\n{i+1}. 参数: {trial.get('params', {})}, 目标值: {trial.get('value', 'N/A'):.4f}"
        
        prompt += """

请以JSON格式返回调整后的搜索空间，格式如下:
{
    "adjusted_space": {
        "参数名": {
            "type": "int/float",
            "distribution": "uniform/log_uniform/int_uniform",
            "min": 新最小值,
            "max": 新最大值,
            "step": 步长,
            "adjustment": "收窄/扩展/不变",
            "reason": "调整理由"
        }
    },
    "key_findings": ["关键发现列表"],
    "next_recommendations": "下一轮优化建议"
}"""

        response = self._make_request(prompt, system_prompt)
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                if "adjusted_space" in result:
                    return result
        except json.JSONDecodeError:
            pass
        
        # 解析失败则返回当前配置
        return {"adjusted_space": current_space.get("search_space", current_space)}
    
    def generate_report_sections(
        self,
        strategy_name: str,
        best_params: Dict[str, Dict],
        optimization_history: Dict[str, List],
        backtest_results: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """
        生成结构化报告片段(JSON)，由上层进行模板化渲染。

        返回格式:
        {
          "executive_summary": str,
          "process_analysis": str,
          "parameters_explained": [str, ...],
          "risks": [str, ...],
          "recommendations": [str, ...],
          "conclusion": str
        }
        """
        system_prompt = """你是一位资深的量化分析师，需输出结构化JSON用于渲染模板。
必须严格返回有效JSON，且只返回JSON，不要输出问候语、解释或额外文本。
若信息不足，请基于已有数据给出保守、合理的总结，严禁让用户再提供信息。"""

        prompt = {
            "instruction": "根据以下数据生成报告各部分内容，中文输出，面向非技术读者可读。",
            "strategy_name": strategy_name,
            "best_params": best_params,
            "backtest_results": backtest_results,
            "guidelines": [
                "executive_summary: 2-4句，概述策略表现与亮点/不足",
                "process_analysis: 解释做了哪些优化与指标的含义",
                "parameters_explained: 列表，每项解释一个关键参数为何有效",
                "risks: 列表，涵盖回撤、过拟合、市场变局等",
                "recommendations: 列表，给出下一步可操作建议",
                "conclusion: 1-2句，给出清晰结论和应用建议"
            ]
        }

        response = self._make_request(json.dumps(prompt, ensure_ascii=False), system_prompt)

        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
                # 关键字段校验
                required = [
                    "executive_summary", "process_analysis", "parameters_explained",
                    "risks", "recommendations", "conclusion"
                ]
                if all(k in data for k in required):
                    # 进一步确保类型
                    if not isinstance(data.get("parameters_explained", []), list):
                        data["parameters_explained"] = [str(data.get("parameters_explained"))]
                    for key in ["risks", "recommendations"]:
                        if not isinstance(data.get(key, []), list):
                            data[key] = [str(data.get(key))]
                    return data
        except Exception:
            pass
        # 返回空字典，上层触发模板回退
        return {}
    
    def _generate_fallback_report(
        self,
        strategy_name: str,
        best_params: Dict,
        backtest_results: Dict
    ) -> str:
        """生成备用报告（当LLM不可用时）"""
        report = f"""
# {strategy_name} 优化报告

## 一、执行摘要

本次优化针对{strategy_name}策略进行了多目标贝叶斯优化，分别以最大化夏普比率、最大化年化收益率、最小化最大回撤为目标，寻找最优参数组合。

## 二、优化结果

"""
        for objective, result in best_params.items():
            report += f"""### {objective}
- 最优值: {result.get('value', 'N/A'):.4f}
- 参数配置:
"""
            for param, value in result.get('params', {}).items():
                report += f"  - {param}: {value}\n"
            report += "\n"
        
        report += """## 三、风险提示

1. 历史回测结果不代表未来表现
2. 参数过度拟合可能导致实盘表现下降
3. 建议进行样本外测试验证

## 四、后续建议

1. 在不同市场环境下进行稳健性测试
2. 考虑添加风险控制机制
3. 定期重新优化参数以适应市场变化
"""
        return report
    
    def check_connection(self) -> bool:
        """检查与Ollama的连接状态"""
        try:
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def list_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        try:
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
        except:
            pass
        return []


# 便捷函数
def get_llm_client(config: LLMConfig = None) -> LLMClient:
    """获取LLM客户端实例"""
    return LLMClient(config)


if __name__ == "__main__":
    # 测试代码
    client = LLMClient()
    
    print("检查Ollama连接...")
    if client.check_connection():
        print("✓ 连接成功")
        models = client.list_available_models()
        print(f"可用模型: {models}")
    else:
        print("✗ 连接失败，请确保Ollama正在运行")
