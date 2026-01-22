# -*- coding: utf-8 -*-
"""
通用LLM客户端
支持多种API接入方式（OpenAI、Ollama、自定义API）
"""

import json
import requests
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class UniversalLLMConfig:
    """通用LLM配置"""
    api_type: str = "openai"  # 'openai', 'ollama', 'custom'
    api_key: str = ""
    base_url: str = ""
    model_name: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120


class BaseLLMClient(ABC):
    """LLM客户端基类"""
    
    def __init__(self, config: UniversalLLMConfig):
        self.config = config
    
    @abstractmethod
    def call_llm(self, messages: List[Dict[str, str]]) -> str:
        """调用LLM接口"""
        pass
    
    def check_connection(self) -> bool:
        """检查连接状态"""
        try:
            response = self.call_llm([{"role": "user", "content": "test"}])
            return len(response) > 0
        except:
            return False


class OpenAIClient(BaseLLMClient):
    """OpenAI API客户端"""
    
    def call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        调用OpenAI API
        
        Args:
            messages: 消息列表
            
        Returns:
            模型回复
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        try:
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"[LLM错误] OpenAI API请求失败: {e}")
            return ""


class OllamaClient(BaseLLMClient):
    """Ollama本地客户端"""
    
    def call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        调用Ollama API
        
        Args:
            messages: 消息列表
            
        Returns:
            模型回复
        """
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
                f"{self.config.base_url}/api/chat",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            print(f"[LLM错误] Ollama请求失败: {e}")
            return ""


class CustomAPIClient(BaseLLMClient):
    """自定义API客户端"""
    
    def call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        调用自定义API
        
        Args:
            messages: 消息列表
            
        Returns:
            模型回复
        """
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        try:
            response = requests.post(
                self.config.base_url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            # 尝试多种可能的返回格式
            if 'choices' in result:
                return result['choices'][0]['message']['content']
            elif 'message' in result:
                return result['message'].get('content', '')
            elif 'content' in result:
                return result['content']
            else:
                return str(result)
                
        except requests.exceptions.RequestException as e:
            print(f"[LLM错误] 自定义API请求失败: {e}")
            return ""


class UniversalLLMClient:
    """
    通用LLM客户端
    根据配置自动选择合适的客户端
    """
    
    # 内置的system prompt
    STRATEGY_ANALYSIS_PROMPT = """你是一位资深的量化交易策略专家和机器学习工程师。
你的任务是分析量化交易策略的超参数，并为贝叶斯优化推荐合适的搜索空间。

请根据策略类型和参数含义，给出合理的：
1. 参数搜索范围（min, max）
2. 参数分布类型（uniform, log_uniform, int_uniform）
3. 参数之间的约束关系
4. 优先级建议（哪些参数更重要）

你的回复必须是有效的JSON格式。"""

    OPTIMIZATION_HISTORY_PROMPT = """你是一位资深的量化交易策略专家和贝叶斯优化专家。
你的任务是根据历史优化结果，动态调整搜索空间以提高优化效率。

分析要点：
1. 识别表现好的参数区间，建议收窄搜索范围
2. 识别表现差的参数区间，建议排除或扩展
3. 发现参数之间的相关性
4. 识别对结果影响较大的关键参数

你的回复必须是有效的JSON格式。"""

    RESULT_EXPLANATION_PROMPT = """你是一位资深的量化交易分析师。
你的任务是解释优化结果，帮助用户理解：
1. 最优参数为什么有效
2. 策略在不同市场环境下的表现
3. 潜在的风险点
4. 实战应用建议

你的回复必须是有效的JSON格式，包含清晰的中文解释。"""
    
    def __init__(self, config: UniversalLLMConfig):
        """
        初始化通用LLM客户端
        
        Args:
            config: LLM配置
        """
        self.config = config
        
        # 根据API类型创建对应的客户端
        if config.api_type == "openai":
            self.client = OpenAIClient(config)
        elif config.api_type == "ollama":
            self.client = OllamaClient(config)
        elif config.api_type == "custom":
            self.client = CustomAPIClient(config)
        else:
            raise ValueError(f"不支持的API类型: {config.api_type}")
    
    def analyze_strategy_params(
        self,
        strategy_info: Dict,
        custom_prompt: str = None
    ) -> Dict:
        """
        分析策略参数并推荐搜索空间
        
        Args:
            strategy_info: 策略信息
            custom_prompt: 自定义提示词（可选）
            
        Returns:
            推荐的搜索空间配置
        """
        system_prompt = custom_prompt or self.STRATEGY_ANALYSIS_PROMPT
        
        prompt = f"""请分析以下量化交易策略的参数，并推荐贝叶斯优化的搜索空间：

策略名称: {strategy_info.get('class_name', 'N/A')}
策略描述: {strategy_info.get('description', 'N/A')}

参数列表:
"""
        for param in strategy_info.get('params', []):
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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.call_llm(messages)
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        return {}
    
    def explain_optimization_result(
        self,
        strategy_name: str,
        best_params: Dict,
        backtest_result: Dict,
        custom_prompt: str = None
    ) -> Dict:
        """
        解释优化结果
        
        Args:
            strategy_name: 策略名称
            best_params: 最优参数
            backtest_result: 回测结果
            custom_prompt: 自定义提示词（可选）
            
        Returns:
            包含解释的字典
        """
        system_prompt = custom_prompt or self.RESULT_EXPLANATION_PROMPT
        
        prompt = f"""请解释以下策略优化结果：

策略名称: {strategy_name}

最优参数:
{json.dumps(best_params, indent=2, ensure_ascii=False)}

回测性能:
{json.dumps(backtest_result, indent=2, ensure_ascii=False)}

请以JSON格式返回解释，格式如下:
{{
    "parameter_explanation": "参数解释（为什么这组参数有效）",
    "performance_analysis": "性能分析（策略表现如何）",
    "risk_assessment": "风险评估（潜在风险点）",
    "practical_suggestions": "实战建议（如何应用）",
    "key_insights": ["关键洞察1", "关键洞察2", ...]
}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.call_llm(messages)
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        return {
            "parameter_explanation": "参数优化完成",
            "performance_analysis": f"夏普比率: {backtest_result.get('sharpe_ratio', 'N/A')}",
            "risk_assessment": "请注意历史回测不代表未来表现",
            "practical_suggestions": "建议进行样本外测试",
            "key_insights": []
        }
    
    def check_connection(self) -> bool:
        """检查LLM连接"""
        return self.client.check_connection()


def create_llm_client(config: UniversalLLMConfig) -> UniversalLLMClient:
    """创建LLM客户端的工厂函数"""
    return UniversalLLMClient(config)


# 预设配置示例
PRESET_CONFIGS = {
    "openai-gpt4": UniversalLLMConfig(
        api_type="openai",
        base_url="https://api.openai.com/v1",
        model_name="gpt-4",
        api_key=""  # 需要用户填写
    ),
    "openai-gpt35": UniversalLLMConfig(
        api_type="openai",
        base_url="https://api.openai.com/v1",
        model_name="gpt-3.5-turbo",
        api_key=""
    ),
    "ollama-xuanyuan": UniversalLLMConfig(
        api_type="ollama",
        base_url="http://localhost:11434",
        model_name="xuanyuan",
        api_key=""
    ),
    "ollama-qwen": UniversalLLMConfig(
        api_type="ollama",
        base_url="http://localhost:11434",
        model_name="qwen",
        api_key=""
    ),
}


if __name__ == "__main__":
    # 测试代码
    print("测试通用LLM客户端\n")
    
    # 测试Ollama
    print("1. 测试Ollama客户端...")
    config = PRESET_CONFIGS["ollama-qwen"]
    client = UniversalLLMClient(config)
    
    if client.check_connection():
        print("✓ Ollama连接成功")
    else:
        print("✗ Ollama连接失败")
    
    print("\n可用预设配置:")
    for name in PRESET_CONFIGS.keys():
        print(f"  - {name}")
