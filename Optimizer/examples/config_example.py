# -*- coding: utf-8 -*-
"""
配置示例文件
展示如何配置通用优化器的各种参数
"""

from universal_llm_client import UniversalLLMConfig, PRESET_CONFIGS

# ============================================================================
# LLM配置示例
# ============================================================================

# 示例1: OpenAI GPT-4
openai_gpt4_config = UniversalLLMConfig(
    api_type="openai",
    base_url="https://api.openai.com/v1",
    model_name="gpt-4",
    api_key="sk-your-api-key-here",  # 替换为实际API密钥
    temperature=0.7,
    max_tokens=4096,
    timeout=120
)

# 示例2: OpenAI GPT-3.5 (更快，更便宜)
openai_gpt35_config = UniversalLLMConfig(
    api_type="openai",
    base_url="https://api.openai.com/v1",
    model_name="gpt-3.5-turbo",
    api_key="sk-your-api-key-here",
    temperature=0.7,
    max_tokens=4096,
    timeout=120
)

# 示例3: 本地Ollama - Qwen
ollama_qwen_config = UniversalLLMConfig(
    api_type="ollama",
    base_url="http://localhost:11434",
    model_name="qwen",
    api_key="",  # Ollama不需要API密钥
    temperature=0.7,
    max_tokens=4096,
    timeout=120
)

# 示例4: 本地Ollama - Xuanyuan
ollama_xuanyuan_config = UniversalLLMConfig(
    api_type="ollama",
    base_url="http://localhost:11434",
    model_name="xuanyuan",
    api_key="",
    temperature=0.7,
    max_tokens=4096,
    timeout=120
)

# 示例5: 自定义API
custom_api_config = UniversalLLMConfig(
    api_type="custom",
    base_url="https://your-custom-api.com/v1/chat/completions",
    model_name="your-model-name",
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=4096,
    timeout=120
)

# 示例6: 使用预设配置
preset_config = PRESET_CONFIGS["openai-gpt4"].copy()
preset_config.api_key = "sk-your-key"

# ============================================================================
# 优化器配置示例
# ============================================================================

# 基本配置（不使用LLM）
BASIC_CONFIG = {
    "data_path": "data/BTC.csv",
    "strategy_path": "strategies/my_strategy.py",
    "objective": "sharpe_ratio",
    "use_llm": False,
    "output_dir": "./results",
    "verbose": True
}

# 使用LLM的配置
LLM_CONFIG = {
    "data_path": "data/BTC.csv",
    "strategy_path": "strategies/my_strategy.py",
    "objective": "sharpe_ratio",
    "use_llm": True,
    "llm_config": openai_gpt4_config,  # 或其他LLM配置
    "output_dir": "./results",
    "verbose": True
}

# 高级配置（批量优化）
BATCH_CONFIG = {
    "data_path": "data/BTC.csv",
    "strategy_path": "strategies/my_strategy.py",
    "use_llm": True,
    "llm_config": openai_gpt4_config,
    "output_dir": "./results",
    "verbose": True
}

# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print("配置示例文件")
    print("\n可用的LLM配置:")
    print("  - openai_gpt4_config")
    print("  - openai_gpt35_config")
    print("  - ollama_qwen_config")
    print("  - ollama_xuanyuan_config")
    print("  - custom_api_config")
    
    print("\n可用的优化器配置模板:")
    print("  - BASIC_CONFIG (不使用LLM)")
    print("  - LLM_CONFIG (使用LLM)")
    print("  - BATCH_CONFIG (批量优化)")
    
    print("\n使用方法:")
    print("""
from config_example import openai_gpt4_config, LLM_CONFIG
from universal_optimizer import UniversalOptimizer

# 创建优化器
optimizer = UniversalOptimizer(**LLM_CONFIG)

# 执行优化
result = optimizer.optimize(n_trials=50)
""")
