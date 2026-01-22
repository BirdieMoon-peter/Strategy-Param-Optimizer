#!/bin/bash
# 通用优化器快速启动脚本

echo "================================"
echo "通用策略优化器 - 快速启动"
echo "================================"
echo ""

# 检查conda环境
if ! conda info --envs | grep -q "quant"; then
    echo "❌ 错误: 未找到quant环境"
    echo "   请先创建conda环境: conda create -n quant python=3.8"
    exit 1
fi

echo "✓ 找到conda环境: quant"
echo ""

# 激活环境
echo "激活conda环境..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate quant

echo "✓ 环境激活成功"
echo ""

# 显示菜单
echo "请选择操作:"
echo "  1. 运行测试（验证安装）"
echo "  2. 运行示例（交互式）"
echo "  3. 基本优化（需要提供数据和策略文件）"
echo "  4. 查看帮助"
echo "  5. 退出"
echo ""

read -p "请输入选项 (1-5): " choice

case $choice in
    1)
        echo ""
        echo "运行测试套件..."
        python test_universal_optimizer.py
        ;;
    2)
        echo ""
        echo "启动交互式示例..."
        python 使用示例.py
        ;;
    3)
        echo ""
        read -p "数据文件路径: " data_path
        read -p "策略文件路径: " strategy_path
        read -p "优化目标 (默认: sharpe_ratio): " objective
        objective=${objective:-sharpe_ratio}
        read -p "试验次数 (默认: 50): " trials
        trials=${trials:-50}
        read -p "是否使用LLM? (y/N): " use_llm
        
        if [[ $use_llm =~ ^[Yy]$ ]]; then
            read -p "LLM类型 (openai/ollama): " llm_type
            read -p "模型名称: " model_name
            read -p "API密钥 (Ollama留空): " api_key
            
            python run_universal_optimizer.py \
                --data "$data_path" \
                --strategy "$strategy_path" \
                --objective "$objective" \
                --trials "$trials" \
                --use-llm \
                --llm-type "$llm_type" \
                --llm-model "$model_name" \
                --api-key "$api_key"
        else
            python run_universal_optimizer.py \
                --data "$data_path" \
                --strategy "$strategy_path" \
                --objective "$objective" \
                --trials "$trials"
        fi
        ;;
    4)
        echo ""
        echo "显示帮助信息..."
        python run_universal_optimizer.py --help
        echo ""
        echo "详细文档请查看:"
        echo "  - 快速开始.md"
        echo "  - UNIVERSAL_OPTIMIZER_GUIDE.md"
        echo "  - README_UNIVERSAL.md"
        ;;
    5)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "================================"
echo "完成"
echo "================================"
