#!/usr/bin/env python3
"""
AI论文RAG系统主程序
"""

import os
import sys
import yaml
import torch
from pathlib import Path

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查MPS
    if torch.backends.mps.is_available():
        print("✅ Apple Metal Performance Shaders 可用")
    else:
        print("⚠️ MPS不可用，将使用CPU")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"🐍 Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查必要目录
    directories = ["data/raw_papers", "vector_db", "logs"]
    for dir_path in directories:
        if Path(dir_path).exists():
            print(f"✅ 目录存在: {dir_path}")
        else:
            print(f"❌ 目录缺失: {dir_path}")
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✅ 已创建: {dir_path}")

def load_config():
    """加载配置"""
    with open("config/config.yaml", 'r') as f:
        return yaml.safe_load(f)

def main():
    """主函数"""
    print("🚀 启动AI论文RAG系统...")
    print("="*50)
    
    # 检查环境
    check_environment()
    
    # 加载配置
    config = load_config()
    print("✅ 配置加载完成")
    
    print("\n系统准备就绪！")
    print("下一步：运行 'python simple_demo.py' 进行测试")

if __name__ == "__main__":
    main()
