# Academic RAG System - Production Dockerfile
# ===========================================

FROM python:3.11-slim as base

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Builder stage - 安装Python依赖
# ============================================
FROM base as builder

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================
# Final stage - 运行应用
# ============================================
FROM base as final

# 从builder复制已安装的包
COPY --from=builder /root/.local /root/.local

# 确保Python可以找到用户安装的包
ENV PATH=/root/.local/bin:$PATH

# 复制应用代码
COPY . .

# 创建非root用户（安全最佳实践）
RUN useradd -m -u 1000 raguser && \
    chown -R raguser:raguser /app

# 切换到非root用户
USER raguser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# 启动命令
CMD ["python", "app/main.py"]
