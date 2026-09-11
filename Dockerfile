# ============================================================
# 电商领域智能体 API 服务镜像（公共源可构建版）
#
# 默认全部使用公共源：
#   - PyTorch CPU wheel: https://download.pytorch.org/whl/cpu
#   - Python 依赖:       https://pypi.org/simple
#
# 国内用户可通过 --build-arg 覆盖为镜像源，例如：
#   docker build \
#     --build-arg TORCH_INDEX_URL=https://mirrors.aliyun.com/pytorch-wheels/cpu \
#     --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ .
# ============================================================
FROM python:3.11-slim

LABEL maintainer="ecommerce-agent"
LABEL description="电商领域智能体 API 服务"

WORKDIR /app

# 创建非 root 运行用户（M3 安全加固：避免容器内以 root 运行）
RUN useradd --create-home --uid 10001 appuser

# 构建参数：默认公共源，可通过 --build-arg 覆盖
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PIP_INDEX_URL=https://pypi.org/simple

# 系统依赖（Debian 官方源）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 缓存层
COPY requirements.txt .

# Step 1: 安装 CPU-only torch（避免下载 1.5GB+ 的 CUDA 依赖）
RUN pip install --no-cache-dir --timeout 60 --retries 10 \
    torch \
    --index-url "$TORCH_INDEX_URL"

# Step 2: 安装其余依赖（torch 已安装，pip 会自动跳过）
RUN pip install --no-cache-dir --timeout 60 --retries 10 \
    -r requirements.txt \
    -i "$PIP_INDEX_URL"

# 复制项目文件
COPY . .

# 创建数据目录并切换到非 root 用户（M3 加固：容器内不以 root 运行，
# 即使发生反序列化等 RCE，攻击者也拿不到 root 权限）
RUN mkdir -p data/faiss_index data/cache data/memory_store models \
    && chown -R appuser:appuser /app

USER appuser

# 暴露端口
EXPOSE 8002

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8002/api/health || exit 1

# 启动命令
# 单 worker：短期记忆/限流/缓存均为进程内状态，多 worker 会导致会话分裂
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
