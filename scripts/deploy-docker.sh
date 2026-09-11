#!/bin/bash
# ============================================
# 电商智能体 — Docker 一键部署脚本
# 在阿里云服务器上执行
# 用法: bash scripts/deploy-docker.sh
# ============================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  电商智能体 Docker 部署脚本${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# 项目目录
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
echo -e "${GREEN}[INFO]${NC} 项目目录: $PROJECT_DIR"

# ============================================
# Step 1: 检查 Docker 是否已安装
# ============================================
echo ""
echo -e "${YELLOW}=== Step 1: 检查 Docker 环境 ===${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}[INSTALL]${NC} Docker 未安装，开始安装（阿里云镜像源）..."
    apt update
    curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun
    systemctl enable docker
    echo -e "${GREEN}[OK]${NC} Docker 安装完成"
else
    echo -e "${GREEN}[OK]${NC} Docker 已安装: $(docker --version)"
fi

if ! docker compose version &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Docker Compose 不可用，请检查安装"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Docker Compose: $(docker compose version --short)"

# ============================================
# Step 2: 配置阿里云镜像加速器
# ============================================
echo ""
echo -e "${YELLOW}=== Step 2: 配置 Docker 镜像加速器 ===${NC}"

# 检查是否已配置加速器
if [ -f /etc/docker/daemon.json ]; then
    echo -e "${GREEN}[OK]${NC} 已存在 daemon.json 配置"
    echo "   当前内容:"
    cat /etc/docker/daemon.json
    echo ""
    read -p "是否覆盖更新镜像加速配置? (y/N): " OVERWRITE
    if [[ "$OVERWRITE" != "y" && "$OVERWRITE" != "Y" ]]; then
        echo -e "${YELLOW}[SKIP]${NC} 保留现有配置"
    else
        CONFIGURE_MIRROR=true
    fi
else
    CONFIGURE_MIRROR=true
fi

if [ "$CONFIGURE_MIRROR" = true ]; then
    echo ""
    echo "请输入阿里云个人加速器地址（从 https://cr.console.aliyun.com/cn-hangzhou/instances/mirrors 获取）"
    echo "格式如: https://a1b2c3d4.mirror.aliyuncs.com"
    echo "直接回车则使用公共加速器（速度稍慢）"
    read -p "加速器地址: " ACCELERATOR_URL

    mkdir -p /etc/docker

    if [ -z "$ACCELERATOR_URL" ]; then
        # 使用公共加速器
        cat > /etc/docker/daemon.json << 'EOF'
{
  "registry-mirrors": [
    "https://registry.cn-hangzhou.aliyuncs.com",
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com"
  ],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "3"
  }
}
EOF
    else
        # 使用个人加速器 + 公共 fallback
        cat > /etc/docker/daemon.json << EOF
{
  "registry-mirrors": [
    "${ACCELERATOR_URL}",
    "https://registry.cn-hangzhou.aliyuncs.com",
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com"
  ],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "3"
  }
}
EOF
    fi

    systemctl daemon-reload
    systemctl restart docker
    echo -e "${GREEN}[OK]${NC} 镜像加速器配置完成"
    docker info 2>/dev/null | grep -A 5 "Registry Mirrors" || true
fi

# ============================================
# Step 3: 检查 .env.docker 配置（compose 实际读取的 env_file）
# ============================================
echo ""
echo -e "${YELLOW}=== Step 3: 检查 .env.docker 配置 ===${NC}"

ENV_FILE=".env.docker"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}[ERROR]${NC} $ENV_FILE 文件不存在！"
    echo "请先创建 $ENV_FILE 文件（包含 DeepSeek key / Neo4j / MySQL 配置）"
    exit 1
fi

# 检查 Docker 环境下的关键配置
NEO4J_URI=$(grep NEO4J_URI "$ENV_FILE" | cut -d'=' -f2)
MYSQL_HOST=$(grep MYSQL_HOST "$ENV_FILE" | cut -d'=' -f2)

echo -e "${GREEN}[OK]${NC} $ENV_FILE 文件存在"

if echo "$NEO4J_URI" | grep -q "localhost"; then
    echo -e "${YELLOW}[WARN]${NC} NEO4J_URI 包含 localhost，Docker 环境下应改为容器名 neo4j"
    echo "  当前: $NEO4J_URI"
    echo "  建议: neo4j://neo4j:7687"
    read -p "是否自动修改? (y/N): " FIX_NEO4J
    if [[ "$FIX_NEO4J" == "y" || "$FIX_NEO4J" == "Y" ]]; then
        sed -i 's|NEO4J_URI=neo4j://localhost:7687|NEO4J_URI=neo4j://neo4j:7687|' "$ENV_FILE"
        echo -e "${GREEN}[OK]${NC} 已修改 NEO4J_URI 为 neo4j://neo4j:7687"
    fi
fi

if echo "$MYSQL_HOST" | grep -q "localhost"; then
    echo -e "${YELLOW}[WARN]${NC} MYSQL_HOST 包含 localhost，Docker 环境下应改为容器名 mysql"
    echo "  当前: $MYSQL_HOST"
    echo "  建议: mysql"
    read -p "是否自动修改? (y/N): " FIX_MYSQL
    if [[ "$FIX_MYSQL" == "y" || "$FIX_MYSQL" == "Y" ]]; then
        sed -i 's|MYSQL_HOST=localhost|MYSQL_HOST=mysql|' "$ENV_FILE"
        echo -e "${GREEN}[OK]${NC} 已修改 MYSQL_HOST 为 mysql"
    fi
fi

# ============================================
# Step 4: 检查端口冲突
# ============================================
echo ""
echo -e "${YELLOW}=== Step 4: 检查端口冲突 ===${NC}"

CONFLICT=false
# H3 修复：数据库端口（3306/7687/7474）已不再对外映射，仅检查对外服务端口
for PORT in 8002 8501; do
    if ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
        echo -e "${YELLOW}[WARN]${NC} 端口 $PORT 已被占用:"
        ss -tlnp 2>/dev/null | grep ":$PORT "
        CONFLICT=true
    fi
done

if [ "$CONFLICT" = true ]; then
    echo -e "${YELLOW}[WARN]${NC} 存在端口冲突，请先释放端口或修改 docker-compose.yml 端口映射"
    read -p "是否继续部署? (y/N): " CONTINUE
    if [[ "$CONTINUE" != "y" && "$CONTINUE" != "Y" ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}[OK]${NC} 对外端口均空闲 (8002/8501；数据库端口已不对外暴露)"
fi

# ============================================
# Step 5: 构建镜像
# ============================================
echo ""
echo -e "${YELLOW}=== Step 5: 构建镜像 ===${NC}"
echo -e "${CYAN}[BUILD]${NC} 开始构建（首次约 10-15 分钟，含 pip install）..."
echo "  - agent-api (FastAPI 后端)"
echo "  - streamlit (前端 UI)"
echo ""

docker compose build --progress=plain

echo -e "${GREEN}[OK]${NC} 镜像构建完成"

# ============================================
# Step 6: 启动服务
# ============================================
echo ""
echo -e "${YELLOW}=== Step 6: 启动全部服务 ===${NC}"

# 先拉取 Neo4j 和 MySQL 镜像（利用加速器）
echo -e "${CYAN}[PULL]${NC} 拉取 Neo4j 和 MySQL 镜像（阿里云加速器）..."
docker compose pull neo4j mysql 2>/dev/null || true

# 启动
docker compose up -d

echo ""
echo -e "${GREEN}[OK]${NC} 服务已启动"
echo ""
docker compose ps

# ============================================
# Step 7: 等待服务就绪
# ============================================
echo ""
echo -e "${YELLOW}=== Step 7: 等待服务就绪 ===${NC}"

echo -e "${CYAN}[WAIT]${NC} 等待 Neo4j 启动..."
for i in $(seq 1 30); do
    if docker compose exec -T neo4j cypher-shell -u neo4j -p "$(grep NEO4J_PASSWORD .env | cut -d'=' -f2)" "RETURN 1;" &>/dev/null; then
        echo -e "${GREEN}[OK]${NC} Neo4j 就绪"
        break
    fi
    echo -n "."
    sleep 2
done
echo ""

echo -e "${CYAN}[WAIT]${NC} 等待 MySQL 启动..."
for i in $(seq 1 30); do
    if docker compose exec -T mysql mysqladmin ping -h localhost -u root -p"$(grep MYSQL_PASSWORD .env | cut -d'=' -f2)" &>/dev/null; then
        echo -e "${GREEN}[OK]${NC} MySQL 就绪"
        break
    fi
    echo -n "."
    sleep 2
done
echo ""

echo -e "${CYAN}[WAIT]${NC} 等待 API 启动..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8002/api/health &>/dev/null; then
        echo -e "${GREEN}[OK]${NC} API 就绪"
        break
    fi
    echo -n "."
    sleep 3
done
echo ""

# ============================================
# Step 8: 导入 Neo4j 数据
# ============================================
echo ""
echo -e "${YELLOW}=== Step 8: 导入 Neo4j 数据 ===${NC}"

NEO4J_PWD=$(grep NEO4J_PASSWORD .env | cut -d'=' -f2)

# 检查是否已有数据
NODE_COUNT=$(docker compose exec -T neo4j cypher-shell -u neo4j -p "$NEO4J_PWD" "MATCH (n) RETURN count(n) AS count;" 2>/dev/null | grep -oP '\d+' | tail -1)

if [ -z "$NODE_COUNT" ] || [ "$NODE_COUNT" = "0" ]; then
    echo -e "${CYAN}[IMPORT]${NC} Neo4j 无数据，开始导入..."

    if [ -f data/processed/neo4j_import.cypher ]; then
        # 复制 cypher 文件到容器并执行
        NEO4J_CONTAINER=$(docker compose ps -q neo4j)
        docker cp data/processed/neo4j_import.cypher "$NEO4J_CONTAINER:/tmp/import.cypher"
        docker compose exec -T neo4j cypher-shell -u neo4j -p "$NEO4J_PWD" --file /tmp/import.cypher
        echo -e "${GREEN}[OK]${NC} Neo4j 数据导入完成"
    else
        echo -e "${YELLOW}[WARN]${NC} data/processed/neo4j_import.cypher 不存在，跳过导入"
    fi
else
    echo -e "${GREEN}[OK]${NC} Neo4j 已有数据 ($NODE_COUNT 个节点)，跳过导入"
fi

# ============================================
# Step 9: 最终验证
# ============================================
echo ""
echo -e "${YELLOW}=== Step 9: 最终验证 ===${NC}"

echo ""
echo -e "${CYAN}[CHECK]${NC} 健康检查:"
curl -s http://localhost:8002/api/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:8002/api/health

echo ""
echo -e "${CYAN}[CHECK]${NC} Agent 列表:"
curl -s http://localhost:8002/api/agents | python3 -m json.tool 2>/dev/null || curl -s http://localhost:8002/api/agents

echo ""
echo -e "${CYAN}[CHECK]${NC} Neo4j 数据统计:"
docker compose exec -T neo4j cypher-shell -u neo4j -p "$NEO4J_PWD" \
    "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC LIMIT 10;"

# ============================================
# 完成
# ============================================
echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${GREEN}  部署完成!${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo "访问地址:"
echo "  API 文档:    http://$(hostname -I | awk '{print $1}'):8002/docs"
echo "  前端 UI:     http://$(hostname -I | awk '{print $1}'):8501"
echo "  健康检查:    http://$(hostname -I | awk '{print $1}'):8002/api/health"
echo ""
echo "（Neo4j/MySQL 仅容器网络内可达，不对外暴露；如需运维访问请使用: docker compose exec）"
echo ""
echo "常用命令:"
echo "  查看状态:  docker compose ps"
echo "  查看日志:  docker compose logs -f agent-api"
echo "  重启服务:  docker compose restart"
echo "  停止服务:  docker compose down"
echo "  访问数据库: docker compose exec -T mysql mysql -u root -p<密码> gmall"
echo ""
echo -e "${YELLOW}提醒: 请在阿里云安全组开放端口 8002 和 8501${NC}"
echo -e "${YELLOW}       访问 /api/* 接口需在 .env.docker 中设置 API_ACCESS_KEY 并携带 X-API-Key 头${NC}"
echo -e "${YELLOW}       控制台: https://ecs.console.aliyun.com${NC}"
