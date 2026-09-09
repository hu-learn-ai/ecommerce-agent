"""
构建商品 FAISS 向量索引

支持两种数据源:
1. 本地 JSON 文件 (data/processed/products_for_faiss.json) — 默认，无需 Neo4j
2. Neo4j 图数据库 — 使用 --neo4j 参数启用

使用方法:
    python scripts/build_faiss_index.py             # 从本地 JSON 构建
    python scripts/build_faiss_index.py --neo4j      # 从 Neo4j 构建

输出:
    data/faiss_index/products.index  — FAISS 索引文件
    data/faiss_index/product_ids.npy — 商品 ID 映射
"""

import json
import os
import sys

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from config.settings import settings


def load_products_from_json():
    """从本地 JSON 文件加载商品数据"""
    json_path = os.path.join(PROJECT_ROOT, "data", "processed", "products_for_faiss.json")
    if not os.path.exists(json_path):
        print(f"[FAISS Builder] 本地 JSON 未找到: {json_path}")
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        products = json.load(f)
    print(f"[FAISS Builder] 从本地 JSON 加载 {len(products)} 个商品")
    return products


def load_products_from_neo4j():
    """从 Neo4j 导出所有 SPU 商品数据"""
    print("[FAISS Builder] 连接 Neo4j...")
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    products = []
    with driver.session() as session:
        result = session.run("""
            MATCH (p:SPU)
            OPTIONAL MATCH (p)-[:Belong]->(c:Category3)
            OPTIONAL MATCH (p)-[:Have]->(t:Trademark)
            RETURN p.id AS id,
                   p.name AS name,
                   p.description AS description,
                   c.name AS category,
                   t.name AS brand
        """)

        for r in result:
            products.append(
                {
                    "id": str(r["id"]) if r["id"] else "",
                    "name": r["name"] or "",
                    "description": r["description"] or "",
                    "category": r["category"] or "",
                    "brand": r["brand"] or "",
                }
            )

    driver.close()
    print(f"[FAISS Builder] 从 Neo4j 获取 {len(products)} 个商品")
    return products


def build_product_index(use_neo4j=False):
    """
    商品数据 → BGE 嵌入 → FAISS 索引

    Args:
        use_neo4j: True 从 Neo4j 获取数据, False 从本地 JSON 获取
    """
    # 1. 设置 HuggingFace 镜像
    os.environ["HF_ENDPOINT"] = settings.hf_endpoint

    # 2. 加载嵌入模型
    print("[FAISS Builder] 加载嵌入模型...")
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(settings.embedding_model)
    print(f"[FAISS Builder] 模型加载完成: {settings.embedding_model}")

    # 3. 获取商品数据
    if use_neo4j:
        products = load_products_from_neo4j()
    else:
        products = load_products_from_json()

    if not products:
        print("[FAISS Builder] 未获取到商品数据，跳过索引构建")
        return

    # 4. 生成嵌入向量
    print("[FAISS Builder] 生成嵌入向量...")
    texts = [
        f"{p['name']} {p.get('description', '')} {p.get('category', '')} {p.get('brand', '')}"
        for p in products
    ]

    # 批量编码
    batch_size = 256
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = embedder.encode(batch, normalize_embeddings=True)
        all_embeddings.append(embeddings)
        print(f"  进度: {min(i + batch_size, len(texts))}/{len(texts)}")

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"[FAISS Builder] 嵌入维度: {embeddings.shape}")

    # 5. 构建 FAISS 索引
    print("[FAISS Builder] 构建 FAISS 索引...")
    import faiss

    dim = embeddings.shape[1]
    # IndexFlatIP: 内积相似度（已归一化 → 等价于余弦相似度）
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # 6. 保存
    os.makedirs(settings.faiss_index_path, exist_ok=True)
    index_file = os.path.join(settings.faiss_index_path, "products.index")
    ids_file = os.path.join(settings.faiss_index_path, "product_ids.npy")

    faiss.write_index(index, index_file)
    np.save(ids_file, np.array([p["id"] for p in products]))

    print("\n[FAISS Builder] [OK] 索引构建完成!")
    print(f"  索引文件: {index_file}")
    print(f"  ID 映射: {ids_file}")
    print(f"  商品数量: {len(products)}")
    print(f"  嵌入维度: {dim}")


if __name__ == "__main__":
    use_neo4j = "--neo4j" in sys.argv
    build_product_index(use_neo4j=use_neo4j)
