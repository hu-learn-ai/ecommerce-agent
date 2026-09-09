# data/processed

本目录为数据处理产物，**不随仓库分发**（已被 `.gitignore` 排除）。

原目录中包含从淘宝用户消费数据派生的导入脚本与训练集（用户/订单/行为数据），
出于隐私考虑已移入本机备份目录 `.private_backup/`。

如需复现，先运行 `python scripts/import_taobao_data.py` 重新生成：

- `mysql_gmall.sql` — MySQL gmall 库初始化脚本
- `neo4j_import.cypher` — Neo4j 图谱导入脚本
- `classify_train_v2.csv` — 商品分类训练集
- `products_for_faiss.json` / `faq_data.json` — FAISS 索引构建输入

生成后可参考根目录 README 的快速开始部分完成数据导入。
