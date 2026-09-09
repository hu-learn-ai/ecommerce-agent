# RAG 评估报告 — 诊断、修复与最终结论

> 更新时间: 2026-08-20 19:45（Key 更新后已完成完整重跑）
> 涉及文件: `tests/rag_evaluation_report.json`、`tests/bm25_baseline_report.json`、`tests/bm25_baseline_report.md`、`tests/rag_evaluation.py`

## 一、事件回顾与最终结论

| 事项 | 状态 |
|---|---|
| 生成指标全 0 → 根因: DeepSeek API Key 失效（401），旧脚本静默吞错 | ✅ 已定位并修复 |
| `.env` 更换新 Key，重跑完整评估（15 用例，生成指标全部有效 error_count=0） | ✅ 已完成 |
| 生成指标真实值: **faithfulness=1.00**（零幻觉），answer_relevance=0.67 | ✅ 已取得 |
| **新发现: Neo4j 未运行**（7687 端口无响应、Docker 不可用），混合检索降级为纯向量 | ⚠️ 待环境恢复 |
| search 用例检索延迟 ~8.2s，主因 Neo4j 连接超时 × 2 轮重试（非真实检索耗时） | ⚠️ 需加快速失败 |

## 二、检索指标（可信，不依赖 LLM）

RAG 评估 15 个用例（搜索 10 + 客服 5），**当前为纯向量检索**（Neo4j 分支连接失败为空）：

| 指标 | 数值 |
|---|---:|
| recall@1 | 0.1933 |
| recall@3 | 0.5800 |
| recall@5 | 0.7000 |
| precision@5 | 0.8400 |
| mrr | 1.0000 |
| ndcg@5 | 1.0000 |
| hit_rate@5 | 1.0000 |

对照 BM25 基线报告（6 个有效用例，中立关键词 ground truth）：

| 指标 | BM25 | 向量检索 | 相对提升 |
|---|---:|---:|---:|
| recall@3 | 0.0122 | 0.0228 | +86.9% |
| recall@5 | 0.0194 | 0.0380 | +95.4% |
| precision@5 | 0.5333 | 1.0000 | +87.5% |
| mrr | 0.6667 | 1.0000 | +50.0% |
| hit_rate@5 | 0.6667 | 1.0000 | +50.0% |

> ⚠️ 口径差异：BM25 报告与 RAG 评估的测试集、ground truth 构建方式不同，两组数字不能直接横向比较，各自内部对比有效。

## 三、生成指标（Key 修复后的真实值）

| 指标 | 总体 | search (10) | cs (5) |
|---|---:|---:|---:|
| faithfulness | **1.0000** | 1.00 | 1.00 |
| answer_relevance | 0.6667 | ~0.50 | **1.00** |
| context_precision | 0.4667 | ~0.45 | ~0.60 |
| context_recall | 0.6000 | ~0.45 | **1.00** |

**解读：**
- **faithfulness 全用例 1.0**：回答完全基于检索 context，零幻觉——RAG 链路最核心的质量底线守住
- **客服 FAQ 链路表现优秀**：answer_relevance 与 context_recall 全部满分
- **search 链路 answer_relevance 偏弱**（"500元以下手机"、"笔记本电脑" 得 0 分）：
  - "500元以下手机"：系统未做价格条件过滤，返回了 ¥4439/¥5073 的手机——**结构化过滤缺失**
  - "蓝牙耳机"：Top 结果混入手机数码类无关商品（相似度 0.41 的 OPPO 手机），语义相关性不足
- context_precision 低与商品库为合成数据（商品名多为"品牌+品类"通用词）有关，向量区分度受限

## 四、性能指标（延迟拆分）

| 指标 | 数值 | 说明 |
|---|---:|---|
| avg_retrieval_latency_ms | 6064 | ⚠️ search ~8.2s 全耗在 Neo4j 超时；cs 仅 ~1.5s（真实水平） |
| avg_gen_latency_ms | 909 | LLM judge 正常耗时 |
| avg_latency_ms (总) | 6973 | — |

**改进建议**：SearchAgent 对 Neo4j 失败应快速失败/熔断（当前每查询等 2 轮超时 ≈ 8s），加健康检查或连接池超时（如 1s）。Neo4j 恢复后，预期 search 检索延迟 < 1s。

## 五、已实施的修复

1. **`.env`**：更换失效 Key（`sk-8ec***86b6` → 新 Key，已验证 200 正常）
2. **`tests/rag_evaluation.py`**：
   - 失败可见化——evaluate 失败返回 `error` 字段，不再静默吞 0
   - summary 新增 `valid` / `error_count` / `first_error`，主入口打印 ⚠️ 警告
   - `_parse_json_response()` 健壮解析（代码块/夹杂文字/尾随逗号，5 场景测试通过）
   - 延迟拆分 `retrieval_latency_ms` / `gen_latency_ms`
3. **`tests/rag_evaluation_diagnosis.md`**：本报告

## 六、待办

- [ ] 恢复 Neo4j（本机 Docker 不可用、7687 端口无响应、无 Windows 服务——需先恢复 Docker Desktop 或本地 Neo4j），重跑评估取得完整混合检索（向量+图+RRF）指标
- [ ] SearchAgent 增加 Neo4j 快速失败/熔断，消除 8s 无效等待
- [ ] search 链路补结构化过滤（价格区间等条件解析），提升 answer_relevance
- [ ] 将回填后的指标更新至 README 评估章节（原「待回填」）
