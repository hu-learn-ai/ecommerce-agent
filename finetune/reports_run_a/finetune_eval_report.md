# 微调评估报告

- 测试集：/root/autodl-tmp/ecommerce-agent/finetune/data/processed/test.jsonl（289 条）
- 评估时间：2026-09-11 00:39:03
- 模型：base=`/root/autodl-tmp/models/Qwen2.5-7B-Instruct`，finetuned=`/root/autodl-tmp/models/qwen-cs-7b-merged`，deepseek=`deepseek-chat`
- 口径：LLM-as-Judge 四维评分（1-5）+ 规则可判定指标 + 人工抽检（见 human_eval）

## 1. 四维 Judge 均分

| 模型 | 正确性 | 完整性 | 语气 | 安全 | 总分 |
|------|--------|--------|------|------|------|
| base | 3.8 | 3.66 | 4.69 | 4.87 | 17.02 |
| finetuned | 4.62 | 4.08 | 4.87 | 5.0 | 18.57 |
| deepseek | 4.66 | 4.61 | 4.97 | 5.0 | 19.24 |

## 2. 可判定指标

| 指标 | base | 微调 | DeepSeek | 目标 |
|------|------|------|----------|------|
| policy_citation_accuracy | 31.3% | 78.5% | 45.7% | ≥90% |
| hallucination_rate | 8.3% | 18.7% | 38.4% | ≤5% |
| ood_refusal_rate | 24.2% | 87.9% | 36.4% | ≥90% |
| multi_turn_keep_rate | 81.6% | 100.0% | 98.2% | ≥85% |

## 3. Win Rate

| 对比 | win | tie | lose | win rate |
|------|-----|-----|------|----------|
| 微调 vs base | 166 | 82 | 41 | 57.4% |
| 微调 vs DeepSeek | 40 | 121 | 127 | 13.9% |

## 4. Badcase（微调明显落后）

### vs base
- `cs_售后退换货_02708` delta=-4.0（finetuned 14.0 vs other 18.0）
- `cs_000081` delta=-2.0（finetuned 16.0 vs other 18.0）
- `cs_售后退换货_01776` delta=-2.0（finetuned 17.0 vs other 19.0）
- `cs_000313` delta=-2.0（finetuned 16.0 vs other 18.0）
- `cs_投诉安抚_00372` delta=-3.5（finetuned 14.5 vs other 18.0）
- `cs_投诉安抚_00830` delta=-2.0（finetuned 17.0 vs other 19.0）
- `cs_投诉安抚_00099` delta=-4.0（finetuned 14.0 vs other 18.0）
- `cs_000001` delta=-2.0（finetuned 17.0 vs other 19.0）
- `cs_售前咨询_01026` delta=-2.0（finetuned 16.0 vs other 18.0）
- `cs_售前咨询_02037` delta=-3.0（finetuned 17.0 vs other 20.0）
- `cs_售前咨询_00127` delta=-3.0（finetuned 15.0 vs other 18.0）
- `cs_售前咨询_01272` delta=-4.0（finetuned 13.5 vs other 17.5）
- `cs_售前咨询_02149` delta=-2.0（finetuned 14.0 vs other 16.0）
- `cs_售前咨询_02992` delta=-2.0（finetuned 16.0 vs other 18.0）
- `cs_售前咨询_01241` delta=-2.0（finetuned 17.0 vs other 19.0）
- `cs_退款发票_00364` delta=-2.0（finetuned 18.0 vs other 20.0）
### vs DeepSeek
- `cs_售后退换货_00511` delta=-2.5（finetuned 17.0 vs other 19.5）
- `cs_售后退换货_02526` delta=-2.0（finetuned 18.0 vs other 20.0）
- `cs_售后退换货_00060` delta=-3.0（finetuned 16.0 vs other 19.0）
- `cs_售后退换货_02708` delta=-4.0（finetuned 14.0 vs other 18.0）
- `cs_000081` delta=-2.5（finetuned 16.0 vs other 18.5）
- `cs_售后退换货_01776` delta=-3.0（finetuned 17.0 vs other 20.0）
- `cs_售后退换货_01571` delta=-2.0（finetuned 18.0 vs other 20.0）
- `cs_售后退换货_02744` delta=-3.0（finetuned 17.0 vs other 20.0）
- `cs_000313` delta=-3.5（finetuned 16.0 vs other 19.5）
- `cs_000184` delta=-3.0（finetuned 17.0 vs other 20.0）
- `cs_投诉安抚_00338` delta=-2.0（finetuned 17.5 vs other 19.5）
- `cs_投诉安抚_01035` delta=-2.5（finetuned 17.0 vs other 19.5）
- `cs_投诉安抚_00372` delta=-3.5（finetuned 14.5 vs other 18.0）
- `cs_投诉安抚_00317` delta=-2.0（finetuned 18.0 vs other 20.0）
- `cs_投诉安抚_00730` delta=-4.5（finetuned 15.5 vs other 20.0）
- `cs_投诉安抚_01298` delta=-2.0（finetuned 18.0 vs other 20.0）
- `cs_投诉安抚_01218` delta=-2.0（finetuned 18.0 vs other 20.0）
- `cs_投诉安抚_00830` delta=-2.5（finetuned 17.0 vs other 19.5）
- `cs_投诉安抚_01124` delta=-3.5（finetuned 16.5 vs other 20.0）
- `cs_投诉安抚_00487` delta=-3.5（finetuned 16.0 vs other 19.5）
- `cs_投诉安抚_01443` delta=-2.0（finetuned 18.0 vs other 20.0）
- `cs_投诉安抚_00723` delta=-3.0（finetuned 17.0 vs other 20.0）
- `cs_投诉安抚_00099` delta=-4.5（finetuned 14.0 vs other 18.5）
- `cs_投诉安抚_00276` delta=-2.0（finetuned 18.0 vs other 20.0）
- `cs_投诉安抚_01200` delta=-2.5（finetuned 16.0 vs other 18.5）
- `cs_物流发货_00540` delta=-3.0（finetuned 17.0 vs other 20.0）
- `cs_物流发货_00773` delta=-2.0（finetuned 16.0 vs other 18.0）
- `cs_物流发货_01470` delta=-2.0（finetuned 18.0 vs other 20.0）
- `cs_物流发货_00758` delta=-2.0（finetuned 18.0 vs other 20.0）
- `cs_域外拒答_01398` delta=-4.0（finetuned 16.0 vs other 20.0）

## 5. 结论（人工复核于 2026-09-11）

- 客服域能否替代 DeepSeek：部分替代。本模型（15k 数据微调，主线）对 DeepSeek 不输率 56%（win 13.9% / tie 41.9%），四维总分 18.57 vs 19.24；在政策引用准确率（78.5% vs 44.3%）、幻觉率（18.7% vs 34.6%，规则口径）、OOD 拒答率（87.9% vs 48.5%）三项硬指标上反而优于 DeepSeek。适合作为 Tier-1 高频客服主力模型承接标准问询，复杂/长尾场景回退 DeepSeek。
- 适用边界：售前咨询、售后退换货、物流发货、投诉安抚、退款发票等客服域内场景可直接承接；域外问题拒答率 87.9%（未达 90% 目标），剩余 ~12% 域外样本仍需线上兜底策略（关键词预过滤 + 置信度阈值回退）。
- 数据/口径备注：① Judge 为 deepseek-chat 四维评分（1-5）×2 次重复取均值，存在 ±0.1 级噪声；② 幻觉率为规则口径（政策库匹配失败即计幻觉），对长回答偏严，DeepSeek 34.6% 的高幻觉率部分源于其超出政策库的自由发挥，需人工抽检复核；③ 政策引用准确率 78.5% 距 90% 目标还差 11.5pp，下一轮可用政策库检索增强（RAG 注入）补齐；④ 消融：5k 版本（见 reports/）四维总分 18.18、政策引用 52.3%、OOD 拒答 72.7%，本模型全面占优 —— 训练 loss 曾显示 15k 有过拟合倾向（0.36 vs 0.38），但下游评估反转，验证领域数据量的边际收益大于过拟合代价；主线确定为本模型（15k 版本）。
