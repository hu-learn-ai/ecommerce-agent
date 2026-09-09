"""
商品分类 Agent — 复用 product_classification 训练好的 BERT 模型

输入商品标题/描述 → 输出模型标签集中的商品类别之一
模型: bert-base-chinese + 加权交叉熵 + 混合精度训练

学习参考: product_classification/src/web/service.py
"""

import os
import re
from typing import List

import torch

from agents.tools.base import BaseAgentTool


class ClassifyAgent(BaseAgentTool):
    """
    商品标题自动分类

    复用 product_classification 项目训练好的 BERT 分类模型
    支持 Top-1 和 Top-K 分类结果输出
    """

    name: str = "classify_agent"
    description: str = "商品分类：根据商品标题文本自动分类到模型标签集中的一个类别，返回分类结果和置信度"

    # 置信度低于该值时判定为"无法判断"，避免模型对陌生/歧义文本硬塞一个类
    MIN_CONFIDENCE: float = 0.60

    # 4 类模型不覆盖的品类关键词：命中直接拒识（避免高置信硬塞）
    OUT_OF_DOMAIN_KEYWORDS = (
        # 服装鞋包
        "连衣裙", "T恤", "衬衫", "裤子", "牛仔裤", "卫衣", "外套", "羽绒服", "内衣", "文胸",
        "运动鞋", "篮球鞋", "跑步鞋", "皮鞋", "靴子",
        "西装", "风衣", "夹克", "棉服", "马甲", "背心", "短裤", "休闲裤", "打底裤",
        "半身裙", "旗袍", "汉服", "泳衣", "睡衣", "袜子", "丝袜", "船袜", "帽子", "手套",
        "围巾", "领带", "鞋垫",
        # 美妆护肤
        "口红", "粉底", "面膜", "眼影", "睫毛膏", "香水", "防晒霜", "精华液", "洁面",
        "面霜", "乳液", "爽肤水", "化妆水", "隔离霜", "遮瑕", "眉笔", "眼线", "腮红",
        "卸妆", "洗面奶", "护肤", "彩妆", "美甲", "指甲油", "假发",
        # 母婴用品
        "纸尿裤", "奶瓶", "婴儿", "童装", "孕妇",
        "奶粉", "辅食", "玩具", "积木", "拼图", "遥控车", "芭比", "乐高", "毛绒",
        "婴儿车", "婴儿床", "安全座椅", "学步车", "摇铃", "布书", "滑板车", "扭扭车",
        "奶嘴", "吸奶器", "待产包", "早教",
        # 汽车用品
        "机油", "轮胎", "车膜", "行车记录仪", "车载",
        "汽车脚垫", "座垫", "车充", "洗车", "玻璃水", "雨刮", "车衣", "车蜡",
        "摩托车", "头盔",
        # 珠宝首饰
        "项链", "戒指", "手链", "耳环", "黄金", "钻石", "翡翠", "银饰",
        "珠宝", "首饰", "手镯", "吊坠", "铂金", "珍珠", "水晶", "琥珀",
        # 图书音像/文具
        "教材", "小说", "绘本", "漫画", "文具", "字帖",
        "图书", "书籍", "杂志", "音像", "电子书", "考试", "考研",
        "中性笔", "圆珠笔", "钢笔", "铅笔", "记号笔", "荧光笔", "文件夹", "订书机",
        "打印纸", "A4纸", "便签", "胶带",
        # 宠物用品
        "猫粮", "狗粮", "猫砂", "宠物", "鱼缸", "鸟笼",
        "猫爬架", "狗窝", "宠物窝", "牵引绳", "宠物玩具", "宠物零食", "仓鼠",
        # 运动户外
        "帐篷", "登山包", "瑜伽垫", "哑铃", "自行车", "羽毛球拍",
        "跑步机", "椭圆机", "划船机", "杠铃", "跳绳", "呼啦圈", "滑板", "轮滑",
        "高尔夫", "鱼竿", "渔具", "乒乓球", "篮球", "足球", "排球", "网球拍",
        "登山杖", "冲锋衣", "徒步", "野餐垫", "烧烤架", "露营", "骑行", "护膝",
        # 家居家装/日用
        "沙发", "床垫", "窗帘", "墙纸", "收纳盒", "灯具", "四件套",
        "床单", "被套", "被子", "枕头", "毛毯", "蚊帐", "地毯", "靠垫", "抱枕",
        "书桌", "办公椅", "电脑椅", "茶几", "餐桌", "电视柜", "书柜", "书架", "衣柜",
        "衣架", "鞋架", "置物架", "床头柜", "梳妆台", "地垫", "门垫",
        "保温杯", "水杯", "马克杯", "雨伞", "拖鞋", "拖把", "扫把", "垃圾桶",
        "垃圾袋", "挂钩", "螺丝刀", "扳手", "锤子", "电钻", "五金",
        "炒锅", "不粘锅", "砂锅", "蒸锅", "高压锅", "刀具", "砧板", "碗", "盘子",
        "餐具", "保鲜盒", "饭盒", "毛巾", "浴巾", "牙膏", "牙刷", "洗发水",
        "沐浴露", "洗衣液", "洗洁精", "纸巾", "抽纸", "湿巾",
        # 乐器
        "吉他", "钢琴", "电子琴", "小提琴", "古筝", "二胡", "架子鼓", "尤克里里",
        # 眼镜/表/箱包配饰
        "眼镜", "太阳镜", "隐形眼镜", "眼镜框", "机械表", "石英表",
        "钱包", "墨镜", "围巾", "行李箱", "箱包", "双肩包", "手提包", "挎包",
        # 礼品鲜花
        "鲜花", "花束", "贺卡", "礼品卡",
    )

    def __init__(self, model_path: str, labels_path: str):
        super().__init__()
        self.model_path = model_path
        self.labels_path = labels_path

        # 延迟加载模型
        self._tokenizer = None
        self._model = None
        self._labels = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._ood_gate = None

    # ------------------------------------------------------------------ #
    #  懒加载
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self):
        """懒加载模型和标签（经 ModelCache 进程内持久复用）"""
        if self._model is not None:
            return

        from models.persistence import load_classification_model, validate_model_dir

        if os.path.isdir(self.model_path) and not validate_model_dir(self.model_path):
            try:
                self._tokenizer, self._model, self._labels, self._device = (
                    load_classification_model(self.model_path, self.labels_path)
                )
                from models.persistence import load_ood_gate

                self._ood_gate = load_ood_gate(self.model_path)
                if self._ood_gate:
                    thr = self._ood_gate.get("thresholds")
                    thr_desc = (
                        ",".join(f"{t:.1f}" for t in thr)
                        if thr is not None and not isinstance(thr, float)
                        else str(thr)
                    )
                    print(
                        f"[ClassifyAgent] 域外门控已加载 (逐类阈值: {thr_desc})"
                    )
                print(
                    f"[ClassifyAgent] 模型加载完成（缓存复用），共 {len(self._labels)} 个类别"
                )
                return
            except Exception as exc:  # noqa: BLE001
                print(f"[ClassifyAgent] 模型加载失败: {exc}")
        elif os.path.isdir(self.model_path):
            from models.persistence import validate_model_dir

            print(
                f"[ClassifyAgent] 模型目录不完整，缺失文件: {validate_model_dir(self.model_path)}"
            )
        else:
            print(f"[ClassifyAgent] 模型路径不存在: {self.model_path}")

        print("[ClassifyAgent] 将使用规则降级方案")
        self._model = None
        self._tokenizer = None
        self._labels = self._default_labels()

    def _default_labels(self) -> List[str]:
        """默认标签（与当前 4 类训练模型一致；仅在模型缺失时兜底）"""
        return [
            "医药保健",
            "家用电器",
            "手机数码",
            "食品生鲜",
        ]

    # ------------------------------------------------------------------ #
    #  对外接口
    # ------------------------------------------------------------------ #

    def run(self, query: str, **kwargs) -> str:
        """同步执行商品分类"""
        return self.classify_product(query)

    async def arun(self, **kwargs) -> str:
        """异步执行"""
        return self.run(**kwargs)

    # ------------------------------------------------------------------ #
    #  核心分类逻辑
    # ------------------------------------------------------------------ #

    def classify_product(self, title: str) -> str:
        """
        根据商品标题自动分类到模型标签类别之一

        Args:
            title: 商品标题文本

        Returns:
            分类结果字符串，包含类别名和置信度
        """
        self._ensure_loaded()

        # 预处理
        title = self._preprocess(title)
        if self._is_junk_input(title):
            return "分类结果: 无法判断 (输入无效)"
        if self._is_out_of_domain(title):
            return f"分类结果: 无法判断 (超出模型覆盖范围: {title[:30]})"

        # 模型推理
        if self._model is not None and self._tokenizer is not None:
            return self._model_predict(title)
        else:
            # 降级：基于关键词规则的分类
            return self._rule_based_classify(title)

    def get_top_k(self, title: str, k: int = 3) -> List[dict]:
        """
        返回 Top-K 分类结果

        Args:
            title: 商品标题
            k: 返回数量

        Returns:
            [{"category": "...", "confidence": "12.34%"}, ...]
        """
        self._ensure_loaded()

        title = self._preprocess(title)
        if self._is_junk_input(title) or self._is_out_of_domain(title):
            return [{"category": "无法判断", "confidence": "N/A"}]

        if self._model is not None and self._tokenizer is not None:
            inputs = self._tokenizer(
                title, max_length=128, padding=True, truncation=True, return_tensors="pt"
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs, output_hidden_states=True)
                probs = torch.softmax(outputs.logits, dim=-1)
                top_k_result = torch.topk(probs, k, dim=-1)

            if self._ood_gate is not None:
                pred_idx = top_k_result.indices[0][0].item()
                ood_score = self._ood_score(outputs.hidden_states[-1][:, 0], pred_idx)
                if ood_score > self._class_threshold(pred_idx):
                    return [
                        {
                            "category": "无法判断",
                            "confidence": f"马氏距离 {ood_score:.2f}",
                        }
                    ]

            top_conf = top_k_result.values[0][0].item()
            if top_conf < self.MIN_CONFIDENCE:
                return [
                    {
                        "category": "无法判断",
                        "confidence": f"{top_conf:.2%}",
                    }
                ]
            return [
                {
                    "category": self._labels[idx],
                    "confidence": f"{conf:.2%}",
                }
                for idx, conf in zip(top_k_result.indices[0], top_k_result.values[0])
            ]
        else:
            # 降级
            return [{"category": self._rule_based_classify(title), "confidence": "N/A"}]

    def _model_predict(self, title: str) -> str:
        """使用 BERT 模型进行推理"""
        inputs = self._tokenizer(
            title,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()

        # 域外门控：到预测类的马氏距离超过该类阈值 → 判定为覆盖范围外
        if self._ood_gate is not None:
            ood_score = self._ood_score(outputs.hidden_states[-1][:, 0], pred_idx)
            if ood_score > self._class_threshold(pred_idx):
                return (
                    f"分类结果: 无法判断 "
                    f"(疑似超出模型覆盖范围, 马氏距离: {ood_score:.2f})"
                )

        category = self._labels[pred_idx] if pred_idx < len(self._labels) else "未知"
        confidence = probs[0][pred_idx].item()

        if confidence < self.MIN_CONFIDENCE:
            return f"分类结果: 无法判断 (置信度: {confidence:.2%})"
        return f"分类结果: {category} (置信度: {confidence:.2%})"

    def _ood_score(self, cls_feature, pred_idx: int) -> float:
        """计算 [CLS] 特征到预测类的马氏距离（域外门控核心，越大越可能域外）。"""
        import numpy as np

        try:
            feat = cls_feature.detach().cpu().numpy().astype(np.float32).reshape(1, -1)
            # 与 build_ood_stats.py 保持一致：特征先做 L2 归一化
            feat = feat / (np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-8)
            means = self._ood_gate["means"].astype(np.float64)
            precs = self._ood_gate["precs"].astype(np.float64)
            c = pred_idx if 0 <= pred_idx < len(means) else 0
            d = feat.astype(np.float64) - means[c]
            q = float(np.einsum("bi,ij,bj->b", d, precs[c], d)[0])
            return float(np.sqrt(max(q, 0.0)))
        except Exception:
            # 门控计算失败时放行（不阻断主分类流程）
            return 0.0

    def _class_threshold(self, pred_idx: int) -> float:
        """返回预测类对应的马氏距离阈值（逐类）。"""
        thresholds = self._ood_gate.get("thresholds")
        if thresholds is None:
            return float(self._ood_gate.get("threshold", 1e9))
        if 0 <= pred_idx < len(thresholds):
            return float(thresholds[pred_idx])
        return 1e9

    def _is_junk_input(self, text: str) -> bool:
        """无效输入检测：空/超长/纯符号/纯数字。"""
        if not text or len(text) < 2 or len(text) > 200:
            return True
        if not re.search(r"[\u4e00-\u9fa5A-Za-z0-9]", text):
            return True
        if re.fullmatch(r"[\d\s.,%\-/]+", text):
            return True
        return False

    def _is_out_of_domain(self, text: str) -> bool:
        """域外拒识：命中 4 类模型不覆盖的品类关键词。"""
        return any(kw.lower() in text.lower() for kw in self.OUT_OF_DOMAIN_KEYWORDS)

    def _rule_based_classify(self, title: str) -> str:
        """
        规则降级方案：基于关键词匹配进行分类
        当 BERT 模型不可用时使用

        仅匹配当前模型标签集内的类别，避免输出模型不认识的分类
        """
        keyword_map = {
            "医药保健": ["药", "保健品", "维生素", "保健", "医用", "口罩", "创可贴", "血压"],
            "图书音像": ["书", "图书", "小说", "教材", "杂志", "CD", "DVD", "音像", "电子书"],
            "宠物用品": ["宠物", "狗粮", "猫粮", "猫砂", "狗", "猫", "鱼缸", "鸟笼", "宠物玩具"],
            "家居家装": [
                "床品",
                "枕头",
                "被子",
                "毛巾",
                "窗帘",
                "地毯",
                "灯",
                "装饰",
                "收纳",
                "墙纸",
                "装修",
            ],
            "家用电器": ["冰箱", "洗衣机", "空调", "电视", "微波炉", "电饭煲", "吸尘器", "电磁炉"],
            "手机数码": [
                "手机",
                "平板",
                "相机",
                "耳机",
                "充电器",
                "数据线",
                "蓝牙",
                "音箱",
                "键盘",
                "鼠标",
                "显示器",
                "打印机",
                "电脑",
                "笔记本",
            ],
            "服装鞋包": [
                "衣服",
                "衬衫",
                "裤子",
                "T恤",
                "裙子",
                "内衣",
                "文胸",
                "鞋",
                "靴",
                "包",
                "箱",
                "背包",
                "手提",
                "外套",
                "夹克",
            ],
            "母婴用品": ["婴儿", "纸尿裤", "奶瓶", "奶粉", "婴儿车", "孕", "宝宝", "母婴", "童装"],
            "汽车用品": [
                "汽车",
                "车载",
                "轮胎",
                "车膜",
                "行车记录仪",
                "车充",
                "座垫",
                "机油",
                "壳牌",
            ],
            "珠宝首饰": ["珠宝", "首饰", "项链", "戒指", "耳环", "手镯", "黄金", "钻石", "银饰"],
            "礼品鲜花": ["礼品", "礼物", "鲜花", "花束", "贺卡", "包装", "礼盒"],
            "箱包配饰": ["箱包", "钱包", "皮带", "围巾", "帽子", "手套", "眼镜", "手表", "配饰"],
            "美妆护肤": [
                "面膜",
                "口红",
                "粉底",
                "精华",
                "乳液",
                "化妆",
                "护肤",
                "香水",
                "防晒",
                "洁面",
            ],
            "运动户外": [
                "运动",
                "健身",
                "跑步",
                "瑜伽",
                "户外",
                "露营",
                "帐篷",
                "登山",
                "骑行",
                "球拍",
            ],
            "食品生鲜": [
                "食品",
                "零食",
                "水果",
                "蔬菜",
                "海鲜",
                "牛肉",
                "猪肉",
                "牛奶",
                "饮料",
                "茶叶",
                "坚果",
                "巧克力",
            ],
        }

        # 只保留当前标签集内的类别（当前为 4 类）
        keyword_map = {cat: kws for cat, kws in keyword_map.items() if cat in self._labels}

        title_lower = title.lower()
        for category, keywords in keyword_map.items():
            for kw in keywords:
                if kw.lower() in title_lower:
                    return f"分类结果: {category} (规则匹配，置信度: N/A)"

        return "分类结果: 其他 (规则匹配，置信度: N/A)"

    def _preprocess(self, text: str) -> str:
        """
        文本预处理（与训练时保持一致）

        - 全角转半角
        - 去除多余空白
        - 去除特殊字符
        """
        import unicodedata

        # 全角转半角
        text = unicodedata.normalize("NFKC", text)

        # 去除多余空白
        text = " ".join(text.split())

        # 去除明显的特殊字符（保留中文、英文、数字、基本标点）
        text = "".join(
            c for c in text if c.isalnum() or c in " ()（）【】[]-—_/,，.。;；:：!！?？%％+"
        )

        return text.strip()
