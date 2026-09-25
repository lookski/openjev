#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编写时间: 2026-09-25 02:48:06
脚本功能: 用本地 torch + HF cache 的 Qwen3-0.6B 跑 OpenJev 核心的三道题
          (yin_yang/hostility/vibe), 对样例输入输出 masked-softmax 概率,
          作为浏览器 JS 版 (transformers.js) 的黄金参照.
参数: 无 (内置两个测试输入)
输入格式: 无
输出格式: stdout: JSON 黄金数据 (写入 tmp_golden.json)
依赖: torch, transformers, openjev (本地仓库)
注意事项: 模型从本地 cache 加载 (Qwen/Qwen3-0.6B), 首跑约 10-30 秒;
          本脚本属于 Space 部署的移植验证环节, 交付后可删.
"""
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, r"D:\starPlan\1.jev")

from openjev.core import LocalJev  # noqa: E402
from openjev.webapp import QUESTIONS  # noqa: E402

SAMPLES = [
    "哎呀,你们大厂做的这个产品真不错,我们小公司哪敢提意见呀,能用了就很好了,反正在你们眼里我们也不算什么重要客户。",
    "今天开会的记录我看完了,流程图有 3 处画反了,汇报里数字对不上,麻烦改一下再发我。",
]

def main():
    engine = LocalJev(model_id="Qwen/Qwen3-0.6B", device="cpu", dtype="float32")
    golden = {"model": "Qwen/Qwen3-0.6B", "samples": []}
    for idx, text in enumerate(SAMPLES):
        entry = {"text": text, "answers": {}}
        for qid, q in QUESTIONS.items():
            ans = engine.answer_one(q, text)
            entry["answers"][qid] = ans
            print(f"sample{idx} {qid}: {json.dumps(ans, ensure_ascii=False)}")
        golden["samples"].append(entry)
    golden["usage"] = dict(engine.usage)
    with open(r"D:\starPlan\1.jev\tmp_golden.json", "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=1)
    print("golden saved -> tmp_golden.json")

if __name__ == "__main__":
    main()
