#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编写时间: 2026-09-25 11:12:00
脚本功能: 用本地 OpenJev 引擎 (torch + Qwen3-0.6B) 为宣传物料生成机器真值演示数据:
          4 条例句 (阴阳/正常/半阴半阳/直球) x 3 道题 (yin_yang/hostility/vibe),
          输出 canonical JSON + Markdown 表格, 供 README/知乎文案直接引用.
参数: 无
输入格式: 无 (内置例句)
输出格式: docs/demo_data.json (canonical), docs/demo_table.md (由 JSON 程序化渲染)
依赖: torch, transformers, openjev 本地包
注意事项: 遵守科研数据真值协议 - 宣传文案中的所有统计数字必须来自本脚本输出,
          禁止人工编造或誊抄旧数字; 模型从本地 cache 加载, CPU 推理约 1-2 分钟.
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, r"D:\starPlan\1.jev")

from openjev.core import LocalJev  # noqa: E402
from openjev.webapp import QUESTIONS  # noqa: E402

SAMPLES = [
    {"id": "yin", "label": "经典阴阳", "text": "哎呀,你们大厂做的这个产品真不错,我们小公司哪敢提意见呀,能用了就很好了,反正在你们眼里我们也不算什么重要客户。"},
    {"id": "normal", "label": "正常沟通", "text": "今天开会的记录我看完了,流程图有 3 处画反了,汇报里数字对不上,麻烦改一下再发我。"},
    {"id": "half", "label": "半阴半阳", "text": "行,你说得都对,我水平低看不懂你的高深方案,那就按你说的办吧,出了问题反正也轮不到我背锅。"},
    {"id": "direct", "label": "直球吐槽", "text": "这个需求改了八遍了还没完,到底想怎么样?能不能一次说清楚再动手,我很累了。"},
]

def main():
    engine = LocalJev(model_id="Qwen/Qwen3-0.6B", device="cpu", dtype="float32")
    out = {"generated_by": "tmp_promo_data.py", "model": "Qwen/Qwen3-0.6B", "samples": []}
    for s in SAMPLES:
        entry = {"id": s["id"], "label": s["label"], "text": s["text"], "answers": {}}
        for qid, q in QUESTIONS.items():
            entry["answers"][qid] = engine.answer_one(q, s["text"])
        out["samples"].append(entry)
        yy = entry["answers"]["yin_yang"]["noul"]
        hs = entry["answers"]["hostility"]["score"]
        vc = entry["answers"]["vibe"]["choice"]
        print(f"{s['label']}: yin_yang={yy:.4f} hostility={hs:.4f} vibe={vc}")

    doc_dir = r"D:\starPlan\1.jev\docs"
    os.makedirs(doc_dir, exist_ok=True)
    with open(os.path.join(doc_dir, "demo_data.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # 程序化渲染 Markdown 表格
    lines = [
        "# 演示数据 (机器真值, 由 tmp_promo_data.py 生成)",
        "",
        f"模型: {out['model']} · 本地 CPU · masked-softmax",
        "",
        "| 例句 | 阴阳怪气概率 (Yes) | 敌意评分 (0-3) | 真实语气 |",
        "|---|---|---|---|",
    ]
    V = {"sincere": "真诚", "passive_aggressive": "被动攻击", "pure_sarcasm": "纯讽刺"}
    for e in out["samples"]:
        yn = e["answers"]["yin_yang"]["noul"]
        hs = e["answers"]["hostility"]["score"]
        v = e["answers"]["vibe"]["choice"]
        lines.append(f"| {e['label']} | {yn*100:.1f}% | {hs:.2f} | {V[v]} |")
    lines += ["", "> 本文所有数字由脚本自动生成, 引用时请以 docs/demo_data.json 为准."]
    with open(os.path.join(doc_dir, "demo_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("canonical -> docs/demo_data.json ; rendered -> docs/demo_table.md")

if __name__ == "__main__":
    main()
