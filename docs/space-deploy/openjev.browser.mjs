/**
 * openjev.browser.js — OpenJev masked-softmax core, portable JS port.
 *
 * 与 Python openjev/core.py 逐字节一致的关键点:
 *   - build_user_text: <state> 包裹 + Question 段 + 选项行 + 答案指令行
 *   - Qwen3 chat template (enable_thinking=false): ChatML + 空 think 块
 *   - 标签 token id (Qwen3 tokenizer): A/B/C=32/33/34, 0..3=15..18, Yes/No=9454/2753
 *   - masked softmax: 先整体 softmax (其余 -inf), 再归一化 picked
 *
 * UMD: node (require) 与浏览器 (window.OpenJev) 双端可用.
 */
// ESM 模块 (浏览器版). 数据/逻辑与 openjev/core.py + webapp.py 一致.
  const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
  const DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
  const YESNO = ["Yes", "No"];

  const DEFAULT_SYSTEM_PROMPT =
    "You are a precise decision engine. " +
    "You do not generate text. " +
    "You answer only with the exact label requested.";

  // Qwen/Qwen3-0.6B tokenizer first-token ids (verified against Python)
  const LABEL_IDS = {
    letters: LETTERS.map((_, i) => 32 + i), // A=32 ... Z=57
    digits: [15, 16, 17, 18, 19, 20, 21, 22, 23, 24], // "0".."9"
    yesno: [9454, 2753], // "Yes", "No"
  };

  function buildUserText(stateText, q) {
    const parts = ["<state>", stateText, "</state>", "", "Question:", q.instructions];
    if (q.type === "choice") {
      const names = Object.keys(q.criteria);
      const lines = names.map((n, i) => LETTERS[i] + ". " + q.criteria[n]);
      parts.push("", lines.join("\n"), "", "Answer with exactly one letter.");
    } else if (q.type === "score") {
      const lines = q.criteria.map((d, i) => DIGITS[i] + " = " + d);
      parts.push(
        "",
        lines.join("\n"),
        "",
        "Answer with exactly one digit for the best level."
      );
    } else if (q.type === "noul") {
      if (q.criteria) parts.push("", String(q.criteria));
      parts.push("", "Answer with exactly one word, Yes or No.");
    } else {
      throw new Error("unknown question type: " + q.type);
    }
    return parts.join("\n");
  }

  // Qwen3 template, add_generation_prompt=True, enable_thinking=False:
  // ...<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
  function buildPrompt(stateText, q, systemPrompt) {
    const sys = systemPrompt || DEFAULT_SYSTEM_PROMPT;
    const user = buildUserText(stateText, q);
    return (
      "<|im_start|>system\n" +
      sys +
      "<|im_end|>\n" +
      "<|im_start|>user\n" +
      user +
      "<|im_end|>\n" +
      "<|im_start|>assistant\n" +
      "<think>\n\n</think>\n\n"
    );
  }

  function softmaxPicked(logitsRow, labelIds) {
    const picked = labelIds.map((id) => Number(logitsRow[id]));
    const max = Math.max.apply(null, picked);
    const exps = picked.map((v) => Math.exp(v - max));
    const s = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / s);
  }

  // 与 openjev/webapp.py QUESTIONS 一致
  const QUESTIONS = {
    yin_yang: {
      type: "noul",
      instructions: "这条消息是不是在阴阳怪气 (表面客气实际讽刺)?",
    },
    hostility: {
      type: "score",
      instructions: "说话人的敌意程度",
      criteria: [
        "没有敌意, 正常交流",
        "有点不爽, 但还在忍",
        "明显不满, 讽刺意味浓",
        "敌意拉满, 就差骂人了",
      ],
    },
    vibe: {
      type: "choice",
      instructions: "这条消息的真实语气更像",
      criteria: {
        sincere: "真诚的, 表面和实际一致",
        passive_aggressive: "被动攻击, 用客气包裹不满",
        pure_sarcasm: "纯讽刺, 明摆着阴阳怪气",
      },
    },
  };

  const VIBE_NAMES = ["sincere", "passive_aggressive", "pure_sarcasm"];

  // 对三行 last-position logits 分别做三题判定 (每题独立 softmax), 返回渲染结果
  function answerAll(ynRow, d4Row, c3Row) {
    const yn = softmaxPicked(ynRow, LABEL_IDS.yesno.slice(0, 2));
    const d4 = softmaxPicked(d4Row, LABEL_IDS.digits.slice(0, 4));
    const c3 = softmaxPicked(c3Row, LABEL_IDS.letters.slice(0, 3));
    const kmax = (ps) => ps.reduce((bi, p, i) => (p > ps[bi] ? i : bi), 0);
    const conf = (ps) => {
      const K = ps.length;
      return (K * Math.max.apply(null, ps) - 1) / (K - 1);
    };
    return {
      yin_yang: { type: "noul", noul: yn[0] },
      hostility: {
        type: "score",
        score: d4.reduce((a, p, i) => a + p * i, 0),
        probabilities: Object.fromEntries(DIGITS.slice(0, 4).map((d, i) => [d, d4[i]])),
        confidence: conf(d4),
      },
      vibe: {
        type: "choice",
        choice: VIBE_NAMES[kmax(c3)],
        probabilities: Object.fromEntries(VIBE_NAMES.map((n, i) => [n, c3[i]])),
        confidence: conf(c3),
      },
    };
  }

export {
  LETTERS,
  DIGITS,
  YESNO,
  DEFAULT_SYSTEM_PROMPT,
  LABEL_IDS,
  QUESTIONS,
  VIBE_NAMES,
  buildUserText,
  buildPrompt,
  softmaxPicked,
  answerAll,
};
