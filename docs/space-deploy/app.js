/**
 * app.js — OpenJev 浏览器版主逻辑 (双模式).
 *
 * 模式选择:
 *   - navigator.gpu 可用且 requestAdapter 成功 -> fp16 (1.2GB, ./model), GPU 执行,
 *     概率与 Python 参考最大误差 0.0032 (高精度模式)
 *   - 否则 -> q4f16 (0.5GB, ./model_q4), wasm/CPU 执行 (量化快速模式, 概率有
 *     量化偏差, UI 明示)
 *
 * 全部权重与 ort wasm 自托管 (HF Static Space 同源), 不依赖外部 CDN;
 * import map 由 index.html 注入, 把 onnxruntime-web/webgpu 与 onnxruntime-common
 * 映射到 vendor/ort.all.bundle.min.mjs (jsep 构建, 含 webgpu EP).
 */
import { AutoTokenizer, AutoModelForCausalLM, env } from "./vendor/transformers.web.js";
import * as O from "./openjev.browser.mjs";

// env 设置与探针 (probe_fp16.html, 已验证 PASS) 完全一致:
// 只设 allowLocalModels / localModelPath / wasmPaths / numThreads / proxy,
// 不动 useBrowserCache / useWasmCache (默认值即探针行为)
env.allowLocalModels = true;
env.localModelPath = "./";
env.backends.onnx.wasm.wasmPaths = "./";
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;

const REPO_FALLBACK = "onnx-community/Qwen3-0.6B-ONNX";

const els = {
  input: document.getElementById("input"),
  btn: document.getElementById("btn"),
  status: document.getElementById("status"),
  progress: document.getElementById("progress"),
  results: document.getElementById("results"),
  mode: document.getElementById("mode"),
};

let tokenizer = null;
let model = null;
let ready = false;
let degraded = false;

function setStatus(t) { els.status.textContent = t; }
function setProgress(frac, label) {
  els.progress.style.width = (frac * 100).toFixed(1) + "%";
  if (label) setStatus(label);
}

async function pickProfile() {
  // 权重档选择:
  //   有 WebGPU 适配器 -> fp16 高精度 (误差 0.0032), 优先本地自托管, 失败则从
  //   Hub 官方仓库 onnx-community/Qwen3-0.6B-ONNX 加载 (Space 配额 1GB 放不下 fp16);
  //   无适配器 -> q4f16 量化快速档 (Space 自托管, wasm 内存可控)
  // 两种档都向 ORT 请求 device:"webgpu" (jsep 内部无适配器时自动回退 CPU,
  // 与探针验证路径一致)
  if (navigator.gpu) {
    try {
      const ad = await navigator.gpu.requestAdapter();
      if (ad) return { path: "./model", dtype: "fp16", device: "webgpu", degraded: false };
    } catch (e) { console.warn("requestAdapter failed:", e); }
  }
  return { path: "./model_q4", dtype: "q4f16", device: "wasm", degraded: true };
}

async function loadModel(profile, cb) {
  const opts = { device: profile.device, dtype: profile.dtype, progress_callback: cb };
  // 进度停滞保护: WebGPU session 创建在某些环境会无限挂起 (下载 100% 后无事件);
  // 无进度超过 idleMs 或总时长超过 totalMs 即放弃, 由调用方回退量化档
  let last = Date.now();
  const wrapped = (p) => { last = Date.now(); cb(p); };
  const work = AutoModelForCausalLM.from_pretrained(
    profile.path, { ...opts, progress_callback: wrapped }
  ).catch((e) =>
    // 本地缺失时回退 Hub 同名仓库 (同 dtype)
    AutoModelForCausalLM.from_pretrained(REPO_FALLBACK, { ...opts, progress_callback: wrapped })
  );
  return new Promise((resolve, reject) => {
    const idle = setInterval(() => {
      if (Date.now() - last > 90000) {
        cleanup(); reject(new Error("模型加载停滞超时 (90s 无进度)"));
      }
    }, 5000);
    const total = setTimeout(() => { cleanup(); reject(new Error("模型加载总超时 (300s)")); }, 300000);
    function cleanup() { clearInterval(idle); clearTimeout(total); }
    work.then(
      (v) => { cleanup(); resolve(v); },
      (e) => { cleanup(); reject(e); }
    );
  });
}

async function init() {
  els.btn.disabled = true;
  let profile = await pickProfile();
  degraded = profile.degraded;
  const showMode = () => {
    if (degraded) {
      els.mode.textContent = "快速模式 (int4 量化, 约 500MB): 当前浏览器无 WebGPU, 概率可能有量化偏差; 用最新 Chrome/Edge 打开可获高精度";
    } else {
      els.mode.textContent = "高精度模式 (fp16 约 1.2GB, WebGPU)";
    }
    els.mode.style.display = "block";
  };
  showMode();
  const cb = (p) => {
    if (p.status === "progress" && p.total) {
      setProgress(p.progress / 100, `下载模型权重 ${String(p.file || "").split("/").pop()} ${(p.progress || 0).toFixed(0)}%`);
    } else if (p.status === "done") {
      setProgress(1, "权重就绪, 编译计算图 ...");
    }
  };
  console.log("[openjev] profile:", JSON.stringify(profile));
  setStatus("加载 tokenizer ...");
  tokenizer = await AutoTokenizer.from_pretrained(profile.path).catch(() =>
    AutoTokenizer.from_pretrained(REPO_FALLBACK)
  );
  setStatus(`加载模型 (${profile.dtype}, ${profile.degraded ? "约 500MB" : "约 1.2GB"}, 首次会缓存) ...`);
  try {
    model = await loadModel(profile, cb);
  } catch (e) {
    if (!profile.degraded) {
      // 高精度档加载失败/挂起超时 -> 回退量化档
      console.warn("fp16 profile failed, fallback to q4f16:", e);
      degraded = true;
      showMode();
      profile = { path: "./model_q4", dtype: "q4f16", device: "wasm", degraded: true };
      setStatus("高精度模式不可用, 回退量化模式 ...");
      model = await loadModel(profile, cb);
    } else {
      throw e;
    }
  }
  setProgress(1, `就绪 — 输入消息后点击「判一判」`);
  ready = true;
  els.btn.disabled = false;
}

const fmtPct = (v) => (v * 100).toFixed(1) + "%";
const bar = (label, p, cls) =>
  `<div class="bar-row"><span class="bar-label">${label}</span>` +
  `<div class="bar-track"><div class="bar-fill ${cls}" style="width:${(p * 100).toFixed(1)}%"></div></div>` +
  `<span class="bar-val">${fmtPct(p)}</span></div>`;

function renderAnswers(ans) {
  const S = "没有敌意, 正常交流|有点不爽, 但还在忍|明显不满, 讽刺意味浓|敌意拉满, 就差骂人了".split("|");
  const yn = ans.yin_yang.noul;
  const yang = yn >= 0.5;
  let html = "";
  html += `<div class="card"><h3>① 阴阳怪气判定</h3>`;
  html += `<div class="verdict ${yang ? "bad" : "ok"}">${yang ? "是, 阴阳怪气" : "不是, 正常话"}</div>`;
  html += bar("Yes", yn, yang ? "fill-bad" : "fill-ok") + bar("No", 1 - yn, "") + `</div>`;

  const h = ans.hostility;
  const idx = Object.keys(h.probabilities).reduce((bi, k, i, a) => (h.probabilities[k] > h.probabilities[a[bi]] ? i : bi), 0);
  html += `<div class="card"><h3>② 敌意程度 — ${h.score.toFixed(2)} / 3</h3>`;
  for (let i = 0; i < 4; i++) html += bar(`${i} = ${S[i]}`, h.probabilities[String(i)], i === idx ? "fill-bad" : "");
  html += `<p class="conf">置信度 ${(h.confidence * 100).toFixed(1)}%</p></div>`;

  const v = ans.vibe;
  const vi = O.VIBE_NAMES.indexOf(v.choice);
  const V = ["真诚的, 表面和实际一致", "被动攻击, 用客气包裹不满", "纯讽刺, 明摆着阴阳怪气"];
  html += `<div class="card"><h3>③ 真实语气 — ${V[vi]}</h3>`;
  for (let i = 0; i < 3; i++) html += bar(V[i], v.probabilities[O.VIBE_NAMES[i]], i === vi ? "fill-bad" : "");
  html += `<p class="conf">置信度 ${(v.confidence * 100).toFixed(1)}%</p></div>`;
  return html;
}

async function run() {
  if (!ready) return;
  const text = els.input.value.trim();
  if (!text) { setStatus("先输入一条消息"); return; }
  els.btn.disabled = true;
  setStatus("推理中 (一次前向, 秒级) ...");
  try {
    // 三道题的 prompt -> 一次 batch 前向
    const prompts = ["yin_yang", "hostility", "vibe"].map((qid) =>
      O.buildPrompt(text, O.QUESTIONS[qid], O.DEFAULT_SYSTEM_PROMPT)
    );
    const inputs = tokenizer(prompts, { add_special_tokens: false, padding: true });
    const out = await model(inputs);
    const [B, S, V] = out.logits.dims;
    const data = out.logits.data;
    const rows = [];
    for (let b = 0; b < B; b++) {
      // last non-pad position
      let last = S - 1;
      const m = inputs.attention_mask ? inputs.attention_mask.data : null;
      if (m) { while (last > 0 && m[b * S + last] === 0) last--; }
      const start = (b * S + last) * V;
      rows.push(data.subarray(start, start + V));
    }
    const ans = O.answerAll(rows[0], rows[1], rows[2]);
    els.results.innerHTML = renderAnswers(ans);
    setStatus("完成");
  } catch (e) {
    console.error(e);
    setStatus("出错: " + e.message);
  } finally {
    els.btn.disabled = false;
  }
}

els.btn.addEventListener("click", run);
init().catch((e) => { console.error(e); setStatus("初始化失败: " + e.message); });
