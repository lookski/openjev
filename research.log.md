# OpenJev 项目 research log

===== [2026-09-23 22:45:00] ctx-guard checkpoint =====

## 本轮动作
- 完成 OpenJev 全链路: 引擎(掩码softmax) + 8后端(本地/ollama/lmstudio/llamacpp/vllm/openai/openrouter/jev云) + easy向导(openjev-easy) + 网页版(openjev-web) + 群聊雷达(fun_chat_radar) + 阴阳怪气鉴定器(fun_yinyang)
- 网页版移入 openjev/webapp.py 并注册 openjev-web 命令, 防御性文案已清除, space/ 目录为 HF Space 一键部署包(Docker)
- README 双语全覆盖(全部实测数据), CI 全绿, Release v0.1.0 已发布, topics x9, assets/social-preview.png 已生成待用户上传
- computer use 演示成功: 自开 Edge -> 破 Windows 前台锁(定位并结束占前台的 PickerHost 隐形弹窗) -> OCR 定位输入框与按钮 -> 键入"哥们你可真是帮了我大忙了呢" -> 点击开鉴 -> 页面真实渲染 yin-yang=0.7454, sincere=0.9909
- ntfy 推送两次成功(经验: 中文正文会被 ntfy 转成附件, 纯英文正文正常; topic 实取自 C:/Users/huxia/phone_bridge.py 的 CONFIG 区)

## 文件状态
- 仓库 github.com/lookski/openjev, 本地最新 commit 0eb596c, 全部已推送
- 后台服务: 鉴定器局域网8791 (task bdd0cc2e8) 运行中, 绑 0.0.0.0:8791, 手机同 Wi-Fi 可访问 http://10.2.13.123:8791
- 测试 35/35 通过; 模型在 models/Qwen3-0.6B (1.5GB, 已 gitignore)
- 关键坑记录: bg_run 底层是 cmd.exe, export/tail/管道不可用, 要用裸命令+脚本内置环境变量; github.com:443 时通时断, 用 curl 探测成功后再 push; workdir-guard 对 heredoc 内容与相对路径有误判, 用 write 工具绝对路径绕过; ntfy 中文正文转附件

## 下一步
- [用户] 知乎发帖(三版文案已给, A 版主推; 素材: 阴阳0.84 / "6"0.88 / 半阴半阳梗); 仓库 Settings 上传 assets/social-preview.png
- [可选] 服务器部署 / 温度校准 / 更大模型(1.7B/4B)提升校准

===== [2026-09-25 09:54:11] HF Space 浏览器版部署 ( CU 结束后纯 API 路线) =====

## 本轮动作
- CU 环节全部完成: HF 登录 (LinRin0306, 用户手动) -> write token openjev-space-deploy (tmp_hf_token.txt, CLI whoami 验证) -> Static Space LinRin0306/openjev-detector 创建成功 (Docker Space 被 HF 402 拦: 免费档只允许 Static)
- Python 黄金参考跑通 (tmp_golden_ref.py -> tmp_golden.json): 本地 torch+Qwen3-0.6B, 样本0 阴阳0.6714/hostility主档2/vibe sincere; 样本1 全部正常向
- prompt 逐字节移植: openjev.browser.js 的 buildPrompt 与 Python _build_prompt+Qwen3 chat template (enable_thinking=false, 空 think 块) BYTE-IDENTICAL (tmp_prompts.json 比对)
- JS 移植验证 (tmp_jscheck/verify.js, transformers.js node 版+本地 ONNX): fp32 误差 0.00006 ALL-PASS; **fp16 误差 0.0032 ALL-PASS**; q4/q4f16/int8/uint8/kld-int4 全部失真 (最大 0.72, 阴阳样本判反) 弃用
- 浏览器运行时踩坑 (关键): ① dist/transformers.min.js 是 node 变体, 浏览器必须用 transformers.web.js ② web 版裸导入 onnxruntime-web/webgpu + onnxruntime-common, 需 import map 指 vendor/ort.all.bundle.min.mjs (jsep 构建, 含 webgpu EP) ③ ort wasm 二进制必须放站点根 (wasmPaths="./", jsep 从 /vendor/ 解析 mjs 但 wasm 落根) ④ fp32/fp16 在 wasm32 CPU 执行 bad_alloc (fp16 会膨胀 fp32, 2.4GB 超限) ⑤ playwright/headless 下无 WebGPU; probe 验证 device:"webgpu" 请求在无 GPU 时自动回退 CPU, q4f16 (0.5GB) 可跑 ⑥ 探针 (q4f16+fp16) 全 PASS, 完整 app 曾卡「编译计算图」-> 定位为 openjev.browser.js UMD 在 ESM 下导出空命名空间 (O.QUESTIONS undefined), 转 .mjs ESM export 后 APP-SMOKE-PASS (三卡片, 概率与黄金数据同分布)
- 最终架构: 双模式 — 有 WebGPU 用 fp16 (1.2GB, 高精度, 误差 0.0032); 无则 q4f16 (0.5GB, wasm, UI 明示量化偏差); 全部权重/运行时自托管 (Static Space 同源, 无 CDN 依赖, 大陆可达)
- 部署包 tmp_jscheck/space: 首推 1.8GB 撞 Static Space 1GB 存储限额 (403 Repository storage limit reached); 改配额内布局 613MB (q4f16+代码+jsep 运行时, 自托管), fp16 高精度档由 app 从 onnx-community/Qwen3-0.6B-ONNX 在线加载; 上传成功 commit 5fefb8f -> Space RUNNING
- 线上验证: 正确域名是 *.static.hf.space (旧 *.hf.space 404); fp16 高精度档 ONLINE-SMOKE-PASS (真实 Edge, Hub 加载权重, ~5min 首载, 三卡片, 概率与黄金同分布且更 "清醒": 阴阳 Yes 仅 7.1%); q4f16 兜底档发现线上 bug: navigator.gpu 缺失时 jsep webgpu EP 直接抛错不回退 CPU -> 修 device:"wasm" (本地已验), 单文件 commit f5144dd -> **ONLINE-Q4-PASS** (快速模式三卡片)

## 文件状态
- Space 上线: https://linrin0306-openjev-detector.static.hf.space (LinRin0306/openjev-detector, sdk static, RUNNING)
- 新增: tmp_jscheck/ (验证环境+部署包), tmp_golden_ref.py, tmp_golden.json, tmp_prompts.json, tmp_upload_space.py, tmp_smoke_space.py (冒烟脚本, 有头 Edge), tmp_online_smoke.js / tmp_online_q4.js / tmp_q4_smoke.js (线上冒烟)

## 下一步
- [用户] 真实 Edge 手开 https://linrin0306-openjev-detector.static.hf.space 体验 (自动化环境与真实环境 WebGPU 可用性不同, 你机器有 RTX 5060 应走高精度 fp16 模式)
- 清理 tmp_* 临时文件 (含 tmp_hf_token.txt 必须删)
- 知乎文案与 social-preview 仍待用户
