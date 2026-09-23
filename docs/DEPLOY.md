# 部署指南 / Deployment

English summary below each section.

## 1. 直接跑 / Bare metal (pip)

```bash
git clone https://github.com/lookski/openjev.git
cd openjev
pip install -e .

# CN mirror for the model download (script has this built in)
python scripts/download_model.py          # default Qwen/Qwen3-0.6B -> models/
python demo.py
```

前台快速验证 / quick check:

```bash
python -m openjev.cli ask \
  --state "server down, losing money" \
  --question '{"urgent":{"type":"noul","instructions":"Is this urgent?"}}'
```

## 2. 常驻服务 / systemd (Linux)

`/etc/systemd/system/openjev.service`:

```ini
[Unit]
Description=OpenJev local decision engine
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/openjev
ExecStart=/opt/openjev/.venv/bin/python -m openjev.server --host 127.0.0.1 --port 8771 --model /opt/openjev/models/Qwen3-0.6B --preload
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now openjev
curl http://127.0.0.1:8771/health
```

Nginx 反代 (局域网/内网):

```nginx
location /systemone/ {
    proxy_pass http://127.0.0.1:8771/v1/systemone;
    proxy_read_timeout 300s;
}
```

## 3. Docker (推荐 / recommended)

```bash
docker compose up -d openjev          # local backend, port 8771
docker compose --profile cloud up -d openjev-cloud   # Jev cloud proxy, port 8772
curl http://127.0.0.1:8771/health
```

- 模型权重落在 named volume `openjev-models`, 首次启动自动下载, 之后离线可用
- torch 安装的是 CPU 轮子, 镜像约 2 GB
- `openjev-cloud` 需要先 `export TYPESAFE_API_KEY=...`

## 4. 云端 Jev 后端 / Real Jev API backend

OpenJev 自带双后端, 接口完全一致, 一行切换:

```bash
# 本地引擎 (默认, 免费离线)
openjev ask --state "..." --question '{"urgent":{"type":"noul","instructions":"urgent?"}}'

# 官方 Jev 云 API (需要 TYPESAFE_API_KEY)
export TYPESAFE_API_KEY=sk-...
openjev ask --backend jev --state "..." --question '{"urgent":{"type":"noul","instructions":"urgent?"}}'
```

服务端同样支持:

```bash
openjev serve --backend jev     # 成为 Jev 云 API 的同构代理网关
```

库 API:

```python
from openjev import Choice, LocalJev, RemoteJev

# same interface, switch by one constructor
local = LocalJev("Qwen/Qwen3-0.6B")
cloud = RemoteJev()                      # reads TYPESAFE_API_KEY

state = "The server is down, we are losing money."
questions = {"urgent": Choice(instructions="...", criteria={"a": "x", "b": "y"})}
print(local.system_one(state, questions)["answers"])
print(cloud.system_one(state, questions)["answers"])   # A/B 对比评测
```

Remote 客户端特性: 从 `$TYPESAFE_API_KEY` 读密钥, 429/5xx 按 `retry-after` 自动退避重试, 透传官方 `usage`.

## 5. GPU / 加速

```bash
# NVIDIA: 安装 CUDA 版 torch 后
openjev serve --device cuda --dtype float16 --model Qwen/Qwen3-4B-Instruct-2507
# Apple Silicon: --device mps
```

决策引擎单次前向, TTFT 即全部延迟: GPU 上 0.6B 约 10–30 ms, 4B 约 50–150 ms.
