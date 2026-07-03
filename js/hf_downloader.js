/**
 * HF Model Downloader — Extensão JavaScript para ComfyUI
 * 
 * Escaneia o workflow automaticamente ao criar o nó,
 * detecta modelos referenciados e verifica faltantes.
 * Agora com suporte a download por URL manual do HuggingFace!
 * Exibe porcentagem de progresso no título do nó.
 * Botão de cancelar para interromper downloads em andamento!
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Extensões de arquivo de modelo
const MODEL_FILE_EXTENSIONS = [".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".onnx", ".h5", ".pb"];

// Categorias disponíveis para download
const CATEGORIES = [
    "checkpoint", "lora", "vae", "clip", "controlnet",
    "embeddings", "upscaler", "ipadapter", "style_models",
    "clip_vision", "gligen", "diffusion_models", "photomaker",
    "insightface", "animatediff_models",
];

// Mapeamento: tipo de nó → categoria (pasta no ComfyUI)
const NODE_CATEGORY = {
    "CheckpointLoader": "checkpoint", "CheckpointLoaderSimple": "checkpoint", "unCLIPCheckpointLoader": "checkpoint",
    "LoraLoader": "lora", "LoraLoaderModelOnly": "lora", "LoraLoaderBlockWeight": "lora",
    "VAELoader": "vae", "VAELoaderAdvanced": "vae",
    "CLIPLoader": "clip", "DualCLIPLoader": "clip", "TripleCLIPLoader": "clip",
    "ControlNetLoader": "controlnet", "ControlNetLoaderAdvanced": "controlnet", "DiffControlNetLoader": "controlnet",
    "UpscaleModelLoader": "upscaler", "ImageUpscaleWithModel": "upscaler",
    "IPAdapterModelLoader": "ipadapter", "StyleModelLoader": "style_models",
    "CLIPVisionLoader": "clip_vision", "GLIGENLoader": "gligen", "UNETLoader": "diffusion_models",
    "PhotoMakerLoader": "photomaker", "InsightFaceLoader": "insightface",
    "AnimateDiffLoader": "animatediff_models", "AnimateDiffLoaderV1": "animatediff_models",
    "AnimateDiffLoraLoader": "animatediff_motion_lora", "AnimateDiffMotionLoraLoader": "animatediff_motion_lora",
    "HypernetworkLoader": "embeddings",
};

// Widgets que contêm modelos em cada tipo de nó
const NODE_WIDGETS = {
    "CheckpointLoader": ["ckpt_name"], "CheckpointLoaderSimple": ["ckpt_name"], "unCLIPCheckpointLoader": ["ckpt_name"],
    "LoraLoader": ["lora_name"], "LoraLoaderModelOnly": ["lora_name"], "LoraLoaderBlockWeight": ["lora_name"],
    "VAELoader": ["vae_name"], "VAELoaderAdvanced": ["vae_name"],
    "CLIPLoader": ["clip_name"], "DualCLIPLoader": ["clip_name1", "clip_name2"], "TripleCLIPLoader": ["clip_name1", "clip_name2", "clip_name3"],
    "ControlNetLoader": ["control_net_name"], "ControlNetLoaderAdvanced": ["control_net_name"], "DiffControlNetLoader": ["control_net_name"],
    "UpscaleModelLoader": ["model_name"], "ImageUpscaleWithModel": ["model_name"],
    "IPAdapterModelLoader": ["model_name"], "StyleModelLoader": ["model_name"],
    "CLIPVisionLoader": ["clip_name"], "GLIGENLoader": ["model_name"], "UNETLoader": ["unet_name"],
    "PhotoMakerLoader": ["model_name", "photomaker_model"], "InsightFaceLoader": ["model_name", "insightface"],
    "AnimateDiffLoader": ["model_name"], "AnimateDiffLoaderV1": ["model_name"],
    "AnimateDiffLoraLoader": ["lora_name"], "AnimateDiffMotionLoraLoader": ["lora_name"],
    "HypernetworkLoader": ["model_name"],
};

const STORAGE_KEY = "hf_node_models_path";

const appInstance = app;

const extension = {
    name: "HFModelDownloader",

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== "HFModelDownloader") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;

            // Timer do polling de download (cancelado se o nó for destruído)
            this._hfPollTimer = null;
            this._hfCurrentJobId = null;

            // ════════════════════════════════════════════════
            // SEÇÃO: Download por URL Manual do HuggingFace
            // ════════════════════════════════════════════════

            // Widget de texto para colar a URL do modelo
            this.addWidget("text", "🔗 URL do HF:", "", (v) => {
                this._hfUrlValue = v;
            });
            // Widget de seleção de categoria (pasta de destino)
            this.addWidget("combo", "📁 Pasta destino:", "checkpoint", (v) => {
                this._hfUrlCategory = v;
            }, { values: CATEGORIES });

            // Botão "Baixar URL"
            const urlBtn = this.addWidget("button", "📎 Baixar URL", "url_download", () => {
                this._hfDownloadUrl();
            });
            urlBtn.serialize = false;

            // ════════════════════════════════════════════════
            // CORREÇÃO #4: Botão Cancelar Download
            // ════════════════════════════════════════════════
            const cancelBtn = this.addWidget("button", "⛔ Cancelar", "cancel", () => {
                this._hfCancelDownload();
            });
            cancelBtn.serialize = false;

            // ════════════════════════════════════════════════
            // SEÇÃO: Botões Originais
            // ════════════════════════════════════════════════

            // Botão de selecionar pasta models
            const folderBtn = this.addWidget("button", "📂 Pasta Models", "folder", () => this._hfSelectFolder());
            folderBtn.serialize = false;

            // Botão Scan
            const scanBtn = this.addWidget("button", "🔍 Scan", "scan", () => {
                this._hfSetStatus("🔍 Escaneando...", "#1a73e8");
                this._hfAutoScan();
            });
            scanBtn.serialize = false;

            // Botão Download (scan automático)
            const dlBtn = this.addWidget("button", "📥 Download", "download", () => {
                const mlWidget = this.widgets.find(w => w.name === "model_list");
                const list = mlWidget ? mlWidget.value : "";
                if (!list || !list.trim()) {
                    // Tenta fazer scan primeiro
                    this._hfSetStatus("🔍 Escaneando antes do download...", "#1a73e8");
                    this._hfAutoScan();
                    return;
                }
                this._hfSetStatus("📥 Iniciando download...", "#1a73e8");
                this._hfDownload(list);
            });
            dlBtn.serialize = false;

            // Botão Login
            const loginBtn = this.addWidget("button", "🔑 Login", "login", () => this._hfLogin());
            loginBtn.serialize = false;

            // Carrega pasta salva + pré-scan automático
            setTimeout(() => {
                this._hfRestoreFolder();
                this._hfAutoScan();
                this._hfLoadStatus();
            }, 800);

            return result;
        };

        // Limpa polling ao destruir nó
        const origOnRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            if (this._hfPollTimer) {
                clearInterval(this._hfPollTimer);
                this._hfPollTimer = null;
            }
            return origOnRemoved ? origOnRemoved.apply(this, arguments) : undefined;
        };

        // ==================== CANCELAR DOWNLOAD ====================

        nodeType.prototype._hfCancelDownload = async function () {
            const jobId = this._hfCurrentJobId;
            if (!jobId) {
                this._hfSetStatus("⚠️ Nenhum download em andamento", "#f0a500");
                return;
            }

            this._hfSetStatus("⛔ Cancelando...", "#f0a500");

            try {
                const resp = await api.fetchApi(`/hf_node/cancel/${jobId}`, {
                    method: "POST",
                });

                if (resp.ok) {
                    const data = await resp.json();
                    if (data.success) {
                        this._hfSetStatus("⛔ Download cancelado", "#f0a500");
                        this.title = "🤗 HF ⛔";
                        this.color = "#4a3a15"; this.bgcolor = "#3a2a10";

                        // Para o polling
                        if (this._hfPollTimer) {
                            clearInterval(this._hfPollTimer);
                            this._hfPollTimer = null;
                        }
                        this._hfCurrentJobId = null;
                    } else {
                        this._hfSetStatus(`⚠️ ${data.message || "Erro ao cancelar"}`, "#f0a500");
                    }
                } else {
                    this._hfSetStatus("⚠️ Erro ao chamar cancelamento", "#f0a500");
                }
            } catch (err) {
                this._hfSetStatus("⚠️ Erro de rede ao cancelar", "#f0a500");
                console.warn("[HF Node] Cancel error:", err);
            }
        };

        // ==================== DOWNLOAD POR URL MANUAL ====================

        nodeType.prototype._hfDownloadUrl = async function () {
            // Pega o valor da URL do widget
            const urlWidget = this.widgets.find(w => w.name === "🔗 URL do HF:");
            const url = urlWidget ? urlWidget.value.trim() : "";

            if (!url) {
                this._hfSetStatus("⚠️ Cole uma URL do HuggingFace primeiro!", "#f0a500");
                return;
            }

            this._hfSetStatus("📥 Baixando via URL...", "#1a73e8");

            try {
                const resp = await api.fetchApi("/hf_node/download_url", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        url: url,
                        category: this._hfUrlCategory || "checkpoint",
                        models_path: this._hfGetStoredPath() || "",
                    }),
                });

                if (!resp.ok) {
                    const errData = await resp.json().catch(() => ({}));
                    this._hfSetStatus(`❌ ${errData.message || "Erro"}`, "#f44336");
                    return;
                }

                const data = await resp.json();
                const jobId = data.job_id;

                if (!jobId) {
                    this._hfSetStatus("❌ Erro: job_id não retornado", "#f44336");
                    return;
                }

                this._hfCurrentJobId = jobId;
                this._hfSetStatus(`📥 Baixando... ${data.message || ""}`, "#1a73e8");
                this.title = "🤗 HF 📥 0%";
                this.color = "#15354a"; this.bgcolor = "#10253a";

                // Inicia polling
                if (this._hfPollTimer) {
                    clearInterval(this._hfPollTimer);
                }
                this._hfPollTimer = setInterval(() => {
                    this._hfPollDownloadStatus(jobId);
                }, 1500);
                setTimeout(() => {
                    this._hfPollDownloadStatus(jobId);
                }, 500);

            } catch (err) {
                this._hfSetStatus("⚠️ Erro de rede", "#f0a500");
                console.warn("[HF Node] URL download error:", err);
            }
        };

        // ==================== PASTA DE MODELOS ====================
        nodeType.prototype._hfCreatePathWidget = function () {
            let w = this.widgets.find(w => w.name === "_hf_path_display");
            if (w) return w;

            const saved = this._hfGetStoredPath();
            w = this.addWidget("text", "📁 Models:", saved || "(automático)", () => {});
            w.name = "_hf_path_display";
            w.serialize = false;
            // Torna read-only após renderizar
            setTimeout(() => {
                if (w.inputEl) {
                    w.inputEl.readOnly = true;
                }
            }, 100);
            return w;
        };

        nodeType.prototype._hfUpdatePathWidget = function (path) {
            let w = this.widgets.find(w => w.name === "_hf_path_display");
            if (!w) {
                w = this._hfCreatePathWidget();
            }
            w.value = path || "(automático)";
        };

        nodeType.prototype._hfSelectFolder = function () {
            const current = this._hfGetStoredPath();
            const input = prompt(
                "📁 Caminho da pasta 'models' do ComfyUI:\n" +
                "Cole o caminho completo ou deixe vazio para detecção automática.\n\n" +
                "Exemplo: E:\\ComfyUI-Portable_Tensor_RT\\ComfyUI\\models",
                current
            );
            if (input === null) return; // cancelou

            if (input.trim() === "") {
                localStorage.removeItem(STORAGE_KEY);
                this._hfUpdatePathWidget("");
                this._hfSetStatus("🔄 Pasta automática ativada", "#1a73e8");
            } else {
                localStorage.setItem(STORAGE_KEY, input.trim());
                this._hfUpdatePathWidget(input.trim());
                this._hfSetStatus(`📂 Pasta salva: ${input.trim()}`, "#4caf50");
            }
        };

        nodeType.prototype._hfGetStoredPath = function () {
            return localStorage.getItem(STORAGE_KEY) || "";
        };

        nodeType.prototype._hfRestoreFolder = function () {
            const saved = this._hfGetStoredPath();
            if (saved) {
                this._hfUpdatePathWidget(saved);
                this._hfSetStatus(`📂 Pasta: ${saved}`, "#4caf50");
            } else {
                this._hfCreatePathWidget();
            }
        };

        // ==================== AUTO SCAN ====================
        nodeType.prototype._hfAutoScan = async function () {
            // Evita scans paralelos
            if (this._hfScanning) return;
            this._hfScanning = true;

            // Limpa model_list ANTES de qualquer operação (evita acúmulo)
            const mlWidget = this.widgets.find(w => w.name === "model_list");
            if (mlWidget) mlWidget.value = "";

            try {
                const models = this._collectModels();
                if (models.length === 0) {
                    this._hfSetStatus("ℹ️ Nenhum modelo no workflow", "#f0a500");
                    this._hfScanning = false;
                    return;
                }

                const modelStr = models.map(m => `${m.cat}:${m.name}`).join("\n");
                if (mlWidget) mlWidget.value = modelStr;

                const resp = await api.fetchApi("/hf_node/scan", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        model_list: modelStr,
                        models_path: this._hfGetStoredPath() || "",
                    }),
                });

                if (resp.ok) {
                    const data = await resp.json();
                    const missing = data.missing || [];
                    const total = data.total || models.length;

                    const missingStr = missing.map(m => `${m.category}:${m.filename}`).join("\n");
                    if (mlWidget) mlWidget.value = missingStr;

                    if (missing.length === 0) {
                        this._hfSetStatus(`✅ Todos os ${total} modelos OK`, "#4caf50");
                        this.title = "🤗 HF ✅";
                        this.color = "#154a15"; this.bgcolor = "#103a10";
                    } else {
                        let msg = `🔴 ${missing.length} faltante(s) de ${total}\n`;
                        missing.forEach(m => {
                            msg += `\n❌ ${m.filename}`;
                            msg += `\n   📁 Destino: models\\${m.folder}\\`;
                        });
                        msg += "\n\nClique em 📥 Download";
                        this._hfSetStatus(msg, "#f44336");
                        this.title = `🤗 HF 🔴 ${missing.length}`;
                        this.color = "#4a1515"; this.bgcolor = "#3a1010";
                    }
                    console.log(`[HF Node] Scan: ${total} total, ${missing.length} missing`);
                }
            } catch (err) {
                console.warn("[HF Node] Scan error:", err);
            } finally {
                this._hfScanning = false;
            }
        };

        // ==================== DOWNLOAD EM BACKGROUND (POLLING) ====================
        nodeType.prototype._hfDownload = async function (modelList) {
            try {
                // 1. Inicia o job de download em background
                const resp = await api.fetchApi("/hf_node/download_start", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        model_list: modelList,
                        models_path: this._hfGetStoredPath() || "",
                    }),
                });

                if (!resp.ok) {
                    const errData = await resp.json().catch(() => ({}));
                    this._hfSetStatus(`❌ ${errData.message || "Erro ao iniciar download"}`, "#f44336");
                    return;
                }

                const data = await resp.json();
                const jobId = data.job_id;
                const total = data.total || 0;

                if (!jobId) {
                    this._hfSetStatus("❌ Erro: job_id não retornado", "#f44336");
                    return;
                }

                this._hfCurrentJobId = jobId;
                this._hfSetStatus(`📥 Download iniciado (job ${jobId})`, "#1a73e8");
                this.title = `🤗 HF 📥 0%`;
                this.color = "#15354a"; this.bgcolor = "#10253a";

                // 2. Polling de progresso a cada 1.5 segundos
                if (this._hfPollTimer) {
                    clearInterval(this._hfPollTimer);
                }

                this._hfPollTimer = setInterval(() => {
                    this._hfPollDownloadStatus(jobId);
                }, 1500);

                // Faz uma consulta imediata também
                setTimeout(() => {
                    this._hfPollDownloadStatus(jobId);
                }, 500);

            } catch (err) {
                this._hfSetStatus("⚠️ Erro de rede ao iniciar download", "#f0a500");
                console.warn("[HF Node] Download start error:", err);
            }
        };

        nodeType.prototype._hfPollDownloadStatus = async function (jobId) {
            try {
                const resp = await api.fetchApi(`/hf_node/download_status/${jobId}`, {
                    method: "GET",
                });

                if (!resp.ok) {
                    // Se a rota não existe (versão antiga), cai fora
                    if (resp.status === 404) {
                        this._hfSetStatus("⚠️ Rota de progresso não disponível", "#f0a500");
                        if (this._hfPollTimer) {
                            clearInterval(this._hfPollTimer);
                            this._hfPollTimer = null;
                        }
                        return;
                    }
                    return;
                }

                const data = await resp.json();
                const status = data.status || "unknown";
                const progress = data.progress || "0/0";
                const current = data.current || "";
                const total = data.total || 0;
                const completed = data.completed || 0;
                const results = data.results || [];
                const bytesDownloaded = data.bytes_downloaded || 0;
                const bytesTotal = data.bytes_total || 0;
                const existed = data.existed || false;

                // ========== CALCULA PORCENTAGEM ==========
                let percent = 0;

                // Prioridade 1: progresso por bytes (URL download)
                if (bytesTotal > 0) {
                    percent = Math.round((bytesDownloaded / bytesTotal) * 100);
                }
                // Prioridade 2: progresso por modelos completados
                else if (total > 0) {
                    percent = Math.round((completed / total) * 100);
                }
                // Prioridade 3: job finalizado sem bytes
                else if (status === "done" || status === "exists") {
                    percent = 100;
                }

                // ========== ATUALIZA TÍTULO COM % ==========
                let titleText = "";
                let titleColor = "#1a73e8";
                let bgColor = "#1a73e8";

                switch (status) {
                    case "starting":
                        titleText = "🤗 HF ⏳";
                        titleColor = "#1a73e8";
                        bgColor = "#1a73e8";
                        break;
                    case "running":
                        titleText = `🤗 HF 📥 ${percent}%`;
                        titleColor = "#15354a";
                        bgColor = "#10253a";
                        break;
                    case "cancelled":
                        titleText = "🤗 HF ⛔";
                        titleColor = "#4a3a15";
                        bgColor = "#3a2a10";
                        break;
                    case "done": {
                        const ok = results.filter(r => r.status === "ok").length;
                        const fail = results.length - ok;
                        titleText = ok > 0 ? "🤗 HF ✅" : "🤗 HF ⚠️";
                        titleColor = ok > 0 ? "#154a15" : "#4a3a15";
                        bgColor = ok > 0 ? "#103a10" : "#3a2a10";
                        break;
                    }
                    case "exists": {
                        titleText = "🤗 HF ⚠️";
                        titleColor = "#4a3a15";
                        bgColor = "#3a2a10";
                        break;
                    }
                    case "failed":
                        titleText = "🤗 HF ❌";
                        titleColor = "#4a1515";
                        bgColor = "#3a1010";
                        break;
                    case "not_found":
                        titleText = "🤗 HF ⚠️";
                        titleColor = "#4a3a15";
                        bgColor = "#3a2a10";
                        break;
                    default:
                        titleText = `🤗 HF ${progress}`;
                        break;
                }

                this.title = titleText;
                this.color = titleColor;
                this.bgcolor = bgColor;

                // ========== ATUALIZA WIDGET DE STATUS ==========
                let statusMsg = "";
                let statusColor = "#1a73e8";

                switch (status) {
                    case "starting":
                        statusMsg = "⏳ Preparando...";
                        statusColor = "#1a73e8";
                        break;
                    case "running":
                        statusMsg = `📥 ${percent}% — ${current}`;
                        statusColor = "#1a73e8";
                        break;
                    case "cancelled":
                        statusMsg = "⛔ Download cancelado pelo usuário";
                        statusColor = "#f0a500";
                        if (this._hfPollTimer) {
                            clearInterval(this._hfPollTimer);
                            this._hfPollTimer = null;
                        }
                        this._hfCurrentJobId = null;
                        break;
                    case "done": {
                        const ok = results.filter(r => r.status === "ok").length;
                        const fail = results.length - ok;
                        statusMsg = `✅ ${ok} sucesso`;
                        if (fail > 0) statusMsg += `, ${fail} falha(s)`;
                        statusMsg += "\n";
                        results.forEach(r => {
                            if (r.status === "ok") {
                                statusMsg += `\n✅ ${r.filename} → models\\${r.folder}\\`;
                            } else {
                                statusMsg += `\n❌ ${r.filename}`;
                            }
                        });
                        statusColor = ok > 0 ? "#4caf50" : "#f0a500";
                        // Para o polling
                        if (this._hfPollTimer) {
                            clearInterval(this._hfPollTimer);
                            this._hfPollTimer = null;
                        }
                        this._hfCurrentJobId = null;
                        break;
                    }
                    case "exists": {
                        statusMsg = `⚠️ Modelo já existe em disco`;
                        results.forEach(r => {
                            if (r.status === "exists") {
                                statusMsg += `\n⚠️ ${r.filename} → models\\${r.folder}\\`;
                            }
                        });
                        statusColor = "#f0a500";
                        if (this._hfPollTimer) {
                            clearInterval(this._hfPollTimer);
                            this._hfPollTimer = null;
                        }
                        this._hfCurrentJobId = null;
                        break;
                    }
                    case "failed":
                        statusMsg = `❌ ${data.error || "Falha no download"}`;
                        statusColor = "#f44336";
                        if (this._hfPollTimer) {
                            clearInterval(this._hfPollTimer);
                            this._hfPollTimer = null;
                        }
                        this._hfCurrentJobId = null;
                        break;
                    case "not_found":
                        statusMsg = "⚠️ Job não encontrado (expirou?)";
                        statusColor = "#f0a500";
                        if (this._hfPollTimer) {
                            clearInterval(this._hfPollTimer);
                            this._hfPollTimer = null;
                        }
                        this._hfCurrentJobId = null;
                        break;
                    default:
                        statusMsg = `📥 ${percent}%`;
                        break;
                }

                this._hfSetStatus(statusMsg, statusColor);

                // Se terminou ou já existia, faz scan automático pra atualizar o nó
                if (status === "done" || status === "exists") {
                    setTimeout(() => {
                        this._hfAutoScan();
                    }, 1000);
                }

            } catch (err) {
                console.warn("[HF Node] Poll error:", err);
            }
        };

        // ==================== LOGIN ====================
        nodeType.prototype._hfLogin = async function () {
            const tokenWidget = this.widgets.find(w => w.name === "hf_token");
            const token = tokenWidget ? tokenWidget.value : "";
            if (!token || token.startsWith("(salvo") || token.startsWith("já logado")) {
                this._hfSetStatus("⚠️ Digite o token HF no campo hf_token!", "#f0a500");
                return;
            }

            this._hfSetStatus("🔑 Verificando...", "#1a73e8");

            try {
                const resp = await api.fetchApi("/hf_node/login", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ token: token.trim() }),
                });

                if (resp.ok) {
                    const data = await resp.json();
                    if (data.success) {
                        this._hfSetStatus(`🟢 Logado: ${data.username}`, "#4caf50");
                        this.title = "🤗 HF 🟢";
                        this.color = "#154a15"; this.bgcolor = "#103a10";
                        if (tokenWidget) tokenWidget.value = "(salvo — " + data.username + ")";
                    } else {
                        this._hfSetStatus(`❌ ${data.message}`, "#f44336");
                    }
                } else {
                    this._hfSetStatus("❌ Erro no login", "#f44336");
                }
            } catch (err) {
                this._hfSetStatus("❌ Erro de rede", "#f44336");
            }
        };

        // ==================== CARREGAR STATUS ====================
        nodeType.prototype._hfLoadStatus = async function () {
            try {
                const resp = await api.fetchApi("/hf_node/status");
                if (resp.ok) {
                    const data = await resp.json();
                    if (data.logged_in && data.username) {
                        const tokenWidget = this.widgets.find(w => w.name === "hf_token");
                        if (tokenWidget && !tokenWidget.value) {
                            tokenWidget.value = "(salvo — " + data.username + ")";
                        }
                        this.title = "🤗 HF 🟢";
                        this.color = "#154a15"; this.bgcolor = "#103a10";
                    }
                }
            } catch (e) { /* ignora */ }
        };

        // ==================== COLETAR MODELOS DO GRAFO ====================
        nodeType.prototype._collectModels = function () {
            const models = [];
            const graph = appInstance.graph;
            if (!graph || !graph._nodes) return models;

            for (const node of graph._nodes) {
                if (!node || !node.type || !node.widgets) continue;
                const nodeType = node.type;
                const knownWidgets = NODE_WIDGETS[nodeType];
                const category = NODE_CATEGORY[nodeType];

                if (category && knownWidgets) {
                    // Nó mapeado → widgets específicos (fonte confiável)
                    // Dedup por nome + categoria: permite clip_name1 e clip_name2 diferentes
                    for (const widget of node.widgets) {
                        const wName = (widget.name || "").toLowerCase();
                        if (!knownWidgets.some(w => w.toLowerCase() === wName)) continue;
                        let val = String(widget.value || "").trim();
                        if (!val || val === "undefined" || val === "none") continue;
                        // 🔥 FILTRO: ignora URLs (http/https) — não são nomes de modelo
                        if (/^https?:\/\//i.test(val)) continue;
                        // 🔥 Extrai só o nome do arquivo se for caminho/URL parcial
                        val = val.split('/').pop().split('\\').pop();
                        if (!val) continue;
                        if (!models.some(m => m.name === val && m.cat === category)) {
                            models.push({ name: val, cat: category });
                        }
                    }
                } else {
                    // Fallback: qualquer widget com extensão de modelo
                    // Dedup só por nome (categoria pode ser inferida diferente)
                    for (const widget of node.widgets) {
                        let val = String(widget.value || "").trim();
                        if (!val || val === "undefined" || val === "none") continue;
                        // 🔥 FILTRO: ignora URLs (http/https) — não são nomes de modelo
                        if (/^https?:\/\//i.test(val)) continue;
                        // 🔥 Extrai só o nome do arquivo se for caminho/URL parcial
                        val = val.split('/').pop().split('\\').pop();
                        if (!val) continue;
                        const hasExt = MODEL_FILE_EXTENSIONS.some(ext => val.toLowerCase().endsWith(ext));
                        if (hasExt && !models.some(m => m.name === val)) {
                            const cat = this._inferCategory(nodeType, widget.name, val);
                            models.push({ name: val, cat });
                        }
                    }
                }
            }
            return models;
        };

        // ==================== INFERIR CATEGORIA ====================
        nodeType.prototype._inferCategory = function (nodeType, widgetName, widgetValue) {
            const t = (nodeType || "").toLowerCase();
            const w = (widgetName || "").toLowerCase();
            const v = (widgetValue || "").toLowerCase();

            if (w.includes("lora") || v.includes("lora") || t.includes("lora")) return "lora";
            if (w.includes("vae") || v.includes("vae") || t.includes("vae")) return "vae";
            if (w.includes("control") || v.includes("control") || t.includes("control")) return "controlnet";
            if (w.includes("clip") || v.includes("clip") || t.includes("clip")) return "clip";
            if (w.includes("upscale") || v.includes("upscale") || t.includes("upscale")) return "upscaler";
            if (w.includes("embed") || v.includes("embed") || t.includes("embed") || t.includes("hypernetwork")) return "embeddings";
            if (w.includes("ip") && (v.includes("ip") || t.includes("ip"))) return "ipadapter";
            if (t.includes("vision")) return "clip_vision";
            if (t.includes("style")) return "style_models";
            if (t.includes("gligen")) return "gligen";
            if (t.includes("unet")) return "diffusion_models";
            if (t.includes("photomaker")) return "photomaker";
            if (t.includes("insight")) return "insightface";
            if (t.includes("animate")) return "animatediff_models";
            return "checkpoint";
        };

        // ==================== HELPERS ====================
        nodeType.prototype._hfSetStatus = function (text, color) {
            const statusWidget = this.widgets.find(w => w.name === "status");
            if (statusWidget) statusWidget.value = text;
            console.log(`[HF Node] ${text.replace(/\n/g, " | ")}`);
        };

        // ==================== ON EXECUTED ====================
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = origOnExecuted ? origOnExecuted.apply(this, arguments) : undefined;
            if (message && message.text) {
                const st = message.text[0] || "";
                if (st.includes("❌") || st.includes("🔴")) {
                    this.color = "#4a1515"; this.bgcolor = "#3a1010";
                } else if (st.includes("✅")) {
                    this.color = "#154a15"; this.bgcolor = "#103a10";
                } else if (st.includes("📥")) {
                    this.color = "#15354a"; this.bgcolor = "#10253a";
                }
            }
            return result;
        };
    },
};

appInstance.registerExtension(extension);
