<div align="center">

# 🤗 HF Model Downloader

**Custom Node para ComfyUI** — Detecta e baixa automaticamente modelos faltantes do HuggingFace.

![Python](https://img.shields.io/badge/python-3.10+-blue?style=flat-square&logo=python&logoColor=white)
![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-brightgreen?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-API-yellow?style=flat-square&logo=huggingface&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

[🇧🇷 Português](#-português) · [🇺🇸 English](#-english)

---

</div>

## 🇧🇷 Português

### O que é?

Nunca mais perca tempo caçando modelos manualmente. O **HF Model Downloader** varre seu workflow do ComfyUI, identifica quais modelos estão faltando e baixa **automaticamente** do HuggingFace na pasta correta — seja checkpoint, LoRA, VAE, ControlNet, embedding, etc.

### ✨ Funcionalidades

| Recurso | Descrição |
|---------|-----------|
| 🔍 **Scan Inteligente** | Varre o grafo do workflow e detecta modelos referenciados mas não encontrados em disco |
| 📥 **Download Automático** | Baixa do HuggingFace direto na pasta correta (checkpoints/, loras/, vae/, etc.) |
| 🧠 **Categorização Automática** | Detecta se é checkpoint, LoRA, VAE, ControlNet, UNet, etc. |
| 🔑 **Login via Token HF** | Autentica com seu token (salvo com segurança localmente) |
| 🟢 **Status Visual** | Título do node muda de cor (verde = logado, vermelho = modelos faltando) |
| ⏱️ **Progresso em Tempo Real** | Barra de progresso com bytes baixados no frontend |
| ⛔ **Cancelamento** | Botão para cancelar downloads em andamento |
| 🌐 **Download por URL** | Cole uma URL do HuggingFace e baixe direto |

### 🚀 Instalação

```bash
# 1. Clone na pasta custom_nodes do seu ComfyUI
cd SEU_COMFYUI/custom_nodes
git clone https://github.com/Ever-brsp/HF_downlosad_models.git

# 2. Instale as dependências (com o Python do ComfyUI)
./python_embeded/python.exe -m pip install -r requirements.txt

# 3. Reinicie o ComfyUI
```

### 📖 Como Usar

1. **Reinicie o ComfyUI** após instalar
2. Clique com botão direito no canvas → **Add Node** → `model_management` → `🤗 HF Model Downloader`
3. Digite seu **token do HuggingFace** ([criar aqui](https://huggingface.co/settings/tokens))
4. Clique em **🔑 Login HF** (título fica verde quando logado)
5. Clique em **🔍 Scan Workflow** para ver o que está faltando
6. Clique em **📥 Download All Missing** — ele baixa tudo nas pastas certas!

> 💡 O token fica salvo **localmente** no `hf_node_config.json`. Use o botão **Logout** para removê-lo.

### 📁 Estrutura

```
HF_downlosad_models/
├── __init__.py          ← Registro do node + rotas HTTP
├── hf_node.py           ← Node principal do ComfyUI
├── hf_downloader.py     ← API de busca e download do HuggingFace
├── hf_config.py         ← Persistência do token
├── hf_constants.py      ← Constantes e mapeamentos
├── requirements.txt     ← Dependências Python
├── js/
│   └── hf_downloader.js ← Extensão frontend (widgets, scan, progresso)
└── utils/               ← Utilitários (em expansão)
```

### 🛠️ Requisitos

- ComfyUI (qualquer versão recente)
- Python 3.10+
- Token do HuggingFace ([gratuito](https://huggingface.co/settings/tokens))

---

## 🇺🇸 English

### What is it?

Never manually hunt for missing models again. **HF Model Downloader** scans your ComfyUI workflow, detects which models are missing, and **automatically downloads** them from HuggingFace into the correct folder — whether it's a checkpoint, LoRA, VAE, ControlNet, embedding, or anything else.

### ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Smart Scan** | Scans the workflow graph and detects referenced models not found on disk |
| 📥 **Auto Download** | Downloads from HuggingFace directly into the correct folder |
| 🧠 **Auto Category** | Detects if it's a checkpoint, LoRA, VAE, ControlNet, UNet, etc. |
| 🔑 **Login via HF Token** | Authenticates with your token (saved locally & securely) |
| 🟢 **Visual Status** | Node title changes color (green = logged in, red = missing models) |
| ⏱️ **Real-time Progress** | Progress bar with downloaded bytes on the frontend |
| ⛔ **Cancel** | Cancel ongoing downloads with one click |
| 🌐 **URL Download** | Paste any HuggingFace URL and download directly |

### 🚀 Installation

```bash
# 1. Clone into your ComfyUI custom_nodes folder
cd YOUR_COMFYUI/custom_nodes
git clone https://github.com/Ever-brsp/HF_downlosad_models.git

# 2. Install dependencies (using ComfyUI's Python)
./python_embeded/python.exe -m pip install -r requirements.txt

# 3. Restart ComfyUI
```

### 📖 How to Use

1. **Restart ComfyUI** after installation
2. Right-click canvas → **Add Node** → `model_management` → `🤗 HF Model Downloader`
3. Enter your **HuggingFace token** ([create one](https://huggingface.co/settings/tokens))
4. Click **🔑 Login HF** (title turns green when logged in)
5. Click **🔍 Scan Workflow** to see what's missing
6. Click **📥 Download All Missing** — it downloads everything to the right folders!

> 💡 The token is saved **locally** in `hf_node_config.json`. Use the **Logout** button to remove it.

---

<div align="center">

**Made with ❤️ for the ComfyUI community**

</div>
