"""Node principal: HF Model Downloader para ComfyUI.

Simplificado para usar folder_paths nativo do ComfyUI.
Scan é instantâneo — só verifica quais modelos da model_list não existem.
Download roda em thread separada para não travar o ComfyUI.
"""

import os
import json
import threading
from pathlib import Path
from typing import Tuple

from .hf_downloader import HFDownloader, MissingModel
from .hf_config import Config

# Cache (instanciado uma vez)
_downloader_cache = {}
_config_cache = None


def _get_config() -> Config:
    global _config_cache
    if _config_cache is None:
        _config_cache = Config()
    return _config_cache


def _get_downloader() -> HFDownloader:
    """Obtém ou cria o downloader, usando folder_paths do ComfyUI para achar models/."""
    global _downloader_cache
    try:
        import folder_paths
        # get_folder_paths retorna algo como "E:/ComfyUI/models/checkpoints"
        # dirname sobe 1 nível: "E:/ComfyUI/models" ← é exatamente o que queremos
        models_root = os.path.dirname(folder_paths.get_folder_paths("checkpoints")[0])
    except Exception:
        # Fallback: sobe do diretório do node
        current = Path(__file__).resolve().parent
        for _ in range(5):
            if (current / "models").exists():
                models_root = str(current / "models")
                break
            current = current.parent
        else:
            models_root = str(current / "models")

    if models_root not in _downloader_cache:
        token = _get_config().get_hf_token()
        _downloader_cache[models_root] = HFDownloader(models_root, token)
    return _downloader_cache[models_root]


class HFModelDownloaderNode:
    """
    Node que detecta modelos faltantes no workflow do ComfyUI e baixa do HuggingFace.
    
    Funcionamento:
    1. JS frontend escaneia app.graph._nodes e popula 'model_list' com categoria:nome
    2. Python verifica com folder_paths.get_full_path() se cada modelo existe
    3. Se action='download_all', baixa os faltantes do HuggingFace
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["none", "scan", "download_all"], {
                    "default": "none",
                }),
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "hf_xxxxxxxxx (deixe vazio se já logado)",
                }),
                "model_list": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Preenchido pelo frontend. Formato: categoria:nome",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "missing_models")
    OUTPUT_TOOLTIPS = (
        "Status da operação",
        "JSON com modelos faltantes (filename, category, folder, full_path)",
    )
    FUNCTION = "execute"
    CATEGORY = "model_management"
    DESCRIPTION = "Detecta e baixa modelos faltantes do HuggingFace automaticamente."

    def __init__(self):
        self._config = _get_config()

    def execute(
        self,
        action: str = "none",
        hf_token: str = "",
        model_list: str = "",
    ) -> Tuple[str, str]:
        dl = _get_downloader()

        # Login se token fornecido
        if hf_token and hf_token.strip():
            dl.set_token(hf_token.strip())
            username = dl.verify_token()
            if username:
                self._config.set_hf_token(hf_token.strip())
                self._config.set_hf_username(username)
            else:
                return ("❌ Token inválido. Verifique seu hf_token.", "")

        # Garante token do cache se já logado
        if not hf_token and self._config.is_logged_in():
            dl.set_token(self._config.get_hf_token())

        if model_list and model_list.strip():
            refs = self._parse_model_list(model_list)
            if not refs:
                return ("⚠️ Formato inválido no model_list.", "")

            missing = dl.find_missing_models(refs)

            if action == "download_all":
                return self._do_download(dl, missing)
            else:
                return self._do_scan(dl, refs, missing)
        else:
            login_status = "🟢" if self._config.is_logged_in() else "🔴"
            user = self._config.get_hf_username() or "Deslogado"
            return (f"Pronto. Login: {login_status} {user}", "")

    def _do_scan(self, dl, refs, missing):
        """Processa scan — retorna status e JSON de faltantes."""
        username = self._config.get_hf_username() or ""
        logged_in = self._config.is_logged_in()

        if not missing:
            login = f"🟢 {username}" if logged_in else "🔴 Deslogado"
            return (f"✅ Todos os {len(refs)} modelos encontrados! {login}", "")

        lines = [f"🔴 {len(missing)} faltante(s) de {len(refs)}:"]
        for m in missing:
            lines.append(f"  ❌ {m.comfy_folder}/{m.filename}")

        login = f"🟢 {username}" if logged_in else "🔴 Deslogado"
        lines.append(f"\n{login}")
        lines.append("\nMude action para 'download_all' para baixar.")

        missing_json = json.dumps([
            {"filename": m.filename, "category": m.category,
             "folder": m.comfy_folder, "full_path": m.full_path}
            for m in missing
        ])

        return ("\n".join(lines), missing_json)

    def _do_download(self, dl, missing):
        """Processa download — baixa todos e retorna status."""
        if not self._config.is_logged_in():
            return ("❌ Faça login primeiro! Configure o token HF.", "")

        if not missing:
            return ("✅ Nenhum modelo faltante. Tudo já baixado!", "")

        status_lines = [f"📥 Baixando {len(missing)} modelo(s)..."]
        results = []

        for m in missing:
            downloaded = dl.download_missing_model(m, search_first=True)
            if downloaded:
                status_lines.append(f"  ✅ {m.comfy_folder}/{m.filename}")
                results.append({"filename": m.filename, "folder": m.comfy_folder,
                                "status": "ok", "path": downloaded})
            else:
                status_lines.append(f"  ❌ {m.comfy_folder}/{m.filename}")
                results.append({"filename": m.filename, "folder": m.comfy_folder,
                                "status": "fail", "path": ""})

        ok = sum(1 for r in results if r["status"] == "ok")
        fail = len(results) - ok
        header = f"📥 Download: {ok} sucesso, {fail} falha(s)"
        return (header + "\n" + "\n".join(status_lines[1:]), json.dumps(results))

    def _parse_model_list(self, model_list: str) -> list:
        """Parse do model_list. Formato: categoria:nome (uma por linha).
        
        Sanitiza entradas: se o filename for URL, extrai só o nome do arquivo.
        """
        refs = []
        for line in model_list.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                cat, fname = line.split(":", 1)
                cat = cat.strip()
                fname = fname.strip()
            else:
                cat = "checkpoint"
                fname = line.strip()
            
            # 🔥 Sanitiza: se parece URL, extrai só o nome do arquivo
            if fname.startswith(("http://", "https://")):
                # Extrai o nome do arquivo da URL
                fname = os.path.basename(fname.split('?')[0])
                print(f"[HF Node] ⚠️ URL detectada em model_list, extraindo: {fname}")
            
            # 🔥 Remove caminhos (se veio como path completo)
            fname = fname.replace('\\', '/').rsplit('/', 1)[-1]
            
            if fname:
                refs.append({"category": cat, "filename": fname})
        return refs
