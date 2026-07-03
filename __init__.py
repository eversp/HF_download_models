"""HF Model Downloader — Custom Node para ComfyUI."""

import os
import uuid
import threading
import time
import asyncio
from pathlib import Path

# ═══════════════════════════════════════════════════════
# Importações com try/except para não quebrar o ComfyUI
# Se algo falhar, o node carrega parcialmente com aviso
# ═══════════════════════════════════════════════════════

_HF_NODE_LOAD_ERRORS = []

try:
    from .hf_node import HFModelDownloaderNode
except Exception as e:
    HFModelDownloaderNode = None
    _HF_NODE_LOAD_ERRORS.append(f"hf_node: {e}")
    print(f"[HF Node] ⚠️ Falha ao importar hf_node: {e}")

try:
    from .hf_config import Config
except Exception as e:
    Config = None
    _HF_NODE_LOAD_ERRORS.append(f"hf_config: {e}")
    print(f"[HF Node] ⚠️ Falha ao importar hf_config: {e}")

try:
    from .hf_downloader import HFDownloader
except Exception as e:
    HFDownloader = None
    _HF_NODE_LOAD_ERRORS.append(f"hf_downloader: {e}")
    print(f"[HF Node] ⚠️ Falha ao importar hf_downloader: {e}")

# Só registra o nó se tudo tiver carregado
if HFModelDownloaderNode is not None:
    NODE_CLASS_MAPPINGS = {
        "HFModelDownloader": HFModelDownloaderNode,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "HFModelDownloader": "🤗 HF Model Downloader",
    }
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    print(f"[HF Node] 🔴 Node NÃO registrado devido a erros de importação!")
    for err in _HF_NODE_LOAD_ERRORS:
        print(f"[HF Node]   • {err}")

WEB_DIRECTORY = "./js"


# ============================================================
# Rotas customizadas para comunicação JS ↔ Python via API
# ============================================================

_config = Config()
_downloader_cache = {}


def _find_models_path():
    """Encontra a raiz models/ usando folder_paths do ComfyUI (autoritativo)."""
    try:
        import folder_paths
        checkpoints = folder_paths.get_folder_paths("checkpoints")
        if checkpoints:
            return str(Path(checkpoints[0]).parent)
    except Exception:
        pass
    # Fallback
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "models").is_dir():
            return str(current / "models")
        current = current.parent
    return ""


def _get_downloader(models_path_override: str = ""):
    """Obtém downloader. Se models_path_override for fornecido, usa esse caminho."""
    models_path = models_path_override.strip() or _find_models_path()
    if not models_path:
        return None
    if models_path not in _downloader_cache:
        token = _config.get_hf_token()
        _downloader_cache[models_path] = HFDownloader(models_path, token)
    return _downloader_cache[models_path]


def _parse_model_list(model_list: str) -> list:
    """Parse do model_list textarea para lista de dicts."""
    refs = []
    for line in model_list.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            cat, fname = line.split(":", 1)
            refs.append({"category": cat.strip(), "filename": fname.strip()})
        else:
            refs.append({"category": "checkpoint", "filename": line.strip()})
    return refs


# ============================================================
# DownloadManager — gerencia downloads em background
# ============================================================

class DownloadManager:
    """Gerencia downloads de modelos em background, fora do event loop do aiohttp."""

    def __init__(self):
        self._jobs = {}
        self._cancel_events = {}  # job_id → threading.Event
        self._lock = threading.Lock()

    def _get_cancel_event(self, job_id: str) -> threading.Event:
        """Retorna o cancel_event para um job, criando se necessário."""
        with self._lock:
            if job_id not in self._cancel_events:
                self._cancel_events[job_id] = threading.Event()
            return self._cancel_events[job_id]

    def cancel_job(self, job_id: str) -> bool:
        """Cancela um job em andamento. Retorna True se cancelado."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                if job.get("status") in ("running", "starting"):
                    job["status"] = "cancelled"
                    job["current"] = "⛔ Cancelado pelo usuário"
                    # Seta o evento de cancelamento
                    if job_id in self._cancel_events:
                        self._cancel_events[job_id].set()
                        print(f"[HF Node] ⛔ Job {job_id} cancelado via API")
                    return True
        return False

    def start_job(self, model_refs: list, models_path: str) -> str:
        """Inicia um job de download em background. Retorna job_id."""
        job_id = str(uuid.uuid4())[:8]

        with self._lock:
            self._jobs[job_id] = {
                "status": "starting",
                "progress": "0/0",
                "current": "Iniciando...",
                "results": [],
                "error": None,
                "total": len(model_refs),
                "completed": 0,
            }
            self._cancel_events[job_id] = threading.Event()

        # Inicia thread em background
        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, model_refs, models_path),
            daemon=True,
        )
        thread.start()

        return job_id

    def _run_job(self, job_id: str, model_refs: list, models_path: str):
        """Executa os downloads em background."""
        try:
            dl = _get_downloader(models_path)
            if not dl:
                self._update_job(job_id, status="failed", error="Pasta models não encontrada!")
                return

            if not _config.is_logged_in():
                self._update_job(job_id, status="failed", error="Faça login primeiro!")
                return

            missing = dl.find_missing_models(model_refs)
            total = len(missing)

            self._update_job(job_id, status="running", progress=f"0/{total}", total=total)

            if total == 0:
                self._update_job(job_id, status="done", progress="0/0",
                                 current="Nenhum modelo faltante!")
                return

            results = []
            for idx, m in enumerate(missing):
                # Verifica cancelamento antes de cada modelo
                cancel_event = self._cancel_events.get(job_id)
                if cancel_event and cancel_event.is_set():
                    print(f"[HF Node] ⛔ Job {job_id} cancelado durante lote")
                    return

                current_msg = f"📥 Baixando {m.filename}..."
                self._update_job(job_id, current=current_msg,
                                 progress=f"{idx}/{total}")

                downloaded = dl.download_missing_model(m, search_first=True)
                if downloaded:
                    results.append({
                        "filename": m.filename,
                        "folder": m.comfy_folder,
                        "category": m.category,
                        "status": "ok",
                        "path": downloaded,
                    })
                else:
                    results.append({
                        "filename": m.filename,
                        "folder": m.comfy_folder,
                        "category": m.category,
                        "status": "fail",
                        "path": m.full_path,
                    })

                completed = idx + 1
                self._update_job(job_id, progress=f"{completed}/{total}",
                                 completed=completed, results=results)

            ok_count = sum(1 for r in results if r["status"] == "ok")
            fail_count = len(results) - ok_count
            print(f"[HF Node] 🏁 Job {job_id}: {ok_count} sucesso, {fail_count} falha(s)")

            self._update_job(job_id, status="done",
                             current=f"{ok_count} sucesso, {fail_count} falha(s)")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._update_job(job_id, status="failed", error=str(e))

    def _update_job(self, job_id: str, **kwargs):
        """Atualiza campos do job de forma thread-safe."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(kwargs)

    def get_job(self, job_id: str) -> dict:
        """Retorna estado atual do job (cópia)."""
        with self._lock:
            if job_id not in self._jobs:
                return {"status": "not_found", "error": "Job não encontrado"}
            return dict(self._jobs[job_id])

    def cleanup_old_jobs(self, max_age_seconds: int = 300):
        """Remove jobs concluídos após um tempo."""
        now = time.time()
        with self._lock:
            finished_statuses = {"done", "failed", "cancelled"}
            to_remove = [
                jid for jid, job in self._jobs.items()
                if job.get("status") in finished_statuses
            ]
            # Mantém os últimos N jobs concluídos
            keep = 10
            if len(to_remove) > keep:
                for jid in to_remove[:-keep]:
                    del self._jobs[jid]
                    self._cancel_events.pop(jid, None)


_download_manager = DownloadManager()


# ============================================================
# Registro de rotas
# ============================================================

try:
    from server import PromptServer
    from aiohttp import web

    routes = PromptServer.instance.routes

    @routes.post("/hf_node/login")
    async def hf_node_login(request):
        """Login no HuggingFace."""
        try:
            data = await request.json()
            token = data.get("token", "").strip()
            if not token:
                return web.json_response({"success": False, "message": "Token vazio"})

            dl = _get_downloader()
            dl.set_token(token)
            username = dl.verify_token()
            if username:
                _config.set_hf_token(token)
                _config.set_hf_username(username)
                print(f"[HF Node] ✅ Login OK: {username}")
                return web.json_response({"success": True, "username": username})
            else:
                return web.json_response({"success": False, "message": "Token inválido ou expirado"})
        except Exception as e:
            print(f"[HF Node] Erro login: {e}")
            return web.json_response({"success": False, "message": str(e)})

    @routes.post("/hf_node/scan")
    async def hf_node_scan(request):
        """Escaneia modelos — retorna lista de faltantes com caminhos completos."""
        try:
            data = await request.json()
            model_list = data.get("model_list", "")
            models_path = data.get("models_path", "")
            if not model_list:
                return web.json_response({"success": False, "message": "Lista vazia", "missing": []})

            refs = _parse_model_list(model_list)
            dl = _get_downloader(models_path)
            if not dl:
                return web.json_response({"success": False, "message": "Pasta models não encontrada!", "missing": []})
            missing = dl.find_missing_models(refs)

            missing_list = [
                {
                    "filename": m.filename,
                    "category": m.category,
                    "folder": m.comfy_folder,
                    "full_path": m.full_path,
                }
                for m in missing
            ]
            print(f"[HF Node] Scan: {len(refs)} total, {len(missing)} missing")
            return web.json_response({
                "success": True,
                "total": len(refs),
                "missing_count": len(missing),
                "missing": missing_list,
            })
        except Exception as e:
            print(f"[HF Node] Erro scan: {e}")
            return web.json_response({"success": False, "message": str(e), "missing": []})

    @routes.post("/hf_node/download_start")
    async def hf_node_download_start(request):
        """
        Inicia download em background (assíncrono, não bloqueante).
        Retorna job_id para consultar progresso via GET /hf_node/download_status/{job_id}.
        """
        try:
            data = await request.json()
            model_list = data.get("model_list", "")
            models_path = data.get("models_path", "")

            if not _config.is_logged_in():
                return web.json_response({"success": False, "message": "Faça login primeiro!"})
            if not model_list:
                return web.json_response({"success": False, "message": "Lista vazia"})

            refs = _parse_model_list(model_list)
            job_id = _download_manager.start_job(refs, models_path)

            print(f"[HF Node] 🚀 Job {job_id} iniciado com {len(refs)} modelo(s)")

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "total": len(refs),
            })

        except Exception as e:
            print(f"[HF Node] Erro download_start: {e}")
            import traceback
            traceback.print_exc()
            return web.json_response({"success": False, "message": str(e)})

    @routes.get("/hf_node/download_status/{job_id}")
    async def hf_node_download_status(request):
        """Retorna o status atual de um job de download."""
        try:
            job_id = request.match_info.get("job_id", "")
            job = _download_manager.get_job(job_id)

            if job.get("status") == "not_found":
                return web.json_response({"success": False, "message": "Job não encontrado"})

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "status": job.get("status", "unknown"),
                "progress": job.get("progress", "0/0"),
                "current": job.get("current", ""),
                "total": job.get("total", 0),
                "completed": job.get("completed", 0),
                "results": job.get("results", []),
                "error": job.get("error"),
                "bytes_downloaded": job.get("bytes_downloaded", 0),
                "bytes_total": job.get("bytes_total", 0),
                "existed": job.get("existed", False),
            })

        except Exception as e:
            print(f"[HF Node] Erro download_status: {e}")
            return web.json_response({"success": False, "message": str(e)})

    # ═══════════════════════════════════════════════════════
    # Rota: Cancelar download
    # ═══════════════════════════════════════════════════════

    @routes.post("/hf_node/cancel/{job_id}")
    async def hf_node_cancel(request):
        """Cancela um download em andamento."""
        try:
            job_id = request.match_info.get("job_id", "")
            cancelled = _download_manager.cancel_job(job_id)
            if cancelled:
                return web.json_response({"success": True, "message": "Download cancelado"})
            else:
                # Pode ser que já terminou ou não existe
                job = _download_manager.get_job(job_id)
                if job.get("status") == "not_found":
                    return web.json_response({"success": False, "message": "Job não encontrado"})
                return web.json_response({"success": True, "message": f"Job já está em estado: {job.get('status')}"})
        except Exception as e:
            print(f"[HF Node] Erro cancel: {e}")
            return web.json_response({"success": False, "message": str(e)})

    # ═══════════════════════════════════════════════════════
    # Rota: Download por URL manual do HuggingFace
    # ═══════════════════════════════════════════════════════

    @routes.post("/hf_node/download_url")
    async def hf_node_download_url(request):
        """
        Baixa um modelo diretamente de uma URL do HuggingFace.
        O usuário cola a URL e escolhe a pasta de destino.
        """
        try:
            data = await request.json()
            url = data.get("url", "").strip()
            category = data.get("category", "checkpoint").strip()
            models_path = data.get("models_path", "")

            if not _config.is_logged_in():
                return web.json_response({"success": False, "message": "Faça login primeiro!"})

            if not url:
                return web.json_response({"success": False, "message": "URL vazia"})

            dl = _get_downloader(models_path)
            if not dl:
                return web.json_response({"success": False, "message": "Pasta models não encontrada!"})

            # Resolve a pasta de destino
            comfy_folder = HFDownloader._resolve_comfy_folder(category)

            destination_folder = str(Path(_find_models_path() if not models_path else models_path) / comfy_folder)

            print(f"[HF Node] 📥 URL download: {url} → {comfy_folder}")

            # Baixa em background (thread)
            job_id = str(uuid.uuid4())[:8]
            cancel_event = _download_manager._get_cancel_event(job_id)

            thread = threading.Thread(
                target=_run_url_download_job,
                args=(job_id, url, destination_folder, dl),
                daemon=True,
            )
            thread.start()

            # Registra o job
            with _download_manager._lock:
                _download_manager._jobs[job_id] = {
                    "status": "starting",
                    "progress": "0/1",
                    "current": "Iniciando download da URL...",
                    "results": [],
                    "error": None,
                    "total": 1,
                    "completed": 0,
                }

            return web.json_response({
                "success": True,
                "job_id": job_id,
                "total": 1,
                "message": f"Baixando para models/{comfy_folder}/",
            })

        except Exception as e:
            print(f"[HF Node] Erro download_url: {e}")
            import traceback
            traceback.print_exc()
            return web.json_response({"success": False, "message": str(e)})


    def _run_url_download_job(job_id: str, url: str, destination_folder: str, dl):
        """Executa download de URL em background com suporte a cancelamento."""
        try:
            parsed = HFDownloader.parse_hf_url(url)

            if not parsed:
                with _download_manager._lock:
                    if job_id in _download_manager._jobs:
                        _download_manager._jobs[job_id]["status"] = "failed"
                        _download_manager._jobs[job_id]["error"] = "URL inválida"
                return

            filename = os.path.basename(parsed.get("filepath") or parsed["repo_id"].split("/")[-1])

            # Verifica cancelamento antes de começar
            cancel_event = _download_manager._cancel_events.get(job_id)
            if cancel_event and cancel_event.is_set():
                return

            # Atualiza progresso
            with _download_manager._lock:
                if job_id in _download_manager._jobs:
                    _download_manager._jobs[job_id]["status"] = "running"
                    _download_manager._jobs[job_id]["current"] = f"📥 Baixando {filename}..."
                    _download_manager._jobs[job_id]["progress"] = "0/1"
                    _download_manager._jobs[job_id]["bytes_downloaded"] = 0
                    _download_manager._jobs[job_id]["bytes_total"] = 0

            # Progress callback: atualiza bytes em tempo real no job
            def _progress_callback(downloaded_bytes, total_bytes):
                # Se foi cancelado, não atualiza mais
                if cancel_event and cancel_event.is_set():
                    return
                with _download_manager._lock:
                    if job_id in _download_manager._jobs:
                        _download_manager._jobs[job_id]["bytes_downloaded"] = downloaded_bytes
                        _download_manager._jobs[job_id]["bytes_total"] = total_bytes

            # Passa o cancel_event para o download
            result = dl.download_from_url(
                url, destination_folder, filename,
                progress_callback=_progress_callback,
                cancel_event=cancel_event,
            )

            # Se foi cancelado, não atualiza resultado (já foi marcado como cancelled)
            if cancel_event and cancel_event.is_set():
                return

            if result:
                existed = result.get("existed", False)
                if existed:
                    status_current = f"⚠️ {filename} já existe em disco (nada baixado)"
                    status_name = "exists"
                else:
                    status_current = f"✅ {filename} baixado!"
                    status_name = "done"

                with _download_manager._lock:
                    if job_id in _download_manager._jobs:
                        _download_manager._jobs[job_id]["status"] = status_name
                        _download_manager._jobs[job_id]["progress"] = "1/1"
                        _download_manager._jobs[job_id]["completed"] = 1
                        _download_manager._jobs[job_id]["current"] = status_current
                        _download_manager._jobs[job_id]["existed"] = existed
                        _download_manager._jobs[job_id]["results"] = [{
                            "filename": filename,
                            "folder": os.path.basename(destination_folder),
                            "status": "exists" if existed else "ok",
                            "path": result.get("path", ""),
                        }]
            else:
                # Se foi cancelado, não marca como failed
                if cancel_event and cancel_event.is_set():
                    return
                with _download_manager._lock:
                    if job_id in _download_manager._jobs:
                        _download_manager._jobs[job_id]["status"] = "failed"
                        _download_manager._jobs[job_id]["error"] = "Falha ao baixar"
                        _download_manager._jobs[job_id]["current"] = f"❌ Falha ao baixar {filename}"
                        _download_manager._jobs[job_id]["results"] = [{
                            "filename": filename,
                            "folder": os.path.basename(destination_folder),
                            "status": "fail",
                            "path": "",
                        }]

        except Exception as e:
            import traceback
            traceback.print_exc()
            # Só marca como erro se não foi cancelado
            cancel_event = _download_manager._cancel_events.get(job_id)
            if cancel_event and cancel_event.is_set():
                return
            with _download_manager._lock:
                if job_id in _download_manager._jobs:
                    _download_manager._jobs[job_id]["status"] = "failed"
                    _download_manager._jobs[job_id]["error"] = str(e)

    # Mantém a rota antiga como fallback para compatibilidade
    @routes.post("/hf_node/download")
    async def hf_node_download_legacy(request):
        """Rota legada — redireciona para download_start (mantida para compatibilidade)."""
        try:
            data = await request.json()
            model_list = data.get("model_list", "")
            models_path = data.get("models_path", "")

            if not _config.is_logged_in():
                return web.json_response({"success": False, "message": "Faça login primeiro!"})
            if not model_list:
                return web.json_response({"success": False, "message": "Lista vazia"})

            refs = _parse_model_list(model_list)
            job_id = _download_manager.start_job(refs, models_path)

            # Aguarda o job terminar (timeout máximo)
            timeout = 600  # 10 minutos
            interval = 1
            waited = 0
            while waited < timeout:
                job = _download_manager.get_job(job_id)
                if job.get("status") in ("done", "failed"):
                    return web.json_response({
                        "success": job.get("status") == "done",
                        "message": job.get("current", ""),
                        "results": job.get("results", []),
                        "job_id": job_id,
                        "background": True,
                    })
                await asyncio.sleep(interval)
                waited += interval

            return web.json_response({
                "success": True,
                "message": "Download ainda em andamento (consulte status)",
                "job_id": job_id,
                "background": True,
            })

        except Exception as e:
            print(f"[HF Node] Erro download (legado): {e}")
            import traceback
            traceback.print_exc()
            return web.json_response({"success": False, "message": str(e), "results": []})

    @routes.get("/hf_node/status")
    async def hf_node_status(request):
        """Status de login."""
        return web.json_response({
            "logged_in": _config.is_logged_in(),
            "username": _config.get_hf_username() or "",
        })

    print("[HF Node] ✅ Rotas: /hf_node/{login,scan,download_start,download_status,cancel,download_url,download,status}")

except Exception as e:
    print(f"[HF Node] ⚠️ Erro ao registrar rotas: {e}")
