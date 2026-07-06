"""Downloader e buscador de modelos do HuggingFace para o node ComfyUI."""

import os
import threading
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass, field

import requests as http_requests

from huggingface_hub import (
    HfApi,
    hf_hub_download,
    hf_hub_url,
    list_repo_files,
    login,
    whoami,
)

from .hf_constants import MODEL_EXTENSIONS


@dataclass
class ModelInfo:
    model_id: str
    author: str = ""
    model_name: str = ""
    tags: List[str] = field(default_factory=list)
    downloads: int = 0
    likes: int = 0
    is_gated: bool = False

@dataclass
class MissingModel:
    filename: str
    category: str
    comfy_folder: str
    full_path: str


class HFDownloader:
    """Gerencia autenticação, busca e downloads do HuggingFace."""

    def __init__(self, comfyui_models_path: str, token: Optional[str] = None):
        self._api = HfApi()
        self._token: Optional[str] = token
        self._comfyui_models_path = Path(comfyui_models_path)
        self._lock = threading.Lock()

        if token:
            self.set_token(token)

    def set_token(self, token: Optional[str]):
        self._token = token
        if token:
            try:
                login(token=token)
            except Exception:
                pass

    def verify_token(self) -> Optional[str]:
        if not self._token:
            return None
        try:
            info = whoami(token=self._token)
            return info.get("name", "")
        except Exception:
            return None

    # --- Scan de modelos (RESPEITA SYMLINKS e extra_model_paths.yaml!) ---
    def _file_exists_comfy(self, folder_name: str, filename: str) -> bool:
        """
        Verifica existência usando a API oficial do ComfyUI.
        Respeita symlinks, extra_model_paths.yaml, múltiplas pastas.
        Design original — 4 passos diretos, sem helpers desnecessários.
        """
        # 1. folder_paths.get_full_path() — API oficial do ComfyUI (resolve tudo)
        try:
            import folder_paths
            resolved = folder_paths.get_full_path(folder_name, filename)
            if resolved and os.path.exists(resolved):
                return True
        except Exception:
            pass

        # 2. os.path.realpath() — resolve symlinks manualmente
        try:
            candidate = os.path.join(str(self._comfyui_models_path), folder_name, filename)
            real = os.path.realpath(candidate)
            if real != candidate and os.path.exists(real):
                return True
        except Exception:
            pass

        # 3. Verificação direta (fallback)
        direct = self._comfyui_models_path / folder_name / filename
        if direct.exists():
            return True

        # 4. Tenta sem extensão (modelos sem .safetensors/.ckpt no nome)
        if not any(filename.lower().endswith(ext) for ext in MODEL_EXTENSIONS):
            for ext in MODEL_EXTENSIONS:
                try:
                    import folder_paths
                    resolved = folder_paths.get_full_path(folder_name, filename + ext)
                    if resolved and os.path.exists(resolved):
                        return True
                except Exception:
                    pass
                alt = self._comfyui_models_path / folder_name / (filename + ext)
                if alt.exists():
                    return True
                try:
                    real = os.path.realpath(str(alt))
                    if os.path.exists(real):
                        return True
                except Exception:
                    pass

        return False

    def find_missing_models(self, model_refs: List[Dict]) -> List[MissingModel]:
        """Verifica modelos faltantes usando folder_paths do ComfyUI (symlinks OK)."""
        missing = []
        for ref in model_refs:
            filename = ref.get("filename", "").strip()
            category = ref.get("category", "checkpoint").strip()
            if not filename:
                continue

            # 🔥 SEGURANÇA: ignora se filename parece URL (nunca deve chegar aqui, mas...)
            if filename.startswith(('http://', 'https://', 'huggingface.co')):
                print(f"[HF Node] ⚠️ Ignorando URL-like filename: {filename}")
                continue

            # Resolve o nome da pasta usando folder_paths nativo do ComfyUI
            comfy_folder = self._resolve_comfy_folder(category)

            if not self._file_exists_comfy(comfy_folder, filename):
                missing.append(MissingModel(
                    filename=filename, category=category,
                    comfy_folder=comfy_folder,
                    full_path=str(self._comfyui_models_path / comfy_folder / filename),
                ))
        return missing

    @staticmethod
    def _resolve_comfy_folder(category: str) -> str:
        """Resolve o nome da pasta ComfyUI a partir de uma categoria.
        
        Usa folder_paths.folder_names_and_paths como fonte autoritativa.
        Aplica map_legacy() para compatibilidade com nomes antigos (ex: 'unet' → 'diffusion_models').
        """
        try:
            import folder_paths
            known = folder_paths.folder_names_and_paths
            # Aplica map_legacy primeiro (ex: 'unet' → 'diffusion_models', 'clip' → 'text_encoders')
            mapped = folder_paths.map_legacy(category)
            if mapped in known:
                return mapped
            if category in known:
                return category
            # Tenta com 's' (ex: lora → loras)
            if category + "s" in known:
                return category + "s"
        except Exception:
            pass
        # Fallback completo — cobre TODAS as categorias que o frontend pode enviar
        FALLBACK = {
            "checkpoint": "checkpoints",
            "checkpoints": "checkpoints",
            "lora": "loras",
            "loras": "loras",
            "vae": "vae",
            "clip": "clip",
            "text_encoders": "clip",
            "controlnet": "controlnet",
            "embeddings": "embeddings",
            "unet": "diffusion_models",
            "diffusion_models": "diffusion_models",
            "upscaler": "upscale_models",
            "upscale_models": "upscale_models",
            "ipadapter": "ipadapter",
            "style_models": "style_models",
            "clip_vision": "clip_vision",
            "gligen": "gligen",
            "photomaker": "photomaker",
            "insightface": "insightface",
            "animatediff_models": "animatediff_models",
            "animatediff_motion_lora": "animatediff_motion_lora",
            "hypernetwork": "embeddings",
        }
        return FALLBACK.get(category, FALLBACK.get(category + "s", "checkpoints"))

    # --- Busca ---
    def search_models(self, query: str, limit: int = 20) -> List[ModelInfo]:
        import requests
        results = []
        clean_query = query.strip()
        for ext in MODEL_EXTENSIONS:
            clean_query = clean_query.replace(ext, '')
        if '/' in clean_query or '\\' in clean_query:
            clean_query = clean_query.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
        clean_query = clean_query.strip().replace('_', ' ').replace('-', ' ')
        try:
            url = "https://huggingface.co/api/models"
            params = {"search": clean_query, "limit": limit, "sort": "downloads", "direction": "-1", "full": "true"}
            headers = {}
            if self._token:
                headers["Authorization"] = f"Bearer {self._token}"
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code == 200:
                for m in resp.json():
                    model_id = m.get("modelId") or m.get("id", "")
                    if not model_id or "/" not in model_id:
                        continue
                    results.append(ModelInfo(
                        model_id=model_id,
                        author=m.get("author") or model_id.split("/")[0],
                        model_name=model_id.split("/")[-1] if "/" in model_id else model_id,
                        tags=m.get("tags", []),
                        downloads=m.get("downloads") or 0,
                        likes=m.get("likes") or 0,
                        is_gated=m.get("gated", False),
                    ))
        except Exception:
            pass
        return results

    def _score_candidate(self, repo_id: str, expected_filename: str) -> tuple:
        """
        Avalia um repositório candidato.
        Retorna (score, filepath, file_size) onde:
          score > 0 = viável, maior = melhor
          score = 0 = falhou
        """
        try:
            files = self._list_model_files(repo_id)
            best_size = 0
            best_file = None
            for f in files:
                for ext in ['.safetensors', '.ckpt', '.pt', '.pth', '.bin']:
                    if f.lower().endswith(ext):
                        size, _ = self._get_file_metadata(repo_id, f)
                        if not self._is_pruned_path(f):
                            if size > best_size:
                                best_size = size
                                best_file = f
                        break

            if best_file and best_size > 0:
                # Score: tamanho em GB + bônus se nome do arquivo casa exatamente
                score = best_size
                if expected_filename and os.path.basename(best_file) == expected_filename:
                    score += 1_000_000_000  # bônus enorme para match exato de filename
                return (score, best_file, best_size)
        except Exception:
            pass
        return (0, None, 0)

    def search_best_match(self, filename: str) -> Optional[str]:
        """
        Busca o melhor match para um filename no HuggingFace.
        Agora com verificação de tamanho real do arquivo e preferência por modelos completos.
        
        Retorna o repo_id do melhor candidato, ou None.
        """
        # Extrai nome base sem extensão
        base = filename
        for ext in ['.safetensors', '.ckpt', '.pt', '.pth', '.bin']:
            if base.lower().endswith(ext):
                base = base[:-len(ext)]
                break

        # Coleciona todos os candidatos com suas pontuações
        candidates = []  # (model_id, score, filepath, file_size)

        def _add_candidates(results_list):
            """Adiciona candidatos da lista de resultados, avaliando cada um."""
            for r in results_list:
                score, fp, sz = self._score_candidate(r.model_id, filename)
                if score > 0:
                    candidates.append((r.model_id, score, fp, sz))
                    size_gb = sz / 1024**3
                    print(f"[HF Node]    Candidato: {r.model_id} | {os.path.basename(fp) if fp else '?'} | {size_gb:.1f}GB")

        # ── Estratégia 1: busca exata com o nome completo (top 10) ──
        results = self.search_models(filename, limit=10)
        base_lower = base.lower().strip().replace('-', ' ').replace('_', ' ')
        filtered = [r for r in results if
                     base_lower in r.model_name.lower().replace('-', ' ').replace('_', ' ') or
                     r.model_name.lower().replace('-', ' ').replace('_', ' ') in base_lower]
        if filtered:
            _add_candidates(filtered)

        # ── Estratégia 2: tenta sem sufixo de versão ──
        if not candidates:
            parts = base.rsplit('-', 1)
            if len(parts) > 1 and any(c.isdigit() for c in parts[-1]):
                results = self.search_models(parts[0].strip(), limit=10)
                if results:
                    _add_candidates(results)

        # ── Estratégia 3: busca mais ampla (top 30) ──
        if not candidates:
            clean = base.replace('-', ' ').replace('_', ' ').strip()
            results = self.search_models(clean, limit=30)
            if results:
                _add_candidates(results)

        # ── Estratégia 4: tenta forjar repo_id com autores conhecidos ──
        KNOWN_AUTHORS = {
            'ltx': 'Lightricks', 'sdxl': 'stabilityai', 'sd': 'stabilityai',
            'flux': 'black-forest-labs', 'wuerstchen': 'wurstmeister',
            'deepfloyd': 'DeepFloyd', 'playground': 'playgroundai',
            'pixart': 'PixArt-alpha', 'cogvideo': 'THUDM', 'animatediff': 'guoyww',
        }
        if not candidates:
            first_word = base.split('-')[0].split('_')[0].lower()
            if first_word in KNOWN_AUTHORS:
                for candidate in (f"{KNOWN_AUTHORS[first_word]}/{base}",
                                  f"{KNOWN_AUTHORS[first_word]}/{filename}"):
                    try:
                        from huggingface_hub import repo_exists
                        if repo_exists(candidate, token=self._token):
                            score, fp, sz = self._score_candidate(candidate, filename)
                            if score > 0:
                                candidates.append((candidate, score, fp, sz))
                    except Exception:
                        pass

        # ── Escolhe o melhor candidato: maior score = maior arquivo + bônus match exato ──
        if candidates:
            candidates.sort(key=lambda x: -x[1])  # maior score primeiro
            best = candidates[0]
            size_gb = best[3] / 1024**3
            print(f"[HF Node] ✅ Melhor match: {best[0]} ({size_gb:.1f}GB, arquivo: {os.path.basename(best[2])})")
            if size_gb < 0.5:
                print(f"[HF Node] ⚠️  Modelo muito pequeno ({size_gb:.1f}GB) — pode estar incompleto!")
            return best[0]

        print(f"[HF Node] ❌ Nenhum match encontrado para '{filename}'")
        return None

    # --- Download arquivo ÚNICO + renomeio ---
    def _get_file_metadata(self, repo_id: str, filepath: str) -> tuple[int, str]:
        """Obtém tamanho (bytes) e url de download de um arquivo no HF."""
        try:
            from huggingface_hub import get_hf_file_metadata, hf_hub_url
            url = hf_hub_url(repo_id=repo_id, filename=filepath)
            meta = get_hf_file_metadata(url, token=self._token)
            return (meta.size or 0, url)
        except Exception:
            return (0, "")

    @staticmethod
    def _is_pruned_path(filepath: str) -> bool:
        f_lower = filepath.lower()
        pruned_kw = ['pruned', 'quantized', 'gguf', 'onnx', 'int8', 'int4', 'fp8']
        return any(kw in f_lower for kw in pruned_kw)

    def _pick_model_file(self, files: List[str]) -> Optional[str]:
        preferred = []
        fallback = []
        for f in files:
            for ext in ['.safetensors', '.ckpt', '.pt', '.pth', '.bin']:
                if f.lower().endswith(ext):
                    if self._is_pruned_path(f):
                        fallback.append(f)
                    else:
                        preferred.append(f)
                    break
        return preferred[0] if preferred else (fallback[0] if fallback else None)

    def _list_model_files(self, repo_id: str) -> List[str]:
        """Lista arquivos de um repositório HF."""
        try:
            return list_repo_files(repo_id, token=self._token)
        except Exception:
            return []

    def download_single_file(self, model_id: str, expected_filename: str, destination_folder: str, 
                             repo_filepath: str = None, progress_callback=None,
                             cancel_event: threading.Event = None) -> str | None:
        """
        Baixa só o .safetensors/.ckpt e renomeia para expected_filename.
        
        ═══════════════════════════════════════════════════════════════
        CORREÇÃO #1: SEM tempfile no C:! Baixa direto no mesmo drive
        CORREÇÃO #2: SEM shutil.move() cross-drive! Renomeio local.
        CORREÇÃO #3: Monitora o CAMINHO EXATO do arquivo, não os.walk()
        CORREÇÃO #4: cancel_event para cancelamento via botão.
        ═══════════════════════════════════════════════════════════════
        
        Args:
            model_id: ID do repositório (ex: "author/model")
            expected_filename: Nome final do arquivo (ex: "model.safetensors")
            destination_folder: Pasta de destino
            repo_filepath: Caminho EXATO no repositório (ex: "v18/model.safetensors").
                          Se fornecido, pula a listagem e usa este arquivo direto.
                          Se None, lista e escolhe automaticamente.
            progress_callback: Função opcional chamada com (downloaded_bytes, total_bytes)
            cancel_event: threading.Event — se setado, o download é abortado.
        
        Retorna o caminho absoluto do arquivo baixado, ou None se falhar.
        """
        import shutil
        try:
            os.makedirs(destination_folder, exist_ok=True)
            dest_path = os.path.join(destination_folder, expected_filename)

            if os.path.exists(dest_path):
                print(f"[HF Node] ✅ Já existe em disco: {expected_filename}")
                self._invalidate_comfy_cache(destination_folder)
                return dest_path

            # Se veio caminho exato da URL, usa ele direto (NÃO lista repositório)
            if repo_filepath:
                model_file = repo_filepath
                print(f"[HF Node]    Usando caminho exato da URL: {model_file}")
            else:
                # Fallback: lista arquivos e escolhe automaticamente
                files = self._list_model_files(model_id)
                if not files:
                    print(f"[HF Node]    Repositório vazio ou inacessível")
                    return None

                model_file = self._pick_model_file(files)
                if not model_file:
                    print(f"[HF Node]    Nenhum .safetensors/.ckpt em {model_id}")
                    print(f"[HF Node]    Arquivos: {files[:5]}...")
                    return None

            print(f"[HF Node]    Arquivo remoto: {model_file}")

            # ──────────────────────────────────────────────────────────────
            # CORREÇÃO #1 e #2: Criar temp NO MESMO DRIVE do destino
            # ──────────────────────────────────────────────────────────────
            # Em vez de tempfile (que usa TEMP do sistema = C:), criamos
            # uma pasta temporária ao lado do destino. shutil.move() entre
            # pastas no mesmo drive é um rename — instantâneo, sem cópia!
            # ──────────────────────────────────────────────────────────────
            tmp_dir = os.path.join(destination_folder, f".hf_download_tmp_{expected_filename}")
            os.makedirs(tmp_dir, exist_ok=True)

            # Se algo der errado, garantimos limpeza
            def _cleanup_tmp():
                if os.path.exists(tmp_dir):
                    try:
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                    except Exception:
                        pass

            try:
                # Descobre tamanho total (para progresso)
                total_bytes = 0
                if progress_callback is not None:
                    try:
                        from huggingface_hub import get_hf_file_metadata, hf_hub_url
                        metadata_url = hf_hub_url(repo_id=model_id, filename=model_file)
                        meta = get_hf_file_metadata(metadata_url, token=self._token)
                        total_bytes = meta.size or 0
                    except Exception:
                        total_bytes = 0
                    print(f"[HF Node]    Tamanho total: {total_bytes} bytes")

                # Inicia o download numa thread (hf_hub_download é síncrono)
                download_result = [None]
                download_error = [None]

                def _do_download():
                    try:
                        path = hf_hub_download(
                            repo_id=model_id,
                            filename=model_file,
                            local_dir=tmp_dir,
                            token=self._token,
                        )
                        download_result[0] = path
                    except Exception as e:
                        download_error[0] = e

                dl_thread = threading.Thread(target=_do_download, daemon=True)
                dl_thread.start()

                # ──────────────────────────────────────────────────────────
                # CORREÇÃO #3: Monitorar CAMINHO EXATO do arquivo
                # ──────────────────────────────────────────────────────────
                # O hf_hub_download cria o arquivo em:
                #   {tmp_dir}/{caminho_dentro_do_repo}/{model_file}
                # Ex: {tmp_dir}/models--author--model/blobs/abc123
                # Em vez de os.walk() caro, monitoramos o tmp_dir
                # e verificamos o maior arquivo dentro dele.
                # ──────────────────────────────────────────────────────────
                _monitor_stop = threading.Event()
                monitor_thread = None

                if progress_callback is not None and total_bytes > 0:
                    def _monitor_size():
                        last_size = 0
                        while not _monitor_stop.is_set() and dl_thread.is_alive():
                            if cancel_event and cancel_event.is_set():
                                break
                            try:
                                # Varre a tmp_dir procurando o arquivo sendo baixado
                                # O hf_hub_download normalmente cria em subpastas
                                current_max = 0
                                for root, dirs, files in os.walk(tmp_dir):
                                    for f in files:
                                        fp = os.path.join(root, f)
                                        try:
                                            sz = os.path.getsize(fp)
                                            if sz > current_max:
                                                current_max = sz
                                        except OSError:
                                            pass
                                    if _monitor_stop.is_set():
                                        break
                                if current_max > last_size:
                                    last_size = current_max
                                    progress_callback(current_max, total_bytes)
                            except Exception:
                                pass
                            _monitor_stop.wait(1.0)
                        # Ao final, se o download terminou, sinaliza 100%
                        if not cancel_event or not cancel_event.is_set():
                            if last_size > 0:
                                progress_callback(total_bytes, total_bytes)

                    monitor_thread = threading.Thread(target=_monitor_size, daemon=True)
                    monitor_thread.start()

                # ──────────────────────────────────────────────────────────
                # CORREÇÃO #4: Verifica cancelamento durante download
                # ──────────────────────────────────────────────────────────
                # Aguarda o download terminar, mas checando cancel a cada 1s
                while dl_thread.is_alive():
                    if cancel_event and cancel_event.is_set():
                        print(f"[HF Node]    ⛔ Download cancelado pelo usuário")
                        # Para o monitor se estiver rodando
                        _monitor_stop.set()
                        _cleanup_tmp()
                        return None
                    dl_thread.join(timeout=1.0)

                # Para o monitor
                _monitor_stop.set()
                if monitor_thread and monitor_thread.is_alive():
                    monitor_thread.join(timeout=2)

                # Verifica se deu erro
                if download_error[0]:
                    raise download_error[0]

                downloaded_path = download_result[0]
                if not downloaded_path:
                    print(f"[HF Node]    ⚠️ hf_hub_download retornou None")
                    return None

                print(f"[HF Node]    Baixado: {os.path.basename(downloaded_path)}")

                # Encontra o arquivo baixado na temp
                src_file = None
                if os.path.isfile(downloaded_path):
                    src_file = downloaded_path
                else:
                    for root, dirs, files in os.walk(tmp_dir):
                        for f in files:
                            if f == os.path.basename(model_file) or f == expected_filename:
                                src_file = os.path.join(root, f)
                                break
                        if src_file:
                            break

                if not src_file or not os.path.isfile(src_file):
                    for root, dirs, files in os.walk(tmp_dir):
                        for f in files:
                            if any(f.lower().endswith(ext) for ext in ['.safetensors', '.ckpt', '.pt', '.pth', '.bin']):
                                src_file = os.path.join(root, f)
                                break
                        if src_file:
                            break

                if not src_file or not os.path.isfile(src_file):
                    print(f"[HF Node]    ⚠️ Não encontrou arquivo baixado na temp")
                    return None

                # Remove destino se já existir
                if os.path.exists(dest_path):
                    os.remove(dest_path)

                # MOVE no mesmo drive = RENOMEIO (instantâneo, sem cópia!)
                shutil.move(src_file, dest_path)
                print(f"[HF Node]    Movido → {expected_filename}")

            finally:
                # Limpa a pasta temporária (no mesmo drive = rápido)
                _cleanup_tmp()

            # Invalida cache do ComfyUI
            self._invalidate_comfy_cache(destination_folder)

            return dest_path

        except Exception as e:
            print(f"[HF Node] Erro download {model_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _invalidate_comfy_cache(self, folder_path: str):
        """Invalida o cache do folder_paths do ComfyUI para forçar redetecção."""
        try:
            import folder_paths
            # Determina o nome da pasta (ex: "loras", "checkpoints")
            folder_name = os.path.basename(folder_path.rstrip("/").rstrip("\\"))
            # Mapeia nomes legacy (ex: "unet" → "diffusion_models")
            folder_name = folder_paths.map_legacy(folder_name)
            # Limpa o cache
            folder_paths.filename_list_cache.pop(folder_name, None)
            # Força refresh
            folder_paths.get_filename_list(folder_name)
            print(f"[HF Node] 🔄 Cache ComfyUI invalidado para: {folder_name}")
        except Exception:
            pass  # Não crítico — o modelo eventualmente aparece

    def download_missing_model(self, missing: MissingModel, search_first: bool = True,
                                 progress_callback=None, cancel_event: threading.Event = None) -> str | None:
        """
        Baixa um modelo faltante. Retorna o caminho absoluto se sucesso, None se falha.
        
        Args:
            missing: MissingModel com filename, category, comfy_folder, full_path
            search_first: Se True, busca o model_id no HF antes de baixar
            progress_callback: Função opcional chamada com (downloaded_bytes, total_bytes)
            cancel_event: threading.Event — se setado, o download é abortado.
        """
        destination = str(self._comfyui_models_path / missing.comfy_folder)

        if search_first:
            model_id = self.search_best_match(missing.filename)
            if not model_id:
                print(f"[HF Node] ❌ Não encontrou '{missing.filename}' no HF")
                return None
        else:
            model_id = missing.filename

        return self.download_single_file(
            model_id, missing.filename, destination,
            progress_callback=progress_callback, cancel_event=cancel_event,
        )

    @staticmethod
    def parse_hf_url(url: str) -> Optional[dict]:
        """
        Analisa uma URL do HuggingFace e extrai repo_id e filepath.
        
        Formatos suportados:
        - https://huggingface.co/{repo_id}/blob/{branch}/{filepath}
        - https://huggingface.co/{repo_id}/resolve/{branch}/{filepath}
        - https://huggingface.co/{repo_id}
        
        Retorna dict com 'repo_id' e 'filepath', ou None se inválido.
        """
        url = url.strip()
        if not url:
            return None
        
        # Remove query params e fragmentos
        url = url.split('?')[0].split('#')[0]
        
        try:
            # Formato: https://huggingface.co/{repo_id}/blob/{branch}/{filepath}
            # ou        https://huggingface.co/{repo_id}/resolve/{branch}/{filepath}
            if '/blob/' in url or '/resolve/' in url:
                # Extrai a parte após 'huggingface.co/'
                prefix = '/blob/' if '/blob/' in url else '/resolve/'
                parts_after_domain = url.split('huggingface.co/')[-1]
                # parts_after_domain = "{repo_id}/blob/{branch}/{filepath}"
                repo_end = parts_after_domain.find(prefix)
                repo_id = parts_after_domain[:repo_end]
                filepath = parts_after_domain[repo_end + len(prefix):]
                # Remove o branch (primeira parte do filepath)
                branch_end = filepath.find('/')
                if branch_end != -1:
                    filepath = filepath[branch_end + 1:]
                return {"repo_id": repo_id, "filepath": filepath}
            
            # Formato: https://huggingface.co/{repo_id}
            elif 'huggingface.co/' in url:
                repo_id = url.split('huggingface.co/')[-1].strip('/')
                # Tenta encontrar o modelo no repositório automaticamente
                return {"repo_id": repo_id, "filepath": None}
            
            # Se for apenas o nome do modelo (author/repo)
            elif '/' in url and not url.startswith('http'):
                return {"repo_id": url, "filepath": None}
                
        except Exception:
            pass
        
        return None

    def download_from_url(self, url: str, destination_folder: str, filename: str = "", 
                          progress_callback=None, cancel_event: threading.Event = None) -> dict | None:
        """
        Baixa um modelo diretamente de uma URL do HuggingFace.
        
        Args:
            url: URL completa do HuggingFace
            destination_folder: Pasta de destino (ex: models/checkpoints)
            filename: Nome do arquivo (opcional - se vazio, usa o nome da URL)
            progress_callback: Função opcional chamada com (downloaded_bytes, total_bytes)
            cancel_event: threading.Event — se setado, o download é abortado.
        
        Retorna dict com 'path' e 'existed' (bool), ou None se falhar.
        """
        parsed = self.parse_hf_url(url)
        if not parsed:
            print(f"[HF Node] ❌ URL inválida: {url}")
            return None
        
        repo_id = parsed["repo_id"]
        filepath = parsed["filepath"]
        
        # Se não veio filepath da URL, tenta listar arquivos do repositório
        if not filepath:
            files = self._list_model_files(repo_id)
            if not files:
                print(f"[HF Node] ❌ Não encontrou arquivos em: {repo_id}")
                return None
            filepath = self._pick_model_file(files)
            if not filepath:
                print(f"[HF Node] ❌ Nenhum .safetensors/.ckpt em: {repo_id}")
                return None
        
        # Determina o nome do arquivo
        if not filename:
            filename = os.path.basename(filepath)
        
        print(f"[HF Node] 📥 URL → repo={repo_id}, file={filepath}")
        
        # --- Verifica se já existe ANTES de baixar ---
        os.makedirs(destination_folder, exist_ok=True)
        dest_path = os.path.join(destination_folder, filename)
        if os.path.exists(dest_path):
            print(f"[HF Node] ⚠️ Já existe em disco: {filename}")
            return {"path": dest_path, "existed": True}
        
        # Baixa passando progress_callback + cancel_event
        result = self.download_single_file(
            repo_id, filename, destination_folder, 
            repo_filepath=filepath, progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        
        if result:
            return {"path": result, "existed": False}
        return None
