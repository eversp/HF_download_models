"""Gerenciador de configurações do node HF Model Downloader."""

import json
import os
from pathlib import Path
from typing import Optional

from .hf_constants import CONFIG_FILE


class Config:
    """Gerencia configurações persistentes do node (token HF)."""

    def __init__(self):
        # Salva config na mesma pasta do node
        self.config_path = Path(__file__).parent / CONFIG_FILE
        self._data: dict = {}
        self.load()

    def load(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._data = {}
        else:
            self._data = {}

        self._data.setdefault("hf_token", "")
        self._data.setdefault("hf_username", "")

    def save(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def get_hf_token(self) -> Optional[str]:
        token = self._data.get("hf_token", "")
        return token if token else None

    def set_hf_token(self, token: str):
        self._data["hf_token"] = token
        self.save()

    def get_hf_username(self) -> str:
        return self._data.get("hf_username", "")

    def set_hf_username(self, username: str):
        self._data["hf_username"] = username
        self.save()

    def is_logged_in(self) -> bool:
        return bool(self._data.get("hf_token", ""))