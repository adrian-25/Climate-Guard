"""Tests for local .env loading without deployment-variable overrides."""

import os
from pathlib import Path

from src.settings import load_project_environment


def test_local_env_file_is_loaded(tmp_path: Path, monkeypatch) -> None:
    """A project's .env supplies a value when the process has none."""

    monkeypatch.delenv("CLIMATEGUARD_TEST_SETTING", raising=False)
    (tmp_path / ".env").write_text("CLIMATEGUARD_TEST_SETTING=from-file\n", encoding="utf-8")

    assert load_project_environment(tmp_path) is True
    assert os.environ["CLIMATEGUARD_TEST_SETTING"] == "from-file"


def test_process_environment_overrides_local_env(tmp_path: Path, monkeypatch) -> None:
    """Hosted configuration always wins over a local .env value."""

    monkeypatch.setenv("CLIMATEGUARD_TEST_SETTING", "from-host")
    (tmp_path / ".env").write_text("CLIMATEGUARD_TEST_SETTING=from-file\n", encoding="utf-8")

    assert load_project_environment(tmp_path) is True
    assert os.environ["CLIMATEGUARD_TEST_SETTING"] == "from-host"
