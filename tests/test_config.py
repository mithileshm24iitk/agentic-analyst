"""Tests for settings loading."""

import os

from app.config import Settings


def test_settings_export_to_environ(monkeypatch) -> None:
    for name in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GOOGLE_GENAI_USE_ENTERPRISE"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(
        _env_file=None,
        google_cloud_project="test-project",
        google_cloud_location="global",
        worker_model="some-model",
    )
    settings.export_to_environ()

    assert os.environ["GOOGLE_CLOUD_PROJECT"] == "test-project"
    assert os.environ["GOOGLE_CLOUD_LOCATION"] == "global"
    assert os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] == "true"
