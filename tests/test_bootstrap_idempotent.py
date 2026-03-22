"""Test that bootstrap skips rebuilding when the manifest matches."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_settings(tmpdir: str):
    """Create a minimal Settings-like object pointing at temp directories."""
    from src.app.settings import Settings

    return Settings(
        octus_raw_dir=str(Path(tmpdir) / "raw" / "octus"),
        processed_dir=str(Path(tmpdir) / "processed"),
        cache_dir=str(Path(tmpdir) / "cache"),
        simfin_api_key="",
        pinecone_api_key="",
        embedding_provider="mock",
        vectorstore_provider="faiss",
        simfin_mode="batch",
    )


def _write_fake_manifest(settings, fingerprint: str) -> None:
    manifest_path = settings.octus_processed_path / "bootstrap_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "fingerprint": fingerprint,
        "built_at": "2024-01-01T00:00:00",
        "settings_subset": settings.fingerprint_subset(),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _write_fake_artifacts(settings) -> None:
    """Create stub artifact files so _artifacts_exist() returns True."""
    out = settings.octus_processed_path
    fmt = settings.table_format
    out.mkdir(parents=True, exist_ok=True)
    for name in [f"documents.{fmt}", f"chunks_heading.{fmt}", f"chunks_token.{fmt}"]:
        (out / name).write_bytes(b"")


class TestBootstrapIdempotency:
    def test_no_rebuild_when_fingerprint_matches(self):
        """
        ensure_ready should NOT call _run_pipeline when:
          - bootstrap_manifest.json exists with matching fingerprint
          - all artifact files exist
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)

            # Compute the fingerprint that bootstrap would compute
            from src.app.fingerprints import compute_fingerprint
            # Raw files don't exist → fingerprint reflects MISSING state
            fingerprint = compute_fingerprint(settings)

            # Pre-write matching manifest + artifacts
            _write_fake_manifest(settings, fingerprint)
            _write_fake_artifacts(settings)

            with patch("src.app.bootstrap._run_pipeline") as mock_pipeline, \
                 patch("src.app.bootstrap._make_vectorstore") as mock_vs, \
                 patch("src.app.bootstrap._make_retriever") as mock_ret, \
                 patch("src.app.bootstrap._make_simfin_client") as mock_sf, \
                 patch("src.app.bootstrap._make_orchestrator") as mock_orch:

                mock_vs.return_value = MagicMock()
                mock_ret.return_value = MagicMock()
                mock_sf.return_value = None
                mock_orch.return_value = MagicMock()

                from src.app.bootstrap import ensure_ready
                ctx = ensure_ready(settings)

                # Pipeline must NOT have been called
                mock_pipeline.assert_not_called()

    def test_rebuild_when_fingerprint_mismatch(self):
        """ensure_ready should call _run_pipeline when fingerprint differs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)

            # Write a WRONG fingerprint
            _write_fake_manifest(settings, "wrong_fingerprint_value")
            _write_fake_artifacts(settings)

            with patch("src.app.bootstrap._run_pipeline") as mock_pipeline, \
                 patch("src.app.bootstrap._make_vectorstore") as mock_vs, \
                 patch("src.app.bootstrap._make_retriever") as mock_ret, \
                 patch("src.app.bootstrap._make_simfin_client") as mock_sf, \
                 patch("src.app.bootstrap._make_orchestrator") as mock_orch:

                mock_vs.return_value = MagicMock()
                mock_ret.return_value = MagicMock()
                mock_sf.return_value = None
                mock_orch.return_value = MagicMock()

                from src.app.bootstrap import ensure_ready
                ctx = ensure_ready(settings)

                # Pipeline must have been called exactly once
                mock_pipeline.assert_called_once_with(settings)

    def test_rebuild_when_no_manifest(self):
        """ensure_ready should call _run_pipeline when no manifest exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            # No manifest, no artifacts

            with patch("src.app.bootstrap._run_pipeline") as mock_pipeline, \
                 patch("src.app.bootstrap._make_vectorstore") as mock_vs, \
                 patch("src.app.bootstrap._make_retriever") as mock_ret, \
                 patch("src.app.bootstrap._make_simfin_client") as mock_sf, \
                 patch("src.app.bootstrap._make_orchestrator") as mock_orch:

                mock_vs.return_value = MagicMock()
                mock_ret.return_value = MagicMock()
                mock_sf.return_value = None
                mock_orch.return_value = MagicMock()

                from src.app.bootstrap import ensure_ready
                ctx = ensure_ready(settings)

                mock_pipeline.assert_called_once()
