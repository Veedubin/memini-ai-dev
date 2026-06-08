"""Tests for ModelManager dimension checks and CUDA validation.

Covers three bug fixes:
  Bug 1: MEMINI_EMBEDDING_DIM was decorative — ModelManager now constrains
         model selection by config.embedding_dim (Approach A).
  Bug 2: _check_cuda_available() falsely returned True when torch was
         importable but CUDA was not available.
  Bug 3: No runtime assertion on first model acquire — now validates
         that loaded model's output dim matches config.embedding_dim.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from memini_ai.model.manager import (
    BGE_LARGE_DIM,
    BGE_LARGE_MODEL_ID,
    MINILM_DIM,
    MINILM_MODEL_ID,
    ModelManager,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _make_config_mock(
    embedding_dim: int = 384, use_gpu: bool = False, device: str = "cpu"
) -> MagicMock:
    """Create a mock config with the given embedding_dim."""
    cfg = MagicMock()
    cfg.embedding_dim = embedding_dim
    cfg.use_gpu = use_gpu
    cfg.device = device
    cfg.precision = "fp16"
    return cfg


def _make_mock_transformer(dim: int) -> MagicMock:
    """Create a mock SentenceTransformer that reports the given dimension."""
    mock = MagicMock()
    mock.get_sentence_embedding_dimension.return_value = dim
    return mock


# ---------------------------------------------------------------------------
# Bug 2 tests: _check_cuda_available no longer false-positive
# ---------------------------------------------------------------------------


class TestCheckCudaAvailable:
    """Tests for Bug 2: _check_cuda_available must check actual CUDA."""

    def test_check_cuda_available_no_torch_returns_false(self) -> None:
        """When torch cannot be imported, _check_cuda_available returns False."""
        ModelManager._instance = None
        with patch(
            "memini_ai.model.manager.get_config", return_value=_make_config_mock()
        ):
            manager = ModelManager()

        with patch.dict(sys.modules, {"torch": None}):
            # When torch is not importable (module set to None in sys.modules),
            # import will fail and _check_cuda_available should return False
            result = manager._check_cuda_available()
        assert result is False
        ModelManager._instance = None

    def test_check_cuda_available_no_cuda_returns_false(self) -> None:
        """When torch is importable but CUDA is unavailable, return False."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.cuda.device_count.return_value = 0

        ModelManager._instance = None
        with patch(
            "memini_ai.model.manager.get_config", return_value=_make_config_mock()
        ):
            manager = ModelManager()

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = manager._check_cuda_available()
        assert result is False
        ModelManager._instance = None

    def test_check_cuda_available_with_cuda_returns_true(self) -> None:
        """When torch is importable and CUDA is available, return True."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1

        ModelManager._instance = None
        with patch(
            "memini_ai.model.manager.get_config", return_value=_make_config_mock()
        ):
            manager = ModelManager()

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = manager._check_cuda_available()
        assert result is True
        ModelManager._instance = None


# ---------------------------------------------------------------------------
# Bug 1 tests: Model selection constrained by config.embedding_dim
# ---------------------------------------------------------------------------


class TestModelSelectionByDim:
    """Tests for Bug 1: Model selection is constrained by embedding_dim."""

    @pytest.mark.asyncio
    async def test_load_model_384_mode_uses_minilm_even_with_torch(self) -> None:
        """When embedding_dim=384 and use_gpu=True, MiniLM is selected even if CUDA is available.

        This is the core bug: previously, having torch installed + use_gpu=True would
        always select BGE-Large (1024-dim), ignoring MEMINI_EMBEDDING_DIM=384.
        Now, embedding_dim=384 forces MiniLM selection.
        """
        mock_transformer = _make_mock_transformer(MINILM_DIM)

        ModelManager._instance = None
        with patch(
            "memini_ai.model.manager.get_config",
            return_value=_make_config_mock(embedding_dim=384, use_gpu=True),
        ):
            manager = ModelManager()

        # Mock CUDA as available (torch importable + cuda available)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1

        # SentenceTransformer is imported locally inside _load_model,
        # so we must patch it at the source module.
        with (
            patch(
                "sentence_transformers.SentenceTransformer",
                return_value=mock_transformer,
            ) as mock_st,
            patch.dict(sys.modules, {"torch": mock_torch}),
        ):
            await manager._load_model()

        # Verify MiniLM was selected, not BGE-Large
        assert manager._model_id == MINILM_MODEL_ID
        assert manager._dimensions == MINILM_DIM
        # Verify the model was constructed with MiniLM's model ID
        mock_st.assert_called_once()
        call_args = mock_st.call_args
        assert call_args[0][0] == MINILM_MODEL_ID
        ModelManager._instance = None

    @pytest.mark.asyncio
    async def test_load_model_1024_mode_uses_bge_large(self) -> None:
        """When embedding_dim=1024, BGE-Large is selected on CPU when no GPU."""
        mock_transformer = _make_mock_transformer(BGE_LARGE_DIM)

        ModelManager._instance = None
        with patch(
            "memini_ai.model.manager.get_config",
            return_value=_make_config_mock(embedding_dim=1024, use_gpu=False),
        ):
            manager = ModelManager()

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_transformer,
        ) as mock_st:
            await manager._load_model()

        assert manager._model_id == BGE_LARGE_MODEL_ID
        assert manager._dimensions == BGE_LARGE_DIM
        mock_st.assert_called_once()
        call_args = mock_st.call_args
        assert call_args[0][0] == BGE_LARGE_MODEL_ID
        ModelManager._instance = None

    @pytest.mark.asyncio
    async def test_load_model_unsupported_dim_raises(self) -> None:
        """When embedding_dim is not 384 or 1024, RuntimeError is raised."""
        ModelManager._instance = None
        with patch(
            "memini_ai.model.manager.get_config",
            return_value=_make_config_mock(embedding_dim=512),
        ):
            manager = ModelManager()

        with (
            patch(
                "sentence_transformers.SentenceTransformer",
                side_effect=RuntimeError("should not be called"),
            ),
            pytest.raises(RuntimeError, match="Unsupported embedding_dim=512"),
        ):
            await manager._load_model()
        ModelManager._instance = None


# ---------------------------------------------------------------------------
# Bug 3 tests: Runtime dim assertion on model load
# ---------------------------------------------------------------------------


class TestDimAssertionOnLoad:
    """Tests for Bug 3: Runtime assertion that model dim matches config."""

    @pytest.mark.asyncio
    async def test_load_model_mismatched_dim_raises(self) -> None:
        """If model reports different dim than config.embedding_dim, RuntimeError is raised.

        This is a defense-in-depth check: even if the model selection logic
        somehow picks the wrong model, the assertion catches the mismatch.
        """
        # Simulate: config says 384, but model reports 1024
        mock_transformer = _make_mock_transformer(1024)  # Model says 1024

        ModelManager._instance = None
        with patch(
            "memini_ai.model.manager.get_config",
            return_value=_make_config_mock(embedding_dim=384),
        ):
            manager = ModelManager()

        # Patch MiniLM selection to load but report wrong dim
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_transformer,
        ), pytest.raises(RuntimeError, match="Model dimension mismatch"):
            await manager._load_model()

        # Verify state was cleaned up after failure
        assert manager._model is None
        assert manager._model_id is None
        assert manager._dimensions is None
        ModelManager._instance = None

    @pytest.mark.asyncio
    async def test_load_model_matching_dim_succeeds(self) -> None:
        """When model dim matches config.embedding_dim, load succeeds normally."""
        mock_transformer = _make_mock_transformer(MINILM_DIM)

        ModelManager._instance = None
        with patch(
            "memini_ai.model.manager.get_config",
            return_value=_make_config_mock(embedding_dim=384),
        ):
            manager = ModelManager()

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_transformer,
        ):
            await manager._load_model()

        assert manager._model_id == MINILM_MODEL_ID
        assert manager._dimensions == MINILM_DIM
        assert manager._model is mock_transformer
        ModelManager._instance = None
