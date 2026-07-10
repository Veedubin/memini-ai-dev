"""Tests for ModelManager dimension checks and CUDA validation.

Covers:
  - Model selection driven by config.model_name (MEMINI_MODEL_NAME), not
    constrained by embedding_dim.  BGE-M3 can now be loaded.
  - _check_cuda_available() no longer false-positives when torch is
    importable but CUDA is unavailable.
  - Runtime dim assertion: loaded model's output dim must match
    config.embedding_dim (sanity check against DB column width).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from memini_ai.model.manager import (
    BGE_LARGE_DIM,
    BGE_LARGE_MODEL_ID,
    BGE_M3_DIM,
    BGE_M3_MODEL_ID,
    MINILM_DIM,
    MINILM_MODEL_ID,
    MODEL_COLUMNS,
    MODEL_DIMS,
    ModelManager,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _make_config_mock(
    embedding_dim: int = 384,
    use_gpu: bool = False,
    device: str = "cpu",
    model_name: str = MINILM_MODEL_ID,
) -> MagicMock:
    """Create a mock config with the given settings."""
    cfg = MagicMock()
    cfg.embedding_dim = embedding_dim
    cfg.use_gpu = use_gpu
    cfg.device = device
    cfg.precision = "fp16"
    cfg.model_name = model_name
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
# Model selection driven by config.model_name
# ---------------------------------------------------------------------------


class TestModelSelectionByName:
    """Tests for model selection driven by config.model_name."""

    @pytest.mark.asyncio
    async def test_load_model_minilm_when_model_name_is_minilm(self) -> None:
        """When model_name=MiniLM and embedding_dim=384, MiniLM is loaded."""
        mock_transformer = _make_mock_transformer(MINILM_DIM)

        ModelManager._instance = None
        with patch(
            "memini_ai.model.manager.get_config",
            return_value=_make_config_mock(
                embedding_dim=384,
                use_gpu=True,
                model_name=MINILM_MODEL_ID,
            ),
        ):
            manager = ModelManager()

        # Mock CUDA as available (torch importable + cuda available)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1

        with (
            patch(
                "sentence_transformers.SentenceTransformer",
                return_value=mock_transformer,
            ) as mock_st,
            patch.dict(sys.modules, {"torch": mock_torch}),
        ):
            await manager._load_model()

        # Verify MiniLM was selected
        assert manager._model_id == MINILM_MODEL_ID
        assert manager._dimensions == MINILM_DIM
        mock_st.assert_called_once()
        call_args = mock_st.call_args
        assert call_args[0][0] == MINILM_MODEL_ID
        ModelManager._instance = None

    @pytest.mark.asyncio
    async def test_load_model_bge_m3_when_model_name_is_bge_m3(self) -> None:
        """When model_name=BAAI/bge-m3 and embedding_dim=1024, BGE-M3 is loaded.

        This is the key regression: previously the 1024-dim branch always
        picked BGE-Large, so BGE-M3 could never be loaded.
        """
        mock_transformer = _make_mock_transformer(BGE_M3_DIM)

        ModelManager._instance = None
        with patch(
            "memini_ai.model.manager.get_config",
            return_value=_make_config_mock(
                embedding_dim=1024,
                use_gpu=False,
                model_name=BGE_M3_MODEL_ID,
            ),
        ):
            manager = ModelManager()

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_transformer,
        ) as mock_st:
            await manager._load_model()

        assert manager._model_id == BGE_M3_MODEL_ID
        assert manager._dimensions == BGE_M3_DIM
        mock_st.assert_called_once()
        call_args = mock_st.call_args
        assert call_args[0][0] == BGE_M3_MODEL_ID
        ModelManager._instance = None

    @pytest.mark.asyncio
    async def test_load_model_bge_large_when_model_name_is_bge_large(self) -> None:
        """When model_name=BAAI/bge-large-en-v1.5 and embedding_dim=1024, BGE-Large is loaded."""
        mock_transformer = _make_mock_transformer(BGE_LARGE_DIM)

        ModelManager._instance = None
        with patch(
            "memini_ai.model.manager.get_config",
            return_value=_make_config_mock(
                embedding_dim=1024,
                use_gpu=False,
                model_name=BGE_LARGE_MODEL_ID,
            ),
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
    async def test_load_model_custom_name_passes_through(self) -> None:
        """An unknown model_name is passed through as a custom HF model ID."""
        custom_model = "intfloat/multilingual-e5-large"
        custom_dim = 1024
        mock_transformer = _make_mock_transformer(custom_dim)

        ModelManager._instance = None
        with patch(
            "memini_ai.model.manager.get_config",
            return_value=_make_config_mock(
                embedding_dim=custom_dim,
                model_name=custom_model,
            ),
        ):
            manager = ModelManager()

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_transformer,
        ) as mock_st:
            await manager._load_model()

        assert manager._model_id == custom_model
        assert manager._dimensions == custom_dim
        mock_st.assert_called_once()
        call_args = mock_st.call_args
        assert call_args[0][0] == custom_model
        ModelManager._instance = None

    @pytest.mark.asyncio
    async def test_load_model_short_alias_bge_m3(self) -> None:
        """Short alias 'bge-m3' normalizes to BAAI/bge-m3."""
        mock_transformer = _make_mock_transformer(BGE_M3_DIM)

        ModelManager._instance = None
        with patch(
            "memini_ai.model.manager.get_config",
            return_value=_make_config_mock(
                embedding_dim=1024,
                model_name="bge-m3",
            ),
        ):
            manager = ModelManager()

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_transformer,
        ) as mock_st:
            await manager._load_model()

        assert manager._model_id == BGE_M3_MODEL_ID
        call_args = mock_st.call_args
        assert call_args[0][0] == BGE_M3_MODEL_ID
        ModelManager._instance = None


# ---------------------------------------------------------------------------
# Dim assertion on load (sanity check)
# ---------------------------------------------------------------------------


class TestDimAssertionOnLoad:
    """Tests for runtime dim assertion: loaded model dim must match config."""

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
        with (
            patch(
                "sentence_transformers.SentenceTransformer",
                return_value=mock_transformer,
            ),
            pytest.raises(RuntimeError, match="Model dimension mismatch"),
        ):
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
