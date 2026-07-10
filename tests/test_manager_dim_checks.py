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
    BGE_M3_DIM,
    BGE_M3_MODEL_ID,
    MINILM_DIM,
    MINILM_MODEL_ID,
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
    strict_embedding_dim: bool = False,
    auto_detect_model: bool = True,
) -> MagicMock:
    """Create a mock config with the given settings."""
    cfg = MagicMock()
    cfg.embedding_dim = embedding_dim
    cfg.use_gpu = use_gpu
    cfg.device = device
    cfg.precision = "fp16"
    cfg.model_name = model_name
    cfg.strict_embedding_dim = strict_embedding_dim
    cfg.auto_detect_model = auto_detect_model
    return cfg


def _make_mock_transformer(dim: int) -> MagicMock:
    """Create a mock SentenceTransformer that reports the given dimension."""
    mock = MagicMock()
    mock.get_embedding_dimension.return_value = dim
    # Keep the old method for backwards-compat tests
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
        picked a fixed 1024-dim model, so BGE-M3 could never be loaded.
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

        v0.7.7: The default is now lenient (warn + degrade). This test sets
        strict_embedding_dim=True to verify the old crash behavior is still
        available via the env var opt-in.
        """
        # Simulate: config says 384, but model reports 1024
        mock_transformer = _make_mock_transformer(1024)  # Model says 1024

        ModelManager._instance = None
        with patch(
            "memini_ai.model.manager.get_config",
            return_value=_make_config_mock(
                embedding_dim=384,
                strict_embedding_dim=True,
            ),
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


# ---------------------------------------------------------------------------
# v0.7.7: Lenient dim mismatch (default behavior)
# ---------------------------------------------------------------------------


class TestLenientDimMismatch:
    """Tests for the default lenient dim-mismatch behavior (v0.7.7)."""

    @pytest.mark.asyncio
    async def test_lenient_mismatch_does_not_raise(self) -> None:
        """When strict_embedding_dim=False (default), a dim mismatch logs a
        warning and sets _has_dim_mismatch=True instead of raising.
        """
        mock_transformer = _make_mock_transformer(1024)  # Model says 1024

        ModelManager._instance = None
        with patch(
            "memini_ai.model.manager.get_config",
            return_value=_make_config_mock(
                embedding_dim=384,
                strict_embedding_dim=False,
            ),
        ):
            manager = ModelManager()

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_transformer,
        ):
            # Should NOT raise
            await manager._load_model()

        # Model is still loaded, but mismatch flag is set
        assert manager._model is not None
        assert manager._has_dim_mismatch is True
        assert manager.has_dim_mismatch is True
        # The model_id is still set (the model loaded successfully)
        assert manager._model_id is not None
        ModelManager._instance = None


# ---------------------------------------------------------------------------
# v0.7.7: Strict dim mismatch (opt-in via MEMINI_STRICT_EMBEDDING_DIM=true)
# ---------------------------------------------------------------------------


class TestStrictDimMismatch:
    """Tests for strict dim-mismatch behavior (opt-in via env var)."""

    @pytest.mark.asyncio
    async def test_strict_mismatch_raises_runtime_error(self) -> None:
        """When strict_embedding_dim=True, a dim mismatch raises RuntimeError."""
        mock_transformer = _make_mock_transformer(1024)

        ModelManager._instance = None
        with patch(
            "memini_ai.model.manager.get_config",
            return_value=_make_config_mock(
                embedding_dim=384,
                strict_embedding_dim=True,
            ),
        ):
            manager = ModelManager()

        with (
            patch(
                "sentence_transformers.SentenceTransformer",
                return_value=mock_transformer,
            ),
            pytest.raises(RuntimeError, match="Model dimension mismatch"),
        ):
            await manager._load_model()

        # State should be cleaned up
        assert manager._model is None
        assert manager._has_dim_mismatch is False
        ModelManager._instance = None


# ---------------------------------------------------------------------------
# v0.7.7: Auto-detect model (new deployment → BGE-M3)
# ---------------------------------------------------------------------------


class TestAutoDetect:
    """Tests for auto_detect_model classmethod."""

    @pytest.mark.asyncio
    async def test_empty_db_selects_bge_m3(self) -> None:
        """When memory_count=0 and model_name is default, returns True and
        switches to BGE-M3.
        """
        from memini_ai.model.manager import DEFAULT_MODEL_NAME

        cfg = MagicMock()
        cfg.auto_detect_model = True
        cfg.model_name = DEFAULT_MODEL_NAME
        cfg.embedding_dim = 384

        with patch("memini_ai.model.manager.get_config", return_value=cfg):
            result = await ModelManager.auto_detect_model(memory_count=0)

        assert result is True
        assert cfg.model_name == BGE_M3_MODEL_ID
        assert cfg.embedding_dim == BGE_M3_DIM

    @pytest.mark.asyncio
    async def test_populated_db_keeps_minilm(self) -> None:
        """When memory_count > 0, auto-detect does NOT override (returns False)."""
        from memini_ai.model.manager import DEFAULT_MODEL_NAME

        cfg = MagicMock()
        cfg.auto_detect_model = True
        cfg.model_name = DEFAULT_MODEL_NAME
        cfg.embedding_dim = 384

        with patch("memini_ai.model.manager.get_config", return_value=cfg):
            result = await ModelManager.auto_detect_model(memory_count=100)

        assert result is False
        # model_name unchanged
        assert cfg.model_name == DEFAULT_MODEL_NAME

    @pytest.mark.asyncio
    async def test_user_set_model_name_respected(self) -> None:
        """When model_name != default, auto-detect respects the user's choice."""
        cfg = MagicMock()
        cfg.auto_detect_model = True
        cfg.model_name = BGE_M3_MODEL_ID  # User explicitly set BGE-M3
        cfg.embedding_dim = 1024

        with patch("memini_ai.model.manager.get_config", return_value=cfg):
            result = await ModelManager.auto_detect_model(memory_count=0)

        assert result is False
        assert cfg.model_name == BGE_M3_MODEL_ID  # unchanged

    @pytest.mark.asyncio
    async def test_auto_detect_disabled_returns_false(self) -> None:
        """When auto_detect_model=False, auto-detect is skipped."""
        from memini_ai.model.manager import DEFAULT_MODEL_NAME

        cfg = MagicMock()
        cfg.auto_detect_model = False
        cfg.model_name = DEFAULT_MODEL_NAME
        cfg.embedding_dim = 384

        with patch("memini_ai.model.manager.get_config", return_value=cfg):
            result = await ModelManager.auto_detect_model(memory_count=0)

        assert result is False
        assert cfg.model_name == DEFAULT_MODEL_NAME  # unchanged


# ---------------------------------------------------------------------------
# v0.7.7: Deprecation fix — get_embedding_dimension (not get_sentence_embedding_dimension)
# ---------------------------------------------------------------------------


class TestDeprecationFix:
    """Tests confirming the deprecated method is NOT called."""

    @pytest.mark.asyncio
    async def test_uses_get_embedding_dimension_not_deprecated(self) -> None:
        """The _load_model path must call get_embedding_dimension (the new
        method), NOT get_sentence_embedding_dimension (deprecated in
        sentence-transformers 3.x and emits a FutureWarning).
        """
        mock_transformer = MagicMock()
        # Set up both methods, but we want to assert only the new one is called
        mock_transformer.get_embedding_dimension.return_value = MINILM_DIM
        mock_transformer.get_sentence_embedding_dimension.return_value = MINILM_DIM

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

        # The new method must have been called
        mock_transformer.get_embedding_dimension.assert_called_once()
        # The deprecated method must NOT have been called
        mock_transformer.get_sentence_embedding_dimension.assert_not_called()
        ModelManager._instance = None
