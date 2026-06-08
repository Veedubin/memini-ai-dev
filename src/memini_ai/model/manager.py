"""Model manager - Singleton for embedding model lifecycle with GPU→CPU fallback."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from memini_ai.config import get_config

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


# Model identifiers
BGE_LARGE_MODEL_ID = "BAAI/bge-large-en-v1.5"
MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Model dimensions
BGE_LARGE_DIM = 1024
MINILM_DIM = 384


@dataclass
class ModelMetadata:
    """Metadata about the currently loaded model."""

    model_id: str
    device: str
    dimensions: int
    precision: str


class ModelManager:
    """Singleton model manager with GPU→CPU fallback and lazy loading.

    Reference counting tracks model lifecycle. Model loads on first acquire().
    Falls back from BGE-Large on CUDA to MiniLM on CPU if GPU unavailable.

    Model selection is constrained by config.embedding_dim to prevent
    dimension mismatches with the database vector column:
      - embedding_dim=384  → MiniLM (384-dim) only
      - embedding_dim=1024 → BGE-Large (1024-dim) only
      - other values       → RuntimeError at model load time

    This ensures MEMINI_EMBEDDING_DIM is authoritative, not decorative.
    """

    _instance: ModelManager | None = None
    _lock = asyncio.Lock()

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None
        self._model_id: str | None = None
        _config = get_config()
        self._device = _config.device
        self._precision = _config.precision
        self._use_gpu = _config.use_gpu
        # Bug 1 fix: ModelManager now reads config.embedding_dim to constrain
        # model selection. Without this, MEMINI_EMBEDDING_DIM was a no-op and
        # the 1024-dim BGE-Large model could be selected for a 384-dim DB.
        self._embedding_dim: int = _config.embedding_dim
        self._ref_count = 0
        self._dimensions: int | None = None

    @classmethod
    def get_instance(cls) -> ModelManager:
        """Get the singleton ModelManager instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def acquire(self) -> SentenceTransformer:
        """Acquire the model, loading it lazily on first call.

        Returns:
            The loaded SentenceTransformer model.

        Raises:
            RuntimeError: If model download fails.
        """
        async with self._lock:
            if self._model is None:
                await self._load_model()
            self._ref_count += 1
            if TYPE_CHECKING:
                assert self._model is not None
            return self._model

    def release(self) -> None:
        """Release the model (reference counting).

        Model is unloaded when ref_count reaches 0.
        """
        if self._ref_count > 0:
            self._ref_count -= 1
        if self._ref_count == 0 and self._model is not None:
            self.unload()

    async def _load_model(self) -> None:
        """Load the appropriate model based on config.embedding_dim and GPU.

        Model selection is constrained by config.embedding_dim (Approach A):
          - embedding_dim=384  → MiniLM only (GPU or CPU)
          - embedding_dim=1024 → BGE-Large only (GPU or CPU)
          - other values       → RuntimeError

        This prevents the 1024-dim BGE-Large model from being loaded when
        the database schema expects vector(384), which caused INSERT errors.
        """
        # Import here to avoid heavy import at module load time
        from sentence_transformers import SentenceTransformer

        # Bug 1 fix: Constrain model selection by embedding_dim.
        # MEMINI_EMBEDDING_DIM is now authoritative, not decorative.
        if self._embedding_dim == MINILM_DIM:
            # 384-dim: always use MiniLM, regardless of GPU availability
            model_id = MINILM_MODEL_ID
            model_dim = MINILM_DIM
            device = (
                "cuda"
                if (self._should_use_gpu() and self._check_cuda_available())
                else "cpu"
            )
        elif self._embedding_dim == BGE_LARGE_DIM:
            # 1024-dim: always use BGE-Large, regardless of GPU availability
            model_id = BGE_LARGE_MODEL_ID
            model_dim = BGE_LARGE_DIM
            # Prefer GPU if available, but BGE-Large on CPU is valid
            device = (
                "cuda"
                if (self._should_use_gpu() and self._check_cuda_available())
                else "cpu"
            )
        else:
            raise RuntimeError(
                f"Unsupported embedding_dim={self._embedding_dim}. "
                f"Must be {MINILM_DIM} (MiniLM) or {BGE_LARGE_DIM} (BGE-Large). "
                f"Set MEMINI_EMBEDDING_DIM accordingly."
            )

        try:
            self._model = await asyncio.to_thread(
                SentenceTransformer,
                model_id,
                device=device,
                cache_folder=self._get_cache_folder(),
            )
            self._model_id = model_id
            self._dimensions = model_dim
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{model_id}' on {device}. "
                f"Please check your internet connection and try again. "
                f"To download models manually, visit: "
                f"https://www.sbert.net/examples/applications/model-download/"
            ) from e

        # Bug 3 fix: Defense-in-depth assertion that the loaded model's
        # actual output dimension matches config.embedding_dim. This catches
        # mismatches even if the selection logic above is bypassed somehow.
        assert self._model is not None  # Guaranteed by try block above
        actual_dim = self._model.get_sentence_embedding_dimension()
        if actual_dim != self._embedding_dim:
            self._model = None
            self._model_id = None
            self._dimensions = None
            raise RuntimeError(
                f"Model dimension mismatch: model '{model_id}' produces "
                f"{actual_dim}-dim vectors but config.embedding_dim={self._embedding_dim}. "
                f"This is a configuration error — set MEMINI_EMBEDDING_DIM={actual_dim} "
                f"or choose a model that produces {self._embedding_dim}-dim vectors."
            )

    def _should_use_gpu(self) -> bool:
        """Determine if GPU should be used based on config."""
        if not self._use_gpu:
            return False
        device = os.environ.get("MEMINI_DEVICE", "").lower()
        return device not in ("cpu", "false", "no", "0")

    def _check_cuda_available(self) -> bool:
        """Check if CUDA is actually available via torch.

        Bug 2 fix: Previous implementation just checked `import torch` which
        returned True even on CPU-only machines. Now checks both
        torch.cuda.is_available() AND torch.cuda.device_count() > 0.
        """
        try:
            import torch

            return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
        except Exception:
            return False

    def _get_cache_folder(self) -> str | None:
        """Get the model cache folder from env var or None for default."""
        cache = os.environ.get("SENTENCE_TRANSFORMERS_CACHE")
        return cache

    def get_dimensions(self) -> int:
        """Get the embedding dimension of the loaded model.

        Returns:
            The embedding dimension (1024 for BGE-Large, 384 for MiniLM).
        """
        if self._dimensions is None:
            raise RuntimeError("Model not loaded. Call acquire() first.")
        return self._dimensions

    def get_metadata(self) -> ModelMetadata:
        """Get metadata about the currently loaded model.

        Returns:
            ModelMetadata with model details.

        Raises:
            RuntimeError: If model not loaded.
        """
        if self._model_id is None or self._dimensions is None:
            raise RuntimeError("Model not loaded. Call acquire() first.")
        return ModelMetadata(
            model_id=self._model_id,
            device=self._device,
            dimensions=self._dimensions,
            precision=self._precision,
        )

    def unload(self) -> None:
        """Unload the model from memory."""
        self._model = None
        self._model_id = None
        self._dimensions = None

    def is_gpu_available(self) -> bool:
        """Check if a GPU is available for embedding generation.

        Returns:
            True if GPU (CUDA) is available, False otherwise.
        """
        return self._check_cuda_available()
