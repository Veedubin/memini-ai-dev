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
        """Load the appropriate model based on GPU availability.

        Tries BGE-Large on CUDA first, falls back to MiniLM on CPU.
        """
        # Import here to avoid heavy import at module load time
        from sentence_transformers import SentenceTransformer

        # Check GPU availability
        if self._should_use_gpu():
            gpu_available = self._check_cuda_available()
        else:
            gpu_available = False

        if gpu_available:
            # Try BGE-Large on GPU
            try:
                self._model = await asyncio.to_thread(
                    SentenceTransformer,
                    BGE_LARGE_MODEL_ID,
                    device="cuda",
                    cache_folder=self._get_cache_folder(),
                )
                self._model_id = BGE_LARGE_MODEL_ID
                self._dimensions = BGE_LARGE_DIM
                return
            except Exception:
                # Fall through to MiniLM fallback
                pass

        # Fallback to MiniLM on CPU
        try:
            self._model = await asyncio.to_thread(
                SentenceTransformer,
                MINILM_MODEL_ID,
                device="cpu",
                cache_folder=self._get_cache_folder(),
            )
            self._model_id = MINILM_MODEL_ID
            self._dimensions = MINILM_DIM
            return
        except Exception as e:
            raise RuntimeError(
                "Failed to load embedding model. "
                "Please check your internet connection and try again. "
                "To download models manually, visit: "
                "https://www.sbert.net/examples/applications/model-download/"
            ) from e

    def _should_use_gpu(self) -> bool:
        """Determine if GPU should be used based on config."""
        if not self._use_gpu:
            return False
        device = os.environ.get("MEMINI_DEVICE", "").lower()
        return device not in ("cpu", "false", "no", "0")

    def _check_cuda_available(self) -> bool:
        """Check if CUDA is available via torch."""
        try:
            import torch  # noqa: F401

            return True
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
