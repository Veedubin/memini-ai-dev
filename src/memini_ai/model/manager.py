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
BGE_M3_MODEL_ID = "BAAI/bge-m3"
MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Model dimensions
BGE_M3_DIM = 1024
MINILM_DIM = 384

# Model name → dimension mapping
# Production-supported models: MiniLM (384-dim) and BGE-M3 (1024-dim).
MODEL_DIMS: dict[str, int] = {
    MINILM_MODEL_ID: MINILM_DIM,
    BGE_M3_MODEL_ID: BGE_M3_DIM,
}

# Model name → column name in memories table
MODEL_COLUMNS: dict[str, str] = {
    MINILM_MODEL_ID: "embedding",
    BGE_M3_MODEL_ID: "embedding_bge_m3",
}

# Short-name aliases → canonical HuggingFace model IDs.
# Users may set MEMINI_MODEL_NAME=bge-m3 instead of the full BAAI/bge-m3.
_MODEL_ALIASES: dict[str, str] = {
    "all-MiniLM-L6-v2": MINILM_MODEL_ID,
    "minilm": MINILM_MODEL_ID,
    "minilm-l6-v2": MINILM_MODEL_ID,
    "bge-m3": BGE_M3_MODEL_ID,
    MINILM_MODEL_ID: MINILM_MODEL_ID,
    BGE_M3_MODEL_ID: BGE_M3_MODEL_ID,
}


def _normalize_model_name(name: str) -> str:
    """Normalize a user-provided model name to its canonical HF ID.

    Accepts full HuggingFace IDs (sentence-transformers/all-MiniLM-L6-v2)
    and short aliases (minilm, bge-m3).  Unknown values are returned
    unchanged so custom HF models can be loaded by name, but they must
    produce either 384 or 1024 dim vectors to match the DB schema.
    """
    key = name.strip()
    return _MODEL_ALIASES.get(key, key)


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

    Model selection is driven by ``config.model_name`` (``MEMINI_MODEL_NAME``):
      - ``sentence-transformers/all-MiniLM-L6-v2`` (alias: ``minilm``) → 384-dim
      - ``BAAI/bge-m3`` (alias: ``bge-m3``) → 1024-dim
      - any other value is treated as a custom HuggingFace model name (must
        produce either 384 or 1024 dim vectors to match the DB schema)

    ``config.embedding_dim`` is kept as a **sanity check**: after the model
    is loaded, a dimension-mismatch assertion ensures the model's output dim
    matches what the database vector column expects.
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
        # a 1024-dim model could be selected for a 384-dim DB.
        self._embedding_dim: int = _config.embedding_dim
        self._ref_count = 0
        self._dimensions: int | None = None
        # Multi-model support (v0.12.0+): lazy-loaded model cache
        self._model_cache: dict[str, SentenceTransformer] = {}
        self._active_model_name: str = _normalize_model_name(_config.model_name)

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
        """Load the model named by ``config.model_name`` (``MEMINI_MODEL_NAME``).

        Model selection is driven by the configured model name, NOT by
        ``embedding_dim``.  Short aliases (``minilm``, ``bge-m3``) are
        accepted.  Unknown names are treated as custom HuggingFace model IDs
        (but must produce either 384 or 1024 dim vectors to match the DB
        schema).

        After loading, a dimension-mismatch assertion ensures the model's
        output dim matches ``config.embedding_dim`` — this is the sanity
        check that prevents writing 1024-dim vectors to a 384-dim column.
        """
        # Import here to avoid heavy import at module load time
        from sentence_transformers import SentenceTransformer

        # Pick model based on model_name, not embedding_dim
        model_id = self._active_model_name
        model_dim = MODEL_DIMS.get(model_id, self._embedding_dim)

        device = (
            "cuda"
            if (self._should_use_gpu() and self._check_cuda_available())
            else "cpu"
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

        # Defense-in-depth assertion that the loaded model's actual output
        # dimension matches config.embedding_dim. This catches mismatches
        # even if the user picks a model whose dim doesn't match the DB column.
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
            The embedding dimension (1024 for BGE-M3, 384 for MiniLM).
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

    # =========================================================================
    # Multi-model support (v0.12.0+)
    # =========================================================================

    async def get_model_for(self, model_name: str) -> SentenceTransformer:
        """Get (or lazy-load) a specific model by name.

        Models are cached after first load. This enables RRF queries to
        embed the query text with multiple models without reloading.

        Args:
            model_name: One of the keys in MODEL_DIMS.

        Returns:
            The loaded SentenceTransformer for the requested model.

        Raises:
            ValueError: If model_name is not recognized.
            RuntimeError: If model download fails.
        """
        if model_name not in MODEL_DIMS:
            raise ValueError(
                f"Unknown model '{model_name}'. Known models: {list(MODEL_DIMS.keys())}"
            )
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        from sentence_transformers import SentenceTransformer

        device = (
            "cuda"
            if (self._should_use_gpu() and self._check_cuda_available())
            else "cpu"
        )
        try:
            model = await asyncio.to_thread(
                SentenceTransformer,
                model_name,
                device=device,
                cache_folder=self._get_cache_folder(),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}' on {device}."
            ) from e
        self._model_cache[model_name] = model
        return model  # type: ignore[no-any-return]

    async def embed(
        self,
        text: str,
        model_name: str | None = None,
    ) -> tuple[list[float], str]:
        """Embed text with a specific model (multi-model dispatch).

        Args:
            text: Input text to embed.
            model_name: Model to use. If None, uses the active model
                (config.model_name).

        Returns:
            Tuple of (embedding_vector, model_name_used).
        """
        model_name = model_name or self._active_model_name
        model = await self.get_model_for(model_name)
        vector = await asyncio.to_thread(model.encode, [text])
        vec = vector[0]
        embedding_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        return embedding_list, model_name

    def get_loaded_models(self) -> list[str]:
        """Get list of currently loaded model names (for memory reporting).

        Returns:
            List of model names currently in the cache.
        """
        return list(self._model_cache.keys())
