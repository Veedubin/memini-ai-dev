"""Tests for _to_float_list vector normalization helper (BUG-4 fix).

pgvector 0.5.0 changed the asyncpg codec to return a ``Vector`` object
that is not iterable. The ``_to_float_list`` helper handles all
representations: pgvector Vector, str, numpy.ndarray, list, tuple.
"""

from __future__ import annotations

import array
import json

import pytest

from memini_ai.postgres.database import _to_float_list

# ── Fake Vector-like objects ──────────────────────────────────────────────


class FakePgvectorVector:
    """Simulates pgvector 0.5.0's Vector object (not iterable, has to_list)."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def to_list(self) -> list[float]:
        return self._values

    def __iter__(self) -> None:
        raise TypeError("'FakePgvectorVector' object is not iterable")


class FakePgvectorVectorNoToList:
    """Simulates an object that has neither to_list nor __iter__."""

    pass


# ── Tests ─────────────────────────────────────────────────────────────────


class TestToFloatList:
    """Test _to_float_list with all input types."""

    def test_pgvector_vector_uses_to_list(self) -> None:
        """pgvector 0.5.0 Vector: not iterable but has to_list()."""
        v = FakePgvectorVector([1.0, 2.0, 3.0])
        result = _to_float_list(v)
        assert result == [1.0, 2.0, 3.0]
        assert all(isinstance(x, float) for x in result)

    def test_pgvector_vector_not_iterable_raises(self) -> None:
        """Verify the fake Vector really isn't iterable (confirms the bug)."""
        v = FakePgvectorVector([1.0, 2.0])
        with pytest.raises(TypeError, match="not iterable"):
            list(v)  # type: ignore[arg-type]

    def test_string_format_parsed(self) -> None:
        """pgvector text format '[1,2,3]' → parsed to list."""
        result = _to_float_list("[1.0, 2.0, 3.0]")
        assert result == [1.0, 2.0, 3.0]

    def test_string_format_negative_floats(self) -> None:
        """String with negative floats."""
        result = _to_float_list("[-0.1, 0.5, -2.3]")
        assert result == [-0.1, 0.5, -2.3]

    def test_list_input(self) -> None:
        """Plain list input."""
        result = _to_float_list([1.0, 2.0, 3.0])
        assert result == [1.0, 2.0, 3.0]

    def test_tuple_input(self) -> None:
        """Tuple input."""
        result = _to_float_list((1.0, 2.0, 3.0))
        assert result == [1.0, 2.0, 3.0]

    def test_array_array_input(self) -> None:
        """array.array input."""
        arr = array.array("d", [1.0, 2.0, 3.0])
        result = _to_float_list(arr)
        assert result == [1.0, 2.0, 3.0]

    def test_numpy_ndarray_input(self) -> None:
        """numpy.ndarray input (uses .tolist())."""
        np = pytest.importorskip("numpy")
        arr = np.array([1.0, 2.0, 3.0])
        result = _to_float_list(arr)
        assert result == [1.0, 2.0, 3.0]

    def test_int_list_converted_to_float(self) -> None:
        """Integer values are converted to float."""
        result = _to_float_list([1, 2, 3])
        assert result == [1.0, 2.0, 3.0]
        assert all(isinstance(x, float) for x in result)

    def test_empty_list(self) -> None:
        """Empty list input."""
        assert _to_float_list([]) == []

    def test_single_element(self) -> None:
        """Single element vector."""
        assert _to_float_list([0.5]) == [0.5]

    def test_large_vector(self) -> None:
        """384-dim vector (typical MiniLM embedding size)."""
        values = [float(i) / 384.0 for i in range(384)]
        result = _to_float_list(values)
        assert len(result) == 384
        assert result[0] == 0.0
        assert result[-1] == pytest.approx(383.0 / 384.0)

    def test_1024_dim_vector(self) -> None:
        """1024-dim vector (BGE-M3 embedding size)."""
        values = [float(i) / 1024.0 for i in range(1024)]
        result = _to_float_list(values)
        assert len(result) == 1024

    def test_invalid_string_raises_json_error(self) -> None:
        """Invalid string that's not valid JSON raises an error."""
        with pytest.raises(json.JSONDecodeError):
            _to_float_list("not a vector string")

    def test_object_without_to_list_or_iter_raises(self) -> None:
        """Object with neither to_list nor __iter__ raises TypeError."""
        with pytest.raises(TypeError):
            _to_float_list(FakePgvectorVectorNoToList())  # type: ignore[arg-type]


class TestToFloatListWithRealPgvector:
    """Tests with real pgvector Vector if available."""

    def test_real_pgvector_vector(self) -> None:
        """If pgvector is installed, test with the real Vector class."""
        try:
            from pgvector import Vector
        except ImportError:
            pytest.skip("pgvector not installed")

        v = Vector([1.0, 2.0, 3.0])
        # Verify Vector is not iterable (the bug)
        assert not hasattr(v, "__iter__")
        # _to_float_list should handle it via to_list()
        result = _to_float_list(v)
        assert result == [1.0, 2.0, 3.0]
        assert all(isinstance(x, float) for x in result)
