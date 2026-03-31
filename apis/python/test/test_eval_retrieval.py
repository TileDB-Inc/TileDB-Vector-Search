import numpy as np

from tiledb.vector_search.evals.retrieval import _n_results_from_distances
from tiledb.vector_search.evals.retrieval import path_matches_retrieved
from tiledb.vector_search.evals.retrieval import recall_at_k


def test_path_matches_retrieved() -> None:
    assert path_matches_retrieved("/a/b/foo.md", "/a/b/foo.md")
    # Same filename, different parent dirs — not a hit (avoids index.qmd false positives).
    assert not path_matches_retrieved("C:\\a\\foo.md", "/x/foo.md")
    assert path_matches_retrieved("/docs/cli.md", "cli.md")
    assert path_matches_retrieved("/prefix/docs/cli.md", "docs/cli.md")
    assert not path_matches_retrieved("/other/bar.md", "foo.md")
    assert path_matches_retrieved(
        "/repo/academy/modalities/vector/algorithms/index.qmd",
        "modalities/vector/algorithms/index.qmd",
    )
    assert not path_matches_retrieved(
        "/repo/academy/modalities/vector/algorithms/index.qmd",
        "modalities/vector/distance-metrics/index.qmd",
    )


def test_recall_at_k() -> None:
    assert recall_at_k(["a.md"], ["x/a.md"]) == 1.0
    assert recall_at_k(["a.md"], ["b.md"]) == 0.0
    assert np.isnan(recall_at_k([], ["a.md"]))


def test_n_results_from_distances() -> None:
    # Trailing zero distance after position 0 is treated as padding (CLI search behavior).
    d = np.array([[0.5, 0.6, 0.7, 0.0]], dtype=np.float32)
    assert _n_results_from_distances(d) == 3
    d2 = np.array([[0.5, 0.6, 0.0]], dtype=np.float32)
    assert _n_results_from_distances(d2) == 2
