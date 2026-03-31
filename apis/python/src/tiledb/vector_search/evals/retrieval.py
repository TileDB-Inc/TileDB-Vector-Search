from __future__ import annotations

from pathlib import PurePosixPath
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tiledb.vector_search.object_api.object_index import ObjectIndex


def _normalize_path(p: str) -> str:
    return str(PurePosixPath(p.replace("\\", "/")))


def path_matches_retrieved(retrieved: str, relevant: str) -> bool:
    """
    Whether a retrieved file_path is considered a hit for a labeled relevant path.

    Matches if normalized full paths are equal, or one path is a suffix of the other
    (so relative labels like ``modalities/.../algorithms/index.qmd`` match stored
    absolute paths), or the last two components (parent directory + filename) match.

    Basename-only matching is intentionally not used: many doc trees use the same
    filename (e.g. ``index.qmd``) in every folder, which would make recall@1 trivially
    high when any ``index.qmd`` chunk appears in top-k.
    """
    r = _normalize_path(retrieved.strip())
    g = _normalize_path(relevant.strip())
    if r == g:
        return True
    if r.endswith(g) or g.endswith(r):
        return True
    pr = PurePosixPath(r)
    pg = PurePosixPath(g)
    if pr.name == pg.name and pr.parent.name == pg.parent.name and pr.name:
        return True
    return False


def recall_at_k(
    relevant_file_paths: Sequence[str],
    retrieved_file_paths: Sequence[str],
) -> float:
    """
    1.0 if any retrieved path matches any relevant path (see path_matches_retrieved), else 0.0.
    """
    if not relevant_file_paths:
        return float("nan")
    for rel in relevant_file_paths:
        for ret in retrieved_file_paths:
            if ret and path_matches_retrieved(str(ret), str(rel)):
                return 1.0
    return 0.0


def _n_results_from_distances(distances: np.ndarray) -> int:
    """Match CLI search: stop at padding rows (distance 0 after first)."""
    n = 0
    row = distances[0]
    for i in range(row.shape[0]):
        dist = float(row[i])
        if dist == 0.0 and i > 0:
            break
        n += 1
    return n


def query_topk_file_paths(
    obj_index: ObjectIndex,
    query: str,
    top_k: int,
) -> List[str]:
    """Run a text query and return file_path values for valid top results."""
    from collections import OrderedDict

    query_objects = OrderedDict({"text": np.array([query])})
    distances, _ids, metadata = obj_index.query(
        query_objects=query_objects,
        k=top_k,
        driver_mode=None,
        return_objects=False,
        return_metadata=True,
    )
    if not metadata or "file_path" not in metadata:
        return []
    n = _n_results_from_distances(distances)
    out: List[str] = []
    for i in range(min(n, distances.shape[1])):
        fp = metadata["file_path"][0, i]
        out.append(str(fp) if fp is not None else "")
    return out


def format_retrieval_context(
    obj_index: ObjectIndex,
    query: str,
    top_k: int,
) -> str:
    """Build a single string of retrieved chunks for the LLM (same shape as CLI search)."""
    from collections import OrderedDict

    query_objects = OrderedDict({"text": np.array([query])})
    distances, _ids, metadata = obj_index.query(
        query_objects=query_objects,
        k=top_k,
        driver_mode=None,
        return_objects=False,
        return_metadata=True,
    )
    if not metadata:
        return ""
    n = _n_results_from_distances(distances)
    parts: list[str] = []
    for i in range(min(n, distances.shape[1])):
        dist = float(distances[0, i])
        file_path = ""
        text = ""
        if "file_path" in metadata:
            file_path = str(metadata["file_path"][0, i])
        if "text" in metadata:
            text = str(metadata["text"][0, i])
        snippet = " ".join(text.split())
        parts.append(f"[{i + 1}] score={dist:.4f} {file_path}\n{snippet}")
    return "\n\n---\n\n".join(parts)


def make_index_retriever(obj_index: ObjectIndex, top_k: int) -> Callable[[str], str]:
    """Bind an open ObjectIndex into a retriever callable for the agent graph."""

    def retrieve(query: str) -> str:
        return format_retrieval_context(obj_index, query, top_k)

    return retrieve
