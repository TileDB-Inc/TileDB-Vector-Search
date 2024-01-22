# Re-import mode from cloud.dag
from tiledb.cloud.dag.mode import Mode

from . import utils
from .flat_index import FlatIndex
from .index import Index
from .ingestion import ingest
from .ivf_flat_index import IVFFlatIndex
from .module import (array_to_matrix, ivf_index, ivf_index_tdb, ivf_query,
                     ivf_query_ram, load_as_array, load_as_matrix,
                     partition_ivf_index, query_vq_heap, query_vq_nth,
                     validate_top_k)
from .storage_formats import STORAGE_VERSION, storage_formats

try:
    from tiledb.vector_search.version import version as __version__
except ImportError:
    __version__ = "0.0.0.local"

__all__ = [
    "FlatIndex",
    "IVFFlatIndex",
    "Mode",
    "load_as_array",
    "load_as_matrix",
    "ingest",
    "query_vq_nth",
    "query_vq_heap",
    "ivf_query",
    "ivf_query_ram",
    "validate_top_k",
    "ivf_index",
    "ivf_index_tdb",
    "array_to_matrix",
    "partition_ivf_index",
    "utils",
]
