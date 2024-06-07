# Re-import mode from cloud.dag
from tiledb.cloud.dag.mode import Mode

from . import utils
from .flat_index import FlatIndex
from .index import Index
from .ingestion import ingest
from .ivf_flat_index import IVFFlatIndex
from .ivf_pq_index import IVFPQIndex
from .module import array_to_matrix
from .module import ivf_index
from .module import ivf_index_tdb
from .module import ivf_query
from .module import ivf_query_ram
from .module import load_as_array
from .module import load_as_matrix
from .module import partition_ivf_index
from .module import query_vq_heap
from .module import query_vq_nth
from .module import validate_top_k
from .storage_formats import STORAGE_VERSION
from .storage_formats import storage_formats
from .vamana_index import VamanaIndex

try:
    from tiledb.vector_search.version import version as __version__
except ImportError:
    __version__ = "0.0.0.local"

__all__ = [
    "Index",
    "FlatIndex",
    "IVFFlatIndex",
    "VamanaIndex",
    "IVFPQIndex",
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
    "STORAGE_VERSION",
    "storage_formats",
]
