"""
FlatIndex implementation.

Stores all vectors in a 2D TileDB array performing exhaustive similarity
search between the query vectors and all the dataset vectors.
"""
from threading import Thread
from typing import Any, Mapping, Sequence

import numpy as np

from tiledb.vector_search import index
from tiledb.vector_search.module import *
from tiledb.vector_search.storage_formats import STORAGE_VERSION
from tiledb.vector_search.storage_formats import storage_formats
from tiledb.vector_search.storage_formats import validate_storage_version
from tiledb.vector_search.utils import MAX_FLOAT32
from tiledb.vector_search.utils import MAX_INT32
from tiledb.vector_search.utils import MAX_UINT64
from tiledb.vector_search.utils import create_array_and_add_to_group

TILE_SIZE_BYTES = 128000000  # 128MB
INDEX_TYPE = "FLAT"


class FlatIndex(index.Index):
    """
    Opens a `FlatIndex` loading all dataset vectors in main memory.

    Parameters
    ----------
    uri: str
        URI of the index.
    config: Optional[Mapping[str, Any]]
        TileDB config dictionary.
    timestamp: int or tuple(int)
        If int, open the index at a given timestamp.
        If tuple, open at the given start and end timestamps.
    open_for_remote_query_execution: bool
        If `True`, do not load any index data in main memory locally, and instead load index data in the TileDB Cloud taskgraph created when a non-`None` `driver_mode` is passed to `query()`.
        If `False`, load index data in main memory locally. Note that you can still use a taskgraph for query execution, you'll just end up loading the data both on your local machine and in the cloud taskgraph.
    """

    def __init__(
        self,
        uri: str,
        config: Optional[Mapping[str, Any]] = None,
        timestamp=None,
        open_for_remote_query_execution: bool = False,
        group: tiledb.Group = None,
        **kwargs,
    ):
        self.index_open_kwargs = {
            "uri": uri,
            "config": config,
            "timestamp": timestamp,
        }
        self.index_open_kwargs.update(kwargs)
        self.index_type = INDEX_TYPE
        super().__init__(
            uri=uri,
            config=config,
            timestamp=timestamp,
            open_for_remote_query_execution=open_for_remote_query_execution,
            group=group,
        )
        self._index = None
        self.db_uri = self.group[
            storage_formats[self.storage_version]["PARTS_ARRAY_NAME"]
            + self.index_version
        ].uri
        schema = tiledb.ArraySchema.load(self.db_uri, ctx=tiledb.Ctx(self.config))
        self.dimensions = schema.shape[0]
        if self.base_size == -1:
            self.size = schema.domain.dim(1).domain[1] + 1
        else:
            self.size = self.base_size

        self.dtype = np.dtype(self.group.meta.get("dtype", None))
        if (
            storage_formats[self.storage_version]["IDS_ARRAY_NAME"] + self.index_version
            in self.group
        ):
            self.ids_uri = self.group[
                storage_formats[self.storage_version]["IDS_ARRAY_NAME"]
                + self.index_version
            ].uri
        else:
            self.ids_uri = ""
        if self.size > 0 and not open_for_remote_query_execution:
            self._db = load_as_matrix(
                self.db_uri,
                ctx=self.ctx,
                config=config,
                size=self.size,
                timestamp=self.base_array_timestamp,
            )
            if self.dtype is None:
                self.dtype = self._db.dtype
            # Check for existence of ids array. Previous versions were not using external_ids in the ingestion assuming
            # that the external_ids were the position of the vector in the array.
            if self.ids_uri == "":
                self._ids = StdVector_u64(np.arange(self.size).astype(np.uint64))
            else:
                self._ids = read_vector_u64(
                    self.ctx, self.ids_uri, 0, self.size, self.base_array_timestamp
                )

    def get_dimensions(self):
        """
        Returns the dimension of the vectors in the index.
        """
        return self.dimensions

    def query_internal(
        self,
        queries: np.ndarray,
        k: int = 10,
        nthreads: int = 8,
        **kwargs,
    ):
        """
        Queries a FlatIndex using the vectors already loaded in main memory.

        Parameters
        ----------
        queries: np.ndarray
            2D array of query vectors. This can be used as a batch query interface by passing multiple queries in one call.
        k: int
            Number of results to return per query vector.
        nthreads: int
            Number of threads to use for query execution.
        """
        # TODO:
        # - typecheck queries
        # - add all the options and query strategies
        if self.size == 0:
            return np.full((queries.shape[0], k), MAX_FLOAT32), np.full(
                (queries.shape[0], k), MAX_UINT64
            )

        queries_m = array_to_matrix(np.transpose(queries))
        d, i = query_vq_heap(
            self._db, queries_m, self._ids, k, nthreads, self.distance_metric
        )

        return np.transpose(np.array(d)), np.transpose(np.array(i))

    def vacuum(self):
        """
        The vacuuming process permanently deletes index files that are consolidated through the consolidation
        process. TileDB separates consolidation from vacuuming, in order to make consolidation process-safe
        in the presence of concurrent reads and writes.

        Note:

        1. Vacuuming is not process-safe and you should take extra care when invoking it.
        2. Vacuuming may affect the granularity of the time traveling functionality.

        The Flat class vacuums consolidated fragment, array metadata and commits for the `db`
        and `ids` arrays.
        """
        super().vacuum()
        if not self.uri.startswith("tiledb://"):
            modes = ["fragment_meta", "commits", "array_meta"]
            for mode in modes:
                conf = tiledb.Config(self.config)
                conf["sm.consolidation.mode"] = mode
                conf["sm.vacuum.mode"] = mode
                tiledb.vacuum(self.db_uri, config=conf)
                tiledb.vacuum(self.ids_uri, config=conf)


def create(
    uri: str,
    dimensions: int,
    vector_type: np.dtype,
    group_exists: bool = False,
    group: tiledb.Group = None,
    config: Optional[Mapping[str, Any]] = None,
    storage_version: str = STORAGE_VERSION,
    distance_metric: vspy.DistanceMetric = vspy.DistanceMetric.SUM_OF_SQUARES,
    asset_creation_threads: Sequence[Thread] = None,
    **kwargs,
) -> FlatIndex:
    """
    Creates an empty FlatIndex.

    Parameters
    ----------
    uri: str
        URI of the index.
    dimensions: int
        Number of dimensions for the vectors to be stored in the index.
    vector_type: np.dtype
        Datatype of vectors.
        Supported values (uint8, int8, float32).
    group_exists: bool
        If False it creates the TileDB group for the index.
        If True the method expects the TileDB group to be already created.
    config: Optional[Mapping[str, Any]]
        TileDB config dictionary.
    storage_version: str
        The TileDB vector search storage version to use.
        If not provided, use hte latest stable storage version.
    distance_metric: vspy.DistanceMetric
        Distance metric to use for the index.
        If not provided, use L2 distance.
    group: tiledb.Group
        TileDB group open in write mode.
        Internal, this is used to avoid opening the group multiple times during
        ingestion.
    asset_creation_threads: Sequence[Thread]
        List of asset creation threads to append new threads.
        Internal, this is used to parallelize all asset creation during
        ingestion.
    """
    validate_storage_version(storage_version)

    with tiledb.scope_ctx(ctx_or_config=config):
        if not group_exists:
            try:
                tiledb.group_create(uri)
            except tiledb.TileDBError as err:
                raise err
        if group is None:
            grp = tiledb.Group(uri, "w")
        else:
            grp = group

        if asset_creation_threads is not None:
            threads = asset_creation_threads
        else:
            threads = []

        index.create_metadata(
            group=grp,
            dimensions=dimensions,
            vector_type=vector_type,
            index_type=INDEX_TYPE,
            storage_version=storage_version,
            distance_metric=distance_metric,
        )

        tile_size = TILE_SIZE_BYTES / np.dtype(vector_type).itemsize / dimensions
        ids_array_name = storage_formats[storage_version]["IDS_ARRAY_NAME"]
        parts_array_name = storage_formats[storage_version]["PARTS_ARRAY_NAME"]
        updates_array_name = storage_formats[storage_version]["UPDATES_ARRAY_NAME"]
        ids_uri = f"{uri}/{ids_array_name}"
        parts_uri = f"{uri}/{parts_array_name}"
        updates_array_uri = f"{uri}/{updates_array_name}"

        ids_array_rows_dim = tiledb.Dim(
            name="rows",
            domain=(0, MAX_INT32),
            tile=tile_size,
            dtype=np.dtype(np.int32),
        )
        ids_array_dom = tiledb.Domain(ids_array_rows_dim)
        ids_attr = tiledb.Attr(
            name="values",
            dtype=np.dtype(np.uint64),
            filters=storage_formats[storage_version]["DEFAULT_ATTR_FILTERS"],
        )
        ids_schema = tiledb.ArraySchema(
            domain=ids_array_dom,
            sparse=False,
            attrs=[ids_attr],
            cell_order="col-major",
            tile_order="col-major",
        )
        thread = Thread(
            target=create_array_and_add_to_group,
            kwargs={
                "array_uri": ids_uri,
                "array_name": ids_array_name,
                "group": grp,
                "schema": ids_schema,
            },
        )
        thread.start()
        threads.append(thread)

        parts_array_rows_dim = tiledb.Dim(
            name="rows",
            domain=(0, dimensions - 1),
            tile=dimensions,
            dtype=np.dtype(np.int32),
        )
        parts_array_cols_dim = tiledb.Dim(
            name="cols",
            domain=(0, MAX_INT32),
            tile=tile_size,
            dtype=np.dtype(np.int32),
        )
        parts_array_dom = tiledb.Domain(parts_array_rows_dim, parts_array_cols_dim)
        parts_attr = tiledb.Attr(
            name="values",
            dtype=vector_type,
            filters=storage_formats[storage_version]["DEFAULT_ATTR_FILTERS"],
        )
        parts_schema = tiledb.ArraySchema(
            domain=parts_array_dom,
            sparse=False,
            attrs=[parts_attr],
            cell_order="col-major",
            tile_order="col-major",
        )
        thread = Thread(
            target=create_array_and_add_to_group,
            kwargs={
                "array_uri": parts_uri,
                "array_name": parts_array_name,
                "group": grp,
                "schema": parts_schema,
            },
        )
        thread.start()
        threads.append(thread)

        external_id_dim = tiledb.Dim(
            name="external_id",
            domain=(0, MAX_UINT64 - 1),
            dtype=np.dtype(np.uint64),
        )
        dom = tiledb.Domain(external_id_dim)
        vector_attr = tiledb.Attr(name="vector", dtype=vector_type, var=True)
        updates_schema = tiledb.ArraySchema(
            domain=dom,
            sparse=True,
            attrs=[vector_attr],
            allows_duplicates=False,
        )
        thread = Thread(
            target=create_array_and_add_to_group,
            kwargs={
                "array_uri": updates_array_uri,
                "array_name": updates_array_name,
                "group": grp,
                "schema": updates_schema,
            },
        )
        thread.start()
        threads.append(thread)

        if asset_creation_threads is None:
            for thread in threads:
                thread.join()
        if group is None:
            grp.close()
            return FlatIndex(uri=uri, config=config)
        else:
            return None
