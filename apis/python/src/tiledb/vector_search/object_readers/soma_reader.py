from typing import Dict, List, OrderedDict, Tuple

from tiledb import Attr
from tiledb.vector_search.object_readers import ObjectPartition
from tiledb.vector_search.object_readers import ObjectReader


class SomaAnnDataPartition(ObjectPartition):
    def __init__(
        self,
        partition_id: int,
        coord_start: int,
        coord_end: int,
        **kwargs,
    ):
        self.partition_id = partition_id
        self.coord_start = coord_start
        self.coord_end = coord_end
        self.size = coord_end - coord_start

    def init_kwargs(self) -> Dict:
        return {
            "partition_id": self.partition_id,
            "coord_start": self.coord_start,
            "coord_end": self.coord_end,
        }

    def id(self) -> int:
        return self.partition_id


class SomaAnnDataReader(ObjectReader):
    def __init__(
        self,
        uri: str,
        measurement_name: str = "RNA",
        X_name: str = "raw",
        external_id_col: str = "soma_joinid",
        obs_value_filter: str = None,
        var_value_filter: str = None,
        max_size: int = -1,
        cells_per_partition: int = 10000,
        timestamp=None,
        **kwargs,
    ):
        self.uri = uri
        self.measurement_name = measurement_name
        self.X_name = X_name
        self.external_id_col = external_id_col
        self.obs_value_filter = obs_value_filter
        self.var_value_filter = var_value_filter
        self.cells_per_partition = cells_per_partition
        self.max_size = max_size
        self.timestamp = timestamp
        self.obs_array_uri = None
        self.context = None
        self.exp = None

    def init_kwargs(self) -> Dict:
        return {
            "uri": self.uri,
            "measurement_name": self.measurement_name,
            "X_name": self.X_name,
            "external_id_col": self.external_id_col,
            "obs_value_filter": self.obs_value_filter,
            "var_value_filter": self.var_value_filter,
            "cells_per_partition": self.cells_per_partition,
            "max_size": self.max_size,
            "timestamp": self.timestamp,
        }

    def init_soma_exp(self):
        if self.exp is None:
            import tiledbsoma

            import tiledb

            self.context = tiledbsoma.SOMATileDBContext(
                tiledb_config=tiledb.default_ctx().config().dict()
            )
            self.exp = tiledbsoma.Experiment.open(self.uri, "r", context=self.context)

    def partition_class_name(self) -> str:
        return "SomaAnnDataPartition"

    def metadata_array_uri(self) -> str:
        self.init_soma_exp()
        if self.obs_array_uri is None:
            self.obs_array_uri = self.exp.obs.uri
        return self.obs_array_uri

    def metadata_attributes(self) -> List[Attr]:
        import tiledb

        with tiledb.open(self.metadata_array_uri(), "r") as obs_array:
            attributes = []
            for i in range(obs_array.schema.nattr):
                attributes.append(obs_array.schema.attr(i))
            return attributes

    def get_partitions(
        self, cells_per_partition: int = -1
    ) -> List[SomaAnnDataPartition]:
        self.init_soma_exp()
        if self.max_size == -1:
            self.num_obs = self.exp.obs.count
        else:
            self.num_obs = min(self.max_size, self.exp.obs.count)
        if cells_per_partition == -1:
            cells_per_partition = self.cells_per_partition

        partitions = []
        partition_id = 0
        for coord_start in range(0, self.num_obs, cells_per_partition):
            coord_end = coord_start + cells_per_partition
            if coord_end > self.num_obs:
                coord_end = self.num_obs
            partitions.append(
                SomaAnnDataPartition(partition_id, coord_start, coord_end)
            )
            partition_id += 1

        return partitions

    def read_objects(
        self, partition: SomaAnnDataPartition
    ) -> Tuple[OrderedDict, OrderedDict]:
        import tiledbsoma

        self.init_soma_exp()
        query = self.exp.axis_query(
            measurement_name=self.measurement_name,
            obs_query=tiledbsoma.AxisQuery(
                value_filter=self.obs_value_filter,
                coords=(slice(partition.coord_start, partition.coord_end - 1),),
            ),
            var_query=tiledbsoma.AxisQuery(value_filter=self.var_value_filter),
        )
        adata = query.to_anndata(X_name=self.X_name)
        return {
            "anndata": adata,
            "external_id": adata.obs[self.external_id_col].to_numpy(),
        }, None

    def read_objects_by_external_ids(self, ids: List[int]) -> OrderedDict:
        import tiledbsoma

        self.init_soma_exp()
        query = self.exp.axis_query(
            measurement_name=self.measurement_name,
            obs_query=tiledbsoma.AxisQuery(
                value_filter=self.obs_value_filter, coords=(ids,)
            ),
            var_query=tiledbsoma.AxisQuery(value_filter=self.var_value_filter),
        )
        adata = query.to_anndata(X_name=self.X_name)
        return {"anndata": adata, "external_id": ids}
