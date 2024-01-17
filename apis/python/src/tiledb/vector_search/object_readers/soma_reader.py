from typing import Any, Mapping, Optional, List
from tiledb.vector_search.object_readers import ObjectPartition, ObjectReader


class SomaPartition(ObjectPartition):
    def __init__(
        self,
        partition_id: str,
        coord_start: int,
        coord_end: int,
    ):
        self.partition_id = partition_id
        self.coord_start = coord_start
        self.coord_end = coord_end
        self.size = coord_end - coord_start

    def size(self):
        return self.size

    def id(self):
        return self.partition_id

    def index_slice(self):
        return [self.coord_start, self.coord_end]


class SomaReader(ObjectReader):
    def __init__(
        self,
        uri: str,
        config: Optional[Mapping[str, Any]] = None,
        timestamp=None,
    ):
        import tiledb
        import tiledbsoma

        super().__init__(uri=uri, config=config, timestamp=timestamp)
        context = tiledbsoma.SOMATileDBContext(tiledb_ctx=tiledb.Ctx(self.config))
        exp = tiledbsoma.Experiment.open(self.uri, "r", context=context)
        self.num_obs = exp.obs.count
        self.obs_array_uri = exp.obs.uri

    def size(self):
        return self.num_obs

    def metadata_schema(self):
        import tiledbsoma

        context = tiledbsoma.SOMATileDBContext(tiledb_ctx=tiledb.Ctx(self.config))
        exp = tiledbsoma.Experiment.open(self.uri, "r", context=context)
        with tiledb.open(exp.obs.uri, "r") as obs_array:
            return obs_array.schema

    def metadata_array_uri(self):
        return self.obs_array_uri
    
    def metadata_array_object_id_dim(self):
        return "soma_joinid"

    def get_partitions(self, partition_size: int = -1) -> List[ObjectPartition]:
        if partition_size == -1:
            partition_size = 100000

        partitions = []
        partition_id = 0
        for coord_start in range(0, self.num_obs, partition_size):
            coord_end = coord_start + partition_size
            if coord_end > self.num_obs:
                coord_end = self.num_obs
            partitions.append(SomaPartition(str(partition_id), coord_start, coord_end))
            partition_id += 1

        return partitions

    def read_objects(self, partition: SomaPartition):
        import tiledb
        import tiledbsoma
        import numpy as np

        context = tiledbsoma.SOMATileDBContext(tiledb_ctx=tiledb.Ctx(self.config))
        exp = tiledbsoma.Experiment.open(self.uri, "r", context=context)
        query = exp.axis_query(
                measurement_name="RNA",
                obs_query=tiledbsoma.AxisQuery(
                    coords=(slice(partition.coord_start, partition.coord_end-1),)
                ),
            )
        
        with tiledb.open(exp.obs.uri, "r", timestamp=self.timestamp, config=self.config) as obs_array:
            metadata = obs_array[partition.coord_start: partition.coord_end-1]
        return {
                "data": query.to_anndata(X_name="data").X.toarray(),
                "soma_joinid": np.arange(partition.coord_start, partition.coord_end)
            }, metadata

    def read_objects_by_ids(self, ids):
        import tiledb
        import tiledbsoma
        context = tiledbsoma.SOMATileDBContext(tiledb_ctx=tiledb.Ctx(self.config))
        exp = tiledbsoma.Experiment.open(self.uri, "r", context=context)
        query = exp.axis_query(
                measurement_name="RNA",
                obs_query=tiledbsoma.AxisQuery(
                    coords=(ids,)
                ),
            )
        return {"data": query.to_anndata(X_name="data").X.toarray()}
