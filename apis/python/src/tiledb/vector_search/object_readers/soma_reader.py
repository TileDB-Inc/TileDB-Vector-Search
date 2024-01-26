from typing import Any, Dict, List, Mapping, Optional, OrderedDict, Tuple

# from tiledb.vector_search.object_readers import ObjectPartition, ObjectReader
from tiledb import Attr


# class SomaRNAXRowPartition(ObjectPartition):
class SomaRNAXRowPartition:
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


# class SomaRNAXRowReader(ObjectReader):
class SomaRNAXRowReader:
    def __init__(
        self,
        uri: str,
        cells_per_partition: int = 10000,
        config: Optional[Mapping[str, Any]] = None,
        timestamp=None,
        **kwargs,
    ):
        import tiledbsoma

        import tiledb

        self.uri = uri
        self.cells_per_partition = cells_per_partition
        self.config = config
        self.timestamp = timestamp
        context = tiledbsoma.SOMATileDBContext(tiledb_ctx=tiledb.Ctx(self.config))
        exp = tiledbsoma.Experiment.open(self.uri, "r", context=context)
        self.num_obs = exp.obs.count
        self.obs_array_uri = exp.obs.uri

    def init_kwargs(self) -> Dict:
        return {
            "uri": self.uri,
            "cells_per_partition": self.cells_per_partition,
            "config": self.config,
            "timestamp": self.timestamp,
        }

    def partition_class_name(self) -> str:
        return "SomaRNAXRowPartition"

    def metadata_array_uri(self) -> str:
        return self.obs_array_uri

    def metadata_attributes(self) -> List[Attr]:
        import tiledbsoma

        import tiledb

        context = tiledbsoma.SOMATileDBContext(tiledb_ctx=tiledb.Ctx(self.config))
        exp = tiledbsoma.Experiment.open(self.uri, "r", context=context)
        with tiledb.open(exp.obs.uri, "r") as obs_array:
            attributes = []
            for i in range(obs_array.schema.nattr):
                attributes.append(obs_array.schema.attr(i))
            return attributes

    def get_partitions(
        self, cells_per_partition: int = -1
    ) -> List[SomaRNAXRowPartition]:
        if cells_per_partition == -1:
            cells_per_partition = self.cells_per_partition

        partitions = []
        partition_id = 0
        for coord_start in range(0, self.num_obs, cells_per_partition):
            coord_end = coord_start + cells_per_partition
            if coord_end > self.num_obs:
                coord_end = self.num_obs
            partitions.append(
                SomaRNAXRowPartition(partition_id, coord_start, coord_end)
            )
            partition_id += 1

        return partitions

    def read_objects(
        self, partition: SomaRNAXRowPartition
    ) -> Tuple[OrderedDict, OrderedDict]:
        import numpy as np
        import tiledbsoma

        import tiledb

        context = tiledbsoma.SOMATileDBContext(tiledb_ctx=tiledb.Ctx(self.config))
        exp = tiledbsoma.Experiment.open(self.uri, "r", context=context)
        query = exp.axis_query(
            measurement_name="RNA",
            obs_query=tiledbsoma.AxisQuery(
                coords=(slice(partition.coord_start, partition.coord_end - 1),)
            ),
        )

        with tiledb.open(
            exp.obs.uri, "r", timestamp=self.timestamp, config=self.config
        ) as obs_array:
            metadata = obs_array[partition.coord_start : partition.coord_end - 1]
        return {
            "data": query.to_anndata(X_name="data").X.toarray(),
            "external_id": np.arange(partition.coord_start, partition.coord_end),
        }, metadata

    def read_objects_by_external_ids(self, ids: List[int]) -> OrderedDict:
        import tiledbsoma

        import tiledb

        context = tiledbsoma.SOMATileDBContext(tiledb_ctx=tiledb.Ctx(self.config))
        exp = tiledbsoma.Experiment.open(self.uri, "r", context=context)
        query = exp.axis_query(
            measurement_name="RNA",
            obs_query=tiledbsoma.AxisQuery(coords=(ids,)),
        )
        return {"data": query.to_anndata(X_name="data").X.toarray(), "external_id": ids}
