from typing import Any, Dict, List, Mapping, Optional, OrderedDict, Tuple

# from tiledb.vector_search.object_readers import ObjectPartition, ObjectReader
import tiledb


# class TileDB1DArrayPartition(ObjectPartition):
class TileDB1DArrayPartition:
    def __init__(
        self,
        partition_id: int,
        start: int,
        end: int,
        **kwargs,
    ):
        self.partition_id = partition_id
        self.start = start
        self.end = end

    def init_kwargs(self) -> Dict:
        return {
            "partition_id": self.partition_id,
            "start": self.start,
            "end": self.end,
        }

    def id(self) -> int:
        return self.partition_id


# class TileDB1DArrayReader(ObjectReader):
class TileDB1DArrayReader:
    """
    Reader that reads objects and their metadata stored in TileDB arrays.

    The first dimension of the object array should represent the `external_ids` of the objects to be read.
    """

    def __init__(
        self,
        *args,
        uri: str,
        partition_tile_size: int = -1,
        metadata_uri: str = None,
        config: Optional[Mapping[str, Any]] = None,
        timestamp=None,
        **kwargs,
    ):
        self.uri = uri
        self.partition_tile_size = partition_tile_size
        self.config = config
        self.timestamp = timestamp
        with tiledb.open(
            uri, mode="r", timestamp=timestamp, config=config
        ) as object_array:
            self.external_id_dimension_name = object_array.schema.domain.dim(0).name
            nonempty_object_array_domain = object_array.nonempty_domain()[0]
            self.domain_length = (
                nonempty_object_array_domain[1] + 1 - nonempty_object_array_domain[0]
            )
            if self.partition_tile_size == -1:
                self.partition_tile_size = object_array.schema.domain.dim(0).tile
        self.metadata_uri = metadata_uri

    def init_kwargs(self) -> Dict:
        return {
            "uri": self.uri,
            "partition_tile_size": self.partition_tile_size,
            "metadata_uri": self.metadata_uri,
            "config": self.config,
            "timestamp": self.timestamp,
        }

    def partition_class_name(self) -> str:
        return "TileDB1DArrayPartition"

    def metadata_array_uri(self) -> str:
        return self.metadata_uri

    def metadata_attributes(self) -> List[tiledb.Attr]:
        import tiledb

        if self.metadata_uri is None:
            return None
        with tiledb.open(self.metadata_uri, "r", config=self.config) as metadata_array:
            attributes = []
            for i in range(metadata_array.schema.nattr):
                attributes.append(metadata_array.schema.attr(i))
            return attributes

    def get_partitions(
        self, partition_tile_size: int = -1
    ) -> List[TileDB1DArrayPartition]:
        if partition_tile_size == -1:
            partition_tile_size = self.partition_tile_size

        partitions = []
        partition_id = 0
        for start in range(0, self.domain_length, partition_tile_size):
            end = int(start + partition_tile_size)
            if end > self.domain_length:
                end = self.domain_length
            partitions.append(TileDB1DArrayPartition(partition_id, start, end))
            partition_id += 1

        return partitions

    def read_objects(
        self, partition: TileDB1DArrayPartition
    ) -> Tuple[OrderedDict, OrderedDict]:
        import tiledb

        with tiledb.open(
            self.uri, "r", timestamp=self.timestamp, config=self.config
        ) as object_array:
            data = object_array[partition.start : partition.end]
            if self.external_id_dimension_name != "external_id":
                external_ids = data.pop(self.external_id_dimension_name, None)
                data["external_id"] = external_ids
        metadata = None
        if self.metadata_uri is not None:
            with tiledb.open(
                self.metadata_uri, "r", timestamp=self.timestamp, config=self.config
            ) as metadata_array:
                metadata = metadata_array[partition.start : partition.end]
        return (data, metadata)

    def read_objects_by_external_ids(self, ids: List[int]) -> OrderedDict:
        import tiledb

        with tiledb.open(
            self.uri, "r", timestamp=self.timestamp, config=self.config
        ) as object_array:
            return object_array.multi_index[ids]
