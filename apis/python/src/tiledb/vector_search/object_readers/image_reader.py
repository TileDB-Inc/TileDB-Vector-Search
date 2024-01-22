from typing import Any, Mapping, Optional, List, Dict, OrderedDict, Tuple
# from tiledb.vector_search.object_readers import ObjectPartition, ObjectReader
from tiledb import Attr

# class ImagePartition(ObjectPartition):
class ImagePartition():
    def __init__(
        self,
        partition_id: str,
        start: int,
        end: int,
        **kwargs,
    ):
        self.partition_id = partition_id
        self.start = start
        self.end = end
        self.size = end - start

    def init_kwargs(self) -> Dict:
        return {
            "partition_id": self.partition_id,
            "start": self.start,
            "end": self.end,
        }

    def num_vectors(self) -> int:
        return self.size

    def index_slice(self) -> Tuple[int,int]:
        return (self.start, self.end)


# class ImageReader(ObjectReader):
class ImageReader():
    def __init__(
        self,
        *args,
        uri: str,
        image_metadata_uri: str,
        config: Optional[Mapping[str, Any]] = None,
        timestamp=None,
        **kwargs,
    ):
        import tiledb

        self.uri = uri
        self.config = config
        self.timestamp = timestamp
        with tiledb.open(uri, mode='r', timestamp=timestamp, config=config) as object_array:
            nonempty_object_array_domain = object_array.nonempty_domain()[0]
            self.num_objects = nonempty_object_array_domain[1] + 1 - nonempty_object_array_domain[0]
        self.image_metadata_uri = image_metadata_uri

    def init_kwargs(self) -> Dict:
        return {
            "uri": self.uri,
            "image_metadata_uri": self.image_metadata_uri,
            "config": self.config,
            "timestamp": self.timestamp,
        }
    
    def num_vectors(self) -> int:
        return self.num_objects

    def partition_class_name(self) -> str:
        return "ImagePartition"

    def metadata_array_uri(self) -> str:
        return self.image_metadata_uri

    def metadata_attributes(self) -> List[Attr]:
        import tiledb

        with tiledb.open(self.image_metadata_uri, "r", config=self.config) as image_metadata_array:
            attributes = []
            for i in range(image_metadata_array.schema.nattr):
                attributes.append(image_metadata_array.schema.attr(i))
            return attributes

    def get_partitions(self, partition_size: int = -1) -> List[ImagePartition]:
        if partition_size == -1:
            partition_size = 500

        partitions = []
        partition_id = 0
        for start in range(0, self.num_objects, partition_size):
            end = start + partition_size
            if end > self.num_objects:
                end = self.num_objects
            partitions.append(ImagePartition(str(partition_id), start, end))
            partition_id += 1

        return partitions

    def read_objects(self, partition: ImagePartition) -> Tuple[OrderedDict, OrderedDict]:
        import tiledb

        with tiledb.open(self.uri, "r", timestamp=self.timestamp, config=self.config) as object_array:
            data = object_array[partition.start:partition.end]
            external_ids = data.pop("image_id", None)
            data["external_id"] = external_ids
            return (data, None)

    def read_objects_by_external_ids(self, ids: List[int]) -> OrderedDict:
        import tiledb

        with tiledb.open(self.uri, "r", timestamp=self.timestamp, config=self.config) as object_array:
            return object_array.multi_index[ids]
