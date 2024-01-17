from typing import Any, Mapping, Optional, List
# from tiledb.vector_search.object_readers import ObjectPartition, ObjectReader


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

    def size(self):
        return self.size

    def id(self):
        return self.partition_id

    def index_slice(self):
        return [self.start, self.end]


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

        # super().__init__(uri=uri, config=config, timestamp=timestamp)
        self.uri = uri
        self.config = config
        self.timestamp = timestamp
        with tiledb.open(uri, mode='r', timestamp=timestamp, config=config) as object_array:
            nonempty_object_array_domain = object_array.nonempty_domain()[0]
            self.num_obs = nonempty_object_array_domain[1] + 1 - nonempty_object_array_domain[0]
        self.image_metadata_uri = image_metadata_uri

    def get_kwargs(self):
        return {
            "uri": self.uri,
            "image_metadata_uri": self.image_metadata_uri,
            "config": self.config,
            "timestamp": self.timestamp,
        }
    
    def size(self):
        return self.num_obs

    def metadata_schema(self):
        import tiledb

        with tiledb.open(self.image_metadata_uri, "r", config=self.config) as image_metadata_array:
            return image_metadata_array.schema

    def metadata_array_uri(self):
        return self.image_metadata_uri
    
    def metadata_array_object_id_dim(self):
        return "image_id"

    # def get_partitions(self, partition_size: int = -1) -> List[ObjectPartition]:
    def get_partitions(self, partition_size: int = -1) -> List:
        if partition_size == -1:
            partition_size = 500

        partitions = []
        partition_id = 0
        for start in range(0, self.num_obs, partition_size):
            end = start + partition_size
            if end > self.num_obs:
                end = self.num_obs
            partitions.append(ImagePartition(str(partition_id), start, end))
            partition_id += 1

        return partitions

    # def read_objects(self, partition: ObjectPartition):
    def read_objects(self, partition):
        import tiledb

        with tiledb.open(self.uri, "r", timestamp=self.timestamp, config=self.config) as object_array:
            data = object_array[partition.start:partition.end]
            return data, None

    def read_objects_by_ids(self, ids):
        import tiledb

        with tiledb.open(self.uri, "r", timestamp=self.timestamp, config=self.config) as object_array:
            return object_array.multi_index[ids]
