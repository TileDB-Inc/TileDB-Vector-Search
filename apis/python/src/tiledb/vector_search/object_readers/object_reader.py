from typing import Any, Mapping, Optional, List


class ObjectPartition:
    def size(self):
        raise NotImplementedError

    def id(self):
        raise NotImplemented

    def index_slice(self):
        raise NotImplemented


class ObjectReader:
    def __init__(
        self,
        uri: str,
        config: Optional[Mapping[str, Any]] = None,
        timestamp=None,
    ):
        self.uri = uri
        self.config = config
        self.timestamp = timestamp

    def size(self):
        raise NotImplementedError

    def metadata_schema(self):
        raise NotImplementedError

    def metadata_array_uri(self):
        raise NotImplementedError
    
    def metadata_array_object_id_dim(self):
        raise NotImplementedError

    def get_partitions(self, partition_size: int = -1) -> List[ObjectPartition]:
        raise NotImplementedError

    def read_objects(self, partition: ObjectPartition):
        raise NotImplementedError

    def read_objects_by_ids(self, ids):
        raise NotImplementedError
