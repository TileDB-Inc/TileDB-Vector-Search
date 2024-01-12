from abc import ABC, abstractmethod 
from typing import Any, Mapping, Optional, List


class ObjectPartition(ABC):

    @abstractmethod
    def size(self):
        raise NotImplementedError

    @abstractmethod
    def id(self):
        raise NotImplemented

    @abstractmethod
    def index_slice(self):
        raise NotImplemented


class ObjectReader(ABC):
    def __init__(
        self,
        uri: str,
        config: Optional[Mapping[str, Any]] = None,
        timestamp=None,
    ):
        self.uri = uri
        self.config = config
        self.timestamp = timestamp

    @abstractmethod
    def size(self):
        raise NotImplementedError

    @abstractmethod
    def metadata_schema(self):
        raise NotImplementedError

    @abstractmethod
    def metadata_array_uri(self):
        raise NotImplementedError
    
    @abstractmethod
    def metadata_array_object_id_dim(self):
        raise NotImplementedError

    @abstractmethod
    def get_partitions(self, partition_size: int = -1) -> List[ObjectPartition]:
        raise NotImplementedError

    @abstractmethod
    def read_objects(self, partition: ObjectPartition):
        raise NotImplementedError

    @abstractmethod
    def read_objects_by_ids(self, ids):
        raise NotImplementedError
