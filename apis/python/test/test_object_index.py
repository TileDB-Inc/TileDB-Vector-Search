import pytest
import tiledb
import os
import numpy as np
from typing import Any, Mapping, Optional, List, Dict, Tuple, OrderedDict

from collections import OrderedDict
from tiledb.vector_search.object_readers import ObjectPartition, ObjectReader
from tiledb.vector_search.embeddings import ObjectEmbedding
from tiledb.vector_search.object_api import object_index

EMBED_DIM = 4

# TestEmbedding with vectors of EMBED_DIM size with all values being the id of the object
# i.e [1, 1, 1, 1], [2, 2, 2, 2], etc.
class TestEmbedding(ObjectEmbedding):
    def __init__(
        self,
    ):
        self.model = None

    def init_kwargs(self) -> Dict:
        return {}
    
    def dimensions(self) -> int:
        return EMBED_DIM

    def vector_type(self) -> np.dtype:
        return np.float32

    def load(self) -> None:
        pass

    def embed(self, objects: OrderedDict, metadata: OrderedDict) -> np.ndarray:
        embeddings = np.zeros((len(objects["object"]), EMBED_DIM), dtype=self.vector_type())
        for i in range(len(objects["object"])):
            for j in range(EMBED_DIM):
                embeddings[i, j] = objects["object"][i][0]
        return embeddings

class TestPartition(ObjectPartition):
    def __init__(
        self,
        start: int,
        end: int,
        **kwargs,
    ):
        self.start = start
        self.end = end
        self.size = end - start

    def init_kwargs(self) -> Dict:
        return {
            "start": self.start,
            "end": self.end,
        }

    def num_vectors(self) -> int:
        return self.size

    def index_slice(self) -> Tuple[int,int]:
        return (self.start, self.end)

class TestReader(ObjectReader):
    def __init__(
        self,
        conf_val: str,
        **kwargs,
    ):
        self.conf_val = conf_val

    def init_kwargs(self) -> Dict:
        return {
            "conf_val": self.conf_val,
        }
    
    def num_vectors(self) -> int:
        return 42

    def partition_class_name(self) -> str:
        return "TestPartition"

    def metadata_array_uri(self) -> str:
        return None

    def metadata_attributes(self) -> List[tiledb.Attr]:
        test_attr = tiledb.Attr(
            name="test_attr",
            dtype=np.uint32,
        )
        return [test_attr]

    def get_partitions(self, partition_size: int = -1) -> List[TestPartition]:
        return [TestPartition(0,21), TestPartition(21,42)]

    def read_objects(self, partition: TestPartition) -> Tuple[OrderedDict, OrderedDict]:
        external_ids = np.arange(partition.start, partition.end)
        objects = np.empty(partition.num_vectors(), dtype="O")
        metadata = np.empty(partition.num_vectors(), dtype="O")
        i = 0 
        for id in range(partition.start, partition.end):
            objects[i]=np.array([id, id, id, id])
            metadata[i]=id
            i += 1
        return (
            {"object": objects, "external_id": external_ids}, 
            {"test_attr": metadata, "external_id": external_ids}
            )

    def read_objects_by_external_ids(self, ids: List[int]) -> OrderedDict:
        objects = np.empty(len(ids), dtype="O")
        external_ids = np.zeros(len(ids))
        i = 0 
        for id in ids:
            objects[i]=np.array([id, id, id, id])
            external_ids[i] = id
            i += 1
        return {"object": objects, "external_id": external_ids}



def test_object_index_ivf_flat(tmp_path):
    reader = TestReader(conf_val="42")
    embedding = TestEmbedding()

    index_uri = f"{tmp_path}/index"

    index = object_index.create(
        uri=index_uri,
        index_type="IVF_FLAT",
        object_reader=reader,
        embedding=embedding,
    )

    index.update_index()

    distances, objects, metadata = index.query(
            {"object": np.array([[12, 12, 12, 12]])}, 
            k=5, 
            nprobe=10,
        )
    assert(np.array_equiv(np.unique(objects["external_id"]), np.array([10, 11, 12, 13, 14])))

    distances, objects, metadata = index.query(
            {"object": np.array([[12, 12, 12, 12]])}, 
            metadata_array_cond=f"test_attr >= 12",
            k=5, 
            nprobe=10
        )
    assert(np.array_equiv(objects["external_id"], np.array([12, 13, 14, 15, 16])))


def test_object_index_flat(tmp_path):
    reader = TestReader(conf_val="42")
    embedding = TestEmbedding()

    index_uri = f"{tmp_path}/index"

    index = object_index.create(
        uri=index_uri,
        index_type="FLAT",
        object_reader=reader,
        embedding=embedding,
    )

    index.update_index()

    distances, objects, metadata = index.query(
            {"object": np.array([[12, 12, 12, 12]])}, 
            k=5
        )
    assert(np.array_equiv(np.unique(objects["external_id"]), np.array([10, 11, 12, 13, 14])))

    distances, objects, metadata = index.query(
            {"object": np.array([[12, 12, 12, 12]])}, 
            metadata_array_cond=f"test_attr >= 12",
            k=5
        )
    assert(np.array_equiv(objects["external_id"], np.array([12, 13, 14, 15, 16])))