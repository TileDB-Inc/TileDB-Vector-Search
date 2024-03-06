from collections import OrderedDict
from typing import Dict, List, OrderedDict, Tuple

import numpy as np

import tiledb
from tiledb.vector_search.embeddings import ObjectEmbedding
from tiledb.vector_search.object_api import object_index
from tiledb.vector_search.object_readers import ObjectPartition
from tiledb.vector_search.object_readers import ObjectReader

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
        embeddings = np.zeros(
            (len(objects["object"]), EMBED_DIM), dtype=self.vector_type()
        )
        for i in range(len(objects["object"])):
            for j in range(EMBED_DIM):
                embeddings[i, j] = objects["object"][i][0]
        return embeddings


class TestPartition(ObjectPartition):
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


class TestReader(ObjectReader):
    def __init__(
        self,
        num_objects: int,
        **kwargs,
    ):
        self.num_objects = num_objects

    def init_kwargs(self) -> Dict:
        return {
            "num_objects": self.num_objects,
        }

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

    def get_partitions(self, objects_per_partition: int = -1) -> List[TestPartition]:
        if objects_per_partition == -1:
            objects_per_partition = 42
        partitions = []
        partition_id = 0
        for start in range(0, self.num_objects, objects_per_partition):
            end = start + objects_per_partition
            if end > self.num_objects:
                end = self.num_objects
            partitions.append(TestPartition(partition_id, start, end))
            partition_id += 1

        return partitions

    def read_objects(self, partition: TestPartition) -> Tuple[OrderedDict, OrderedDict]:
        external_ids = np.arange(partition.start, partition.end)
        num_vectors = partition.end - partition.start
        objects = np.empty(num_vectors, dtype="O")
        metadata = np.empty(num_vectors, dtype="O")
        i = 0
        for id in range(partition.start, partition.end):
            objects[i] = np.array([id, id, id, id])
            metadata[i] = id
            i += 1
        return (
            {"object": objects, "external_id": external_ids},
            {"test_attr": metadata, "external_id": external_ids},
        )

    def read_objects_by_external_ids(self, ids: List[int]) -> OrderedDict:
        objects = np.empty(len(ids), dtype="O")
        external_ids = np.zeros(len(ids))
        i = 0
        for id in ids:
            objects[i] = np.array([id, id, id, id])
            external_ids[i] = id
            i += 1
        return {"object": objects, "external_id": external_ids}


def test_object_index_ivf_flat(tmp_path):
    reader = TestReader(num_objects=1000)
    embedding = TestEmbedding()

    index_uri = f"{tmp_path}/index"

    index = object_index.create(
        uri=index_uri,
        index_type="IVF_FLAT",
        object_reader=reader,
        embedding=embedding,
    )

    index.update_index()
    index = object_index.ObjectIndex(uri=index_uri)

    index = object_index.ObjectIndex(uri=index_uri, load_metadata_in_memory=False)
    distances, objects, metadata = index.query(
        {"object": np.array([[12, 12, 12, 12]])},
        k=5,
        nprobe=10,
    )
    assert np.array_equiv(
        np.unique(objects["external_id"]), np.array([10, 11, 12, 13, 14])
    )

    distances, objects, metadata = index.query(
        {"object": np.array([[12, 12, 12, 12]])},
        metadata_array_cond="test_attr >= 12",
        k=5,
        nprobe=10,
    )
    assert np.array_equiv(objects["external_id"], np.array([12, 13, 14, 15, 16]))

    index = object_index.ObjectIndex(uri=index_uri, load_metadata_in_memory=False)
    distances, objects, metadata = index.query(
        {"object": np.array([[12, 12, 12, 12]])},
        k=5,
        nprobe=10,
    )
    assert np.array_equiv(
        np.unique(objects["external_id"]), np.array([10, 11, 12, 13, 14])
    )

    def df_filter(row):
        return row["test_attr"] >= 12

    distances, objects, metadata = index.query(
        {"object": np.array([[12, 12, 12, 12]])},
        metadata_df_filter_fn=df_filter,
        k=5,
        nprobe=10,
    )
    assert np.array_equiv(objects["external_id"], np.array([12, 13, 14, 15, 16]))


def test_object_index_flat(tmp_path):
    reader = TestReader(num_objects=1000)
    embedding = TestEmbedding()

    index_uri = f"{tmp_path}/index"

    index = object_index.create(
        uri=index_uri,
        index_type="FLAT",
        object_reader=reader,
        embedding=embedding,
    )

    index.update_index()
    index = object_index.ObjectIndex(uri=index_uri)

    index = object_index.ObjectIndex(uri=index_uri)
    distances, objects, metadata = index.query(
        {"object": np.array([[12, 12, 12, 12]])}, k=5
    )
    assert np.array_equiv(
        np.unique(objects["external_id"]), np.array([10, 11, 12, 13, 14])
    )
    distances, object_ids = index.query(
        {"object": np.array([[12, 12, 12, 12]])},
        k=5,
        return_objects=False,
        return_metadata=False,
    )
    assert np.array_equiv(np.unique(object_ids), np.array([10, 11, 12, 13, 14]))

    def df_filter(row):
        return row["test_attr"] >= 12

    distances, objects, metadata = index.query(
        {"object": np.array([[12, 12, 12, 12]])},
        metadata_df_filter_fn=df_filter,
        k=5,
    )
    assert np.array_equiv(objects["external_id"], np.array([12, 13, 14, 15, 16]))

    distances, object_ids = index.query(
        {"object": np.array([[12, 12, 12, 12]])},
        metadata_df_filter_fn=df_filter,
        k=5,
        return_objects=False,
        return_metadata=False,
    )
    assert np.array_equiv(object_ids, np.array([12, 13, 14, 15, 16]))

    index = object_index.ObjectIndex(uri=index_uri, load_metadata_in_memory=False)
    distances, objects, metadata = index.query(
        {"object": np.array([[12, 12, 12, 12]])}, k=5
    )
    assert np.array_equiv(
        np.unique(objects["external_id"]), np.array([10, 11, 12, 13, 14])
    )
    distances, object_ids = index.query(
        {"object": np.array([[12, 12, 12, 12]])},
        k=5,
        return_objects=False,
        return_metadata=False,
    )
    assert np.array_equiv(np.unique(object_ids), np.array([10, 11, 12, 13, 14]))

    distances, objects, metadata = index.query(
        {"object": np.array([[12, 12, 12, 12]])},
        metadata_array_cond="test_attr >= 12",
        k=5,
    )
    assert np.array_equiv(objects["external_id"], np.array([12, 13, 14, 15, 16]))

    distances, object_ids = index.query(
        {"object": np.array([[12, 12, 12, 12]])},
        metadata_array_cond="test_attr >= 12",
        k=5,
        return_objects=False,
        return_metadata=False,
    )
    assert np.array_equiv(object_ids, np.array([12, 13, 14, 15, 16]))
