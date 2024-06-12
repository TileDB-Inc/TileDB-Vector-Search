import os
from typing import Dict, List, OrderedDict, Tuple

import numpy as np

import tiledb
from tiledb.cloud.dag import Mode
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
        object_id_start: int,
        object_id_end: int,
        vector_dim_offset: int,
        **kwargs,
    ):
        self.object_id_start = object_id_start
        self.object_id_end = object_id_end
        self.vector_dim_offset = vector_dim_offset

    def init_kwargs(self) -> Dict:
        return {
            "object_id_start": self.object_id_start,
            "object_id_end": self.object_id_end,
            "vector_dim_offset": self.vector_dim_offset,
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

    def get_partitions(
        self, objects_per_partition: int = -1, **kwargs
    ) -> List[TestPartition]:
        if objects_per_partition == -1:
            objects_per_partition = 42
        partitions = []
        partition_id = 0
        for start in range(
            self.object_id_start, self.object_id_end, objects_per_partition
        ):
            end = start + objects_per_partition
            if end > self.object_id_end:
                end = self.object_id_end
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
            vector_dim = self.vector_dim_offset + id
            objects[i] = np.array([vector_dim, vector_dim, vector_dim, vector_dim])
            metadata[i] = vector_dim
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
            vector_dim = self.vector_dim_offset + id
            objects[i] = np.array([vector_dim, vector_dim, vector_dim, vector_dim])
            external_ids[i] = id
            i += 1
        return {"object": objects, "external_id": external_ids}


def evaluate_query(index_uri, query_kwargs, dim_id, vector_dim_offset, config=None):
    v_id = dim_id - vector_dim_offset
    index = object_index.ObjectIndex(uri=index_uri, config=config)
    distances, objects, metadata = index.query(
        {"object": np.array([[dim_id, dim_id, dim_id, dim_id]])}, k=5, **query_kwargs
    )
    assert np.array_equiv(
        np.unique(objects["external_id"]),
        np.array([v_id - 2, v_id - 1, v_id, v_id + 1, v_id + 2]),
    )
    distances, object_ids = index.query(
        {"object": np.array([[dim_id, dim_id, dim_id, dim_id]])},
        k=5,
        return_objects=False,
        return_metadata=False,
        **query_kwargs,
    )
    assert np.array_equiv(
        np.unique(object_ids), np.array([v_id - 2, v_id - 1, v_id, v_id + 1, v_id + 2])
    )

    def df_filter(row):
        return row["test_attr"] >= dim_id

    distances, objects, metadata = index.query(
        {"object": np.array([[dim_id, dim_id, dim_id, dim_id]])},
        metadata_df_filter_fn=df_filter,
        k=5,
        **query_kwargs,
    )
    assert np.array_equiv(
        objects["external_id"], np.array([v_id, v_id + 1, v_id + 2, v_id + 3, v_id + 4])
    )

    distances, object_ids = index.query(
        {"object": np.array([[dim_id, dim_id, dim_id, dim_id]])},
        metadata_df_filter_fn=df_filter,
        k=5,
        return_objects=False,
        return_metadata=False,
        **query_kwargs,
    )
    assert np.array_equiv(
        object_ids, np.array([v_id, v_id + 1, v_id + 2, v_id + 3, v_id + 4])
    )

    index = object_index.ObjectIndex(
        uri=index_uri, load_metadata_in_memory=False, config=config
    )
    distances, objects, metadata = index.query(
        {"object": np.array([[dim_id, dim_id, dim_id, dim_id]])}, k=5, **query_kwargs
    )
    assert np.array_equiv(
        np.unique(objects["external_id"]),
        np.array([v_id - 2, v_id - 1, v_id, v_id + 1, v_id + 2]),
    )
    distances, object_ids = index.query(
        {"object": np.array([[dim_id, dim_id, dim_id, dim_id]])},
        k=5,
        return_objects=False,
        return_metadata=False,
        **query_kwargs,
    )
    assert np.array_equiv(
        np.unique(object_ids), np.array([v_id - 2, v_id - 1, v_id, v_id + 1, v_id + 2])
    )

    distances, objects, metadata = index.query(
        {"object": np.array([[dim_id, dim_id, dim_id, dim_id]])},
        metadata_array_cond=f"test_attr >= {dim_id}",
        k=5,
        **query_kwargs,
    )
    assert np.array_equiv(
        objects["external_id"], np.array([v_id, v_id + 1, v_id + 2, v_id + 3, v_id + 4])
    )

    distances, object_ids = index.query(
        {"object": np.array([[dim_id, dim_id, dim_id, dim_id]])},
        metadata_array_cond=f"test_attr >= {dim_id}",
        k=5,
        return_objects=False,
        return_metadata=False,
        **query_kwargs,
    )
    assert np.array_equiv(
        object_ids, np.array([v_id, v_id + 1, v_id + 2, v_id + 3, v_id + 4])
    )


def test_object_index(tmp_path):
    from common import INDEXES

    for index_type in INDEXES:
        index_uri = os.path.join(tmp_path, f"object_index_{index_type}")
        reader = TestReader(
            object_id_start=0,
            object_id_end=1000,
            vector_dim_offset=0,
        )
        embedding = TestEmbedding()

        index = object_index.create(
            uri=index_uri,
            index_type=index_type,
            object_reader=reader,
            embedding=embedding,
            num_subspaces=4,
        )

        # Check initial ingestion
        index.update_index(partitions=10)

        # TODO(SC-48908): Fix IVF_PQ with object index queries and remove.
        if index_type == "IVF_PQ":
            continue

        evaluate_query(
            index_uri=index_uri,
            query_kwargs={"nprobe": 10, "l_search": 250},
            dim_id=42,
            vector_dim_offset=0,
        )

        # Check that updating the same data doesn't create duplicates
        index = object_index.ObjectIndex(uri=index_uri)
        index.update_index(partitions=10)
        evaluate_query(
            index_uri=index_uri,
            query_kwargs={"nprobe": 10, "l_search": 500},
            dim_id=42,
            vector_dim_offset=0,
        )

        # Add new data with a new reader
        reader = TestReader(
            object_id_start=1000,
            object_id_end=2000,
            vector_dim_offset=0,
        )
        index = object_index.ObjectIndex(uri=index_uri)
        index.update_object_reader(reader)
        index.update_index(partitions=10)
        evaluate_query(
            index_uri=index_uri,
            query_kwargs={"nprobe": 10, "l_search": 500},
            dim_id=1042,
            vector_dim_offset=0,
        )

        # Check overwritting existing data
        reader = TestReader(
            object_id_start=1000,
            object_id_end=2000,
            vector_dim_offset=1000,
        )
        index = object_index.ObjectIndex(uri=index_uri)
        index.update_object_reader(reader)
        index.update_index(partitions=10)
        evaluate_query(
            index_uri=index_uri,
            query_kwargs={"nprobe": 10, "l_search": 500},
            dim_id=2042,
            vector_dim_offset=1000,
        )


def test_object_index_ivf_flat_cloud(tmp_path):
    from common import create_cloud_uri
    from common import delete_uri
    from common import setUpCloudToken

    setUpCloudToken()
    config = tiledb.cloud.Config().dict()
    index_uri = create_cloud_uri("object_index_ivf_flat")
    worker_resources = {"cpu": "1", "memory": "2Gi"}
    reader = TestReader(
        object_id_start=0,
        object_id_end=1000,
        vector_dim_offset=0,
    )
    embedding = TestEmbedding()

    index = object_index.create(
        uri=index_uri,
        index_type="IVF_FLAT",
        object_reader=reader,
        embedding=embedding,
        config=config,
    )

    # Check initial ingestion
    index.update_index(
        embeddings_generation_driver_mode=Mode.BATCH,
        embeddings_generation_mode=Mode.BATCH,
        vector_indexing_mode=Mode.BATCH,
        workers=2,
        worker_resources=worker_resources,
        driver_resources=worker_resources,
        kmeans_resources=worker_resources,
        ingest_resources=worker_resources,
        consolidate_partition_resources=worker_resources,
        objects_per_partition=500,
        partitions=10,
        config=config,
    )
    evaluate_query(
        index_uri=index_uri,
        query_kwargs={"nprobe": 10},
        dim_id=42,
        vector_dim_offset=0,
        config=config,
    )

    # Add new data with a new reader
    reader = TestReader(
        object_id_start=1000,
        object_id_end=2000,
        vector_dim_offset=0,
    )
    index = object_index.ObjectIndex(uri=index_uri, config=config)
    index.update_object_reader(reader, config=config)
    index.update_index(
        embeddings_generation_driver_mode=Mode.BATCH,
        embeddings_generation_mode=Mode.BATCH,
        vector_indexing_mode=Mode.BATCH,
        workers=2,
        worker_resources=worker_resources,
        driver_resources=worker_resources,
        kmeans_resources=worker_resources,
        ingest_resources=worker_resources,
        consolidate_partition_resources=worker_resources,
        objects_per_partition=500,
        partitions=10,
        config=config,
    )
    evaluate_query(
        index_uri=index_uri,
        query_kwargs={"nprobe": 10},
        dim_id=1042,
        vector_dim_offset=0,
        config=config,
    )
    delete_uri(index_uri, config)


def test_object_index_flat(tmp_path):
    reader = TestReader(
        object_id_start=0,
        object_id_end=1000,
        vector_dim_offset=0,
    )
    embedding = TestEmbedding()

    index_uri = f"{tmp_path}/index"

    index = object_index.create(
        uri=index_uri,
        index_type="FLAT",
        object_reader=reader,
        embedding=embedding,
    )
    # Check initial ingestion
    index.update_index()
    evaluate_query(
        index_uri=index_uri,
        query_kwargs={},
        dim_id=42,
        vector_dim_offset=0,
    )

    # Check that updating the same data doesn't create duplicates
    index = object_index.ObjectIndex(uri=index_uri)
    index.update_index()
    evaluate_query(
        index_uri=index_uri,
        query_kwargs={},
        dim_id=42,
        vector_dim_offset=0,
    )

    # Add new data with a new reader
    reader = TestReader(
        object_id_start=1000,
        object_id_end=2000,
        vector_dim_offset=0,
    )
    index = object_index.ObjectIndex(uri=index_uri)
    index.update_object_reader(reader)
    index.update_index()
    evaluate_query(
        index_uri=index_uri,
        query_kwargs={},
        dim_id=1042,
        vector_dim_offset=0,
    )

    # Check overwritting existing data
    reader = TestReader(
        object_id_start=1000,
        object_id_end=2000,
        vector_dim_offset=1000,
    )
    index = object_index.ObjectIndex(uri=index_uri)
    index.update_object_reader(reader)
    index.update_index()
    evaluate_query(
        index_uri=index_uri,
        query_kwargs={},
        dim_id=2042,
        vector_dim_offset=1000,
    )
