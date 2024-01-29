from typing import Dict, OrderedDict

import numpy as np

# from tiledb.vector_search.embeddings import ObjectEmbedding

EMBED_DIM = 2048


# class RandomEmbedding(ObjectEmbedding):
class RandomEmbedding:
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
        size = len(objects[list(objects.keys())[0]])
        return np.random.rand(size, EMBED_DIM).astype(self.vector_type())
