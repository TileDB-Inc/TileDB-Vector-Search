from typing import Any, Mapping, Optional

import numpy as np
# from tiledb.vector_search.embeddings import ObjectEmbedding

EMBED_DIM = 2048


# class RandomEmbedding(ObjectEmbedding):
class RandomEmbedding():
    def __init__(
        self,
    ):
        self.model = None

    def get_kwargs(self):
        return {}
    
    def dimensions(self) -> int:
        return EMBED_DIM

    def vector_type(self) -> np.dtype:
        return np.float32

    def load(self):
        pass

    def embed(self, objects, metadata=None) -> np.ndarray:
        size = len(objects["image"])
        return np.random.rand(size,EMBED_DIM).astype(self.vector_type())
