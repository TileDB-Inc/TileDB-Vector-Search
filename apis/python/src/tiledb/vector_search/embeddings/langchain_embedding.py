from typing import Dict, Optional, OrderedDict

import numpy as np

# from tiledb.vector_search.embeddings import ObjectEmbedding


# class LangChainEmbedding(ObjectEmbedding):
class LangChainEmbedding:
    """
    Embedding functions from Langchain.

    This attempts to import the embedding_class from the following modules:
    - langchain_openai
    - langchain.embeddings
    """

    def __init__(
        self,
        dimensions: int,
        embedding_class: str = "OpenAIEmbeddings",
        embedding_kwargs: Optional[Dict] = None,
    ):
        self.dim_num = dimensions
        self.embedding_class = embedding_class
        self.embedding_kwargs = embedding_kwargs

    def init_kwargs(self) -> Dict:
        return {
            "dimensions": self.dim_num,
            "embedding_class": self.embedding_class,
            "embedding_kwargs": self.embedding_kwargs,
        }

    def dimensions(self) -> int:
        return self.dim_num

    def vector_type(self) -> np.dtype:
        return np.float32

    def load(self) -> None:
        import importlib

        try:
            embeddings_module = importlib.import_module("langchain_openai")
            embedding_class_ = getattr(embeddings_module, self.embedding_class)
            self.embedding = embedding_class_(**self.embedding_kwargs)
        except ImportError:
            embeddings_module = importlib.import_module("langchain.embeddings")
            embedding_class_ = getattr(embeddings_module, self.embedding_class)
            self.embedding = embedding_class_(**self.embedding_kwargs)

    def embed(self, objects: OrderedDict, metadata: OrderedDict) -> np.ndarray:
        return np.array(
            self.embedding.embed_documents(objects["text"]), dtype=np.float32
        )
