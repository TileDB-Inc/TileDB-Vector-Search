from typing import Dict, Optional, OrderedDict, Sequence, Union

import numpy as np

# from tiledb.vector_search.embeddings import ObjectEmbedding


class OllamaEmbedding:
    """
    Embedding functions from Ollama.

    This attempts to import the embedding_class from the ollama module.
    """

    def __init__(
        self,
        dimensions: int,
        embedding_class: str = "embed",  # really it's the method
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
            embeddings_module = importlib.import_module("ollama")
            embedding_method_ = getattr(embeddings_module, self.embedding_class)
            self.embedding = embedding_method_(**self.embedding_kwargs)
        except ImportError as e:
            print(e)

    def embed(self, objects: Union[str, Sequence[str]]) -> np.ndarray:
        return np.array(self.embedding(input=objects).embeddings, dtype=np.float32)
