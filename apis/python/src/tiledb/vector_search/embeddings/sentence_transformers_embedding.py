from typing import Dict, Optional, OrderedDict

import numpy as np

# from tiledb.vector_search.embeddings import ObjectEmbedding


# class SentenceTransformersEmbedding(ObjectEmbedding):
class SentenceTransformersEmbedding:
    """
    Hugging SentenceTransformer model that can be used to map sentences / text to embeddings.

    :param model_name_or_path: If it is a filepath on disc, it loads the model from that path. If it is not a path,
        it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model
        from the Hugging Face Hub with that name.
    :param device: Device (like "cuda", "cpu", "mps") that should be used for computation. If None, checks if a GPU
        can be used.
    :param cache_folder: Path to store models. Can also be set by the SENTENCE_TRANSFORMERS_HOME environment variable.
    :param dimensions: Number of dimensions of the embedding.

    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        dimensions: Optional[int] = -1,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.cache_folder = cache_folder
        self.dim_num = dimensions
        self.model = None
        if self.dim_num == -1:
            self.load()
            self.dim_num = self.model.get_sentence_embedding_dimension()

    def init_kwargs(self) -> Dict:
        return {
            "model_name_or_path": self.model_name_or_path,
            "device": self.device,
            "cache_folder": self.cache_folder,
            "dimensions": self.dim_num,
        }

    def dimensions(self) -> int:
        return self.dim_num

    def vector_type(self) -> np.dtype:
        return np.float32

    def load(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name_or_path=self.model_name_or_path,
            device=self.device,
            cache_folder=self.cache_folder,
        )

    def embed(self, objects: OrderedDict, metadata: OrderedDict) -> np.ndarray:
        return self.model.encode(objects["text"], normalize_embeddings=True)
