from abc import ABC
from abc import abstractmethod
from typing import Dict, OrderedDict

import numpy as np


class ObjectEmbedding(ABC):
    """
    Abstract class that can be used to create embeddings for Objects of a specific format.
    """

    @abstractmethod
    def init_kwargs(self) -> Dict:
        """
        Returns a dictionary containing kwargs that can be used to re-initialize the ObjectEmbedding.

        This is used to serialize the ObjectEmbedding and pass it as argument to UDF tasks.
        """
        raise NotImplementedError

    @abstractmethod
    def dimensions(self) -> int:
        """
        Returns the number of dimensions of the embedding vectors.
        """
        raise NotImplementedError

    @abstractmethod
    def vector_type(self) -> np.dtype:
        """
        Returns the datatype of the embedding vectors.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        """
        Loads the model in order to be ready for embedding objects.

        This method will be called once per worker to avoid loading the model multiple times.
        """
        raise NotImplementedError

    @abstractmethod
    def embed(self, objects: OrderedDict, metadata: OrderedDict) -> np.ndarray:
        """
        Creates embedding vectors for objects. Returns a numpy array of embedding vectors.
        There is no enforced restriction on the object format. ObjectReaders and ObjectEmbeddings should use comatible object and metadata formats.

        Parameters
        ----------
        objects: OrderedDict
            An OrderedDict, containing the object data, having structure similar to TileDB-Py read results.
        metadata: OrderedDict
            An OrderedDict, containing the object metadata, having structure similar to TileDB-Py read results.
        """
        raise NotImplementedError
