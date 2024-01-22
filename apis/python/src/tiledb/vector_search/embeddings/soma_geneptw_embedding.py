from typing import Any, Mapping, Optional, Dict, OrderedDict

import numpy as np
from tiledb.vector_search.embeddings import ObjectEmbedding

EMBED_DIM = 1536  # embedding dim from GPT-3.5


class SomaGenePTwEmbedding(ObjectEmbedding):
    def __init__(
        self,
        gene_embeddings_uri: str,
        soma_uri: str,
        config: Optional[Mapping[str, Any]] = None,
    ):
        self.gene_embeddings_uri = gene_embeddings_uri
        self.soma_uri = soma_uri
        self.config = config
        self.gene_embedding = None
        self.gene_names = None

    def init_kwargs(self) -> Dict:
        return {
            "gene_embeddings_uri": self.gene_embeddings_uri,
            "soma_uri": self.soma_uri,
            "config": self.config,
        }

    def dimensions(self) -> int:
        return EMBED_DIM

    def vector_type(self) -> np.dtype:
        return np.float32

    def load(self) -> None:
        import tiledb
        import tiledbsoma
        import numpy as np

        print("Loading gene embeddings...")
        gene_pt_embeddings = {}
        with tiledb.open(self.gene_embeddings_uri, "r", config=self.config) as gene_pt_array:
            gene_pt = gene_pt_array[:]
            i = 0
            for gene in np.array(gene_pt["genes"], dtype=str):
                gene_pt_embeddings[str(gene)] = gene_pt["embeddings"][i]
                i += 1

        print("Loading var genes...")
        context = tiledbsoma.SOMATileDBContext(tiledb_ctx=tiledb.Ctx(self.config))
        experiment = tiledbsoma.Experiment.open(self.soma_uri, "r", context=context)
        self.gene_names = experiment.ms["RNA"].var.read().concat().to_pandas()["feature_name"].to_numpy()

        print("Computing model...")
        count_missing = 0
        self.gene_embedding = np.zeros(shape=(len(self.gene_names), EMBED_DIM))
        for i, gene in enumerate(self.gene_names):
            if gene in gene_pt_embeddings:
                self.gene_embedding[i, :] = gene_pt_embeddings[gene]
            else:
                count_missing += 1

    def embed(self, objects: OrderedDict, metadata: OrderedDict) -> np.ndarray:
        import numpy as np
        return np.array(np.dot(objects["data"], self.gene_embedding)/len(self.gene_names), dtype=np.float32)
