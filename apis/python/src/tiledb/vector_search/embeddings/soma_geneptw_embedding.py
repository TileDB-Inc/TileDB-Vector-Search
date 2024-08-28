from typing import Dict, OrderedDict

import numpy as np

from tiledb.vector_search.embeddings import ObjectEmbedding

EMBED_DIM = 1536  # embedding dim from GPT-3.5


class SomaGenePTwEmbedding(ObjectEmbedding):
    def __init__(
        self,
        gene_embeddings_uri: str,
        soma_uri: str,
        measurement_name: str = "RNA",
        gene_col: str = "feature_name",
    ):
        self.gene_embeddings_uri = gene_embeddings_uri
        self.soma_uri = soma_uri
        self.measurement_name = measurement_name
        self.gene_col = gene_col

        self.gene_embedding = None
        self.gene_names = None

    def init_kwargs(self) -> Dict:
        return {
            "gene_embeddings_uri": self.gene_embeddings_uri,
            "soma_uri": self.soma_uri,
            "measurement_name": self.measurement_name,
            "gene_col": self.gene_col,
        }

    def dimensions(self) -> int:
        return EMBED_DIM

    def vector_type(self) -> np.dtype:
        return np.float32

    def load(self) -> None:
        import numpy as np
        import tiledbsoma

        import tiledb

        gene_pt_embeddings = {}
        with tiledb.open(
            self.gene_embeddings_uri, "r", config=self.config
        ) as gene_pt_array:
            gene_pt = gene_pt_array[:]
            i = 0
            for gene in np.array(gene_pt["genes"], dtype=str):
                gene_pt_embeddings[str(gene)] = gene_pt["embeddings"][i]
                i += 1

        tiledbsoma.SOMATileDBContext(tiledb_config=tiledb.default_ctx().config().dict())
        experiment = tiledbsoma.Experiment.open(
            self.soma_uri, "r", context=self.context
        )
        self.gene_names = (
            experiment.ms[self.measurement_name]
            .var.read()
            .concat()
            .to_pandas()[self.gene_col]
            .to_numpy()
        )

        self.gene_embedding = np.zeros(shape=(len(self.gene_names), EMBED_DIM))
        for i, gene in enumerate(self.gene_names):
            if gene in gene_pt_embeddings:
                self.gene_embedding[i, :] = gene_pt_embeddings[gene]

    def embed(self, objects: OrderedDict, metadata: OrderedDict) -> np.ndarray:
        import numpy as np

        return np.array(
            np.dot(objects["anndata"].X.toarray(), self.gene_embedding)
            / len(self.gene_names),
            dtype=np.float32,
        )
