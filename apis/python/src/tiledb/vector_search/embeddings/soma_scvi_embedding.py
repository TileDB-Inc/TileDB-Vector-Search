from typing import Dict, OrderedDict

import numpy as np

from tiledb.vector_search.embeddings import ObjectEmbedding


class SomaSCVIEmbedding(ObjectEmbedding):
    def __init__(
        self,
        model_uri: str,
        gene_col: str = "feature_id",
        embedding_dimensions: int = 50,
        local_model_cache_dir: str = None,
    ):
        self.model_uri = model_uri
        self.gene_col = gene_col
        self.embedding_dimensions = embedding_dimensions
        self.local_model_cache_dir = local_model_cache_dir

        self.model_local_path = self.get_model_local_path()
        self.loaded = False

    def init_kwargs(self) -> Dict:
        return {
            "model_uri": self.model_uri,
            "gene_col": self.gene_col,
            "embedding_dimensions": self.embedding_dimensions,
            "local_model_cache_dir": self.local_model_cache_dir,
        }

    def dimensions(self) -> int:
        return self.embedding_dimensions

    def vector_type(self) -> np.dtype:
        return np.float32

    def get_model_local_path(self):
        import base64
        import os
        import tempfile

        encoded_model_uri = base64.b64encode(self.model_uri.encode("ascii")).decode(
            "ascii"
        )
        local_dir = tempfile.gettempdir()
        if self.local_model_cache_dir is not None:
            local_dir = self.local_model_cache_dir

        return os.path.join(
            local_dir,
            f"local_scvi_model_{encoded_model_uri}",
        )

    def load(self) -> None:
        import os

        import tiledb

        if self.loaded:
            return

        filename = "model.pt"
        # Check if model_uri is a local path
        if os.path.exists(self.model_uri):
            self.model_local_path = self.model_uri
            self.loaded = True
            return
        elif self.model_uri.startswith("http"):
            # Download model file from URL
            import logging

            from scvi.data._download import _download

            logging.getLogger("scvi.data._download").setLevel(logging.ERROR)

            _download(self.model_uri, self.model_local_path, filename)
            self.loaded = True
            return
        elif tiledb.array_exists(self.model_uri):
            # Load from TileDB URI
            if os.path.exists(os.path.join(self.model_local_path, filename)):
                self.loaded = True
                return
            # Fetch model from TileDB URI
            if not os.path.exists(self.model_local_path):
                os.makedirs(self.model_local_path)
            model_file = tiledb.filestore.Filestore(self.model_uri)
            local_file = os.path.join(self.model_local_path, "model.pt")
            with open(local_file, "wb") as f:
                f.write(model_file.read())
            self.loaded = True
            return

        raise ValueError(f"model_uri {self.model_uri} not a valid model URI.")

    def embed(self, objects: OrderedDict, metadata: OrderedDict) -> np.ndarray:
        import warnings

        warnings.filterwarnings("ignore")
        import scvi

        adata = objects["anndata"]

        adata.var_names = adata.var[self.gene_col]
        adata.obs["batch"] = "unassigned"

        scvi.model.SCVI.prepare_query_anndata(adata, self.model_local_path)
        vae_q = scvi.model.SCVI.load_query_data(adata, self.model_local_path)
        vae_q.is_trained = True
        return vae_q.get_latent_representation()
