from typing import Dict, OrderedDict, Tuple

import numpy as np

from tiledb.vector_search.embeddings import ObjectEmbedding

EMBED_DIM = 128


class ColpaliEmbedding(ObjectEmbedding):
    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.2",
        device: str = None,
        batch_size: int = 4,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.processor = None

    def init_kwargs(self) -> Dict:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
        }

    def dimensions(self) -> int:
        return EMBED_DIM

    def vector_type(self) -> np.dtype:
        return np.float32

    def load(self) -> None:
        import torch
        from colpali_engine.models import ColPali
        from colpali_engine.models import ColPaliProcessor

        if self.device is None:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # Load model
        self.model = ColPali.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map=self.device
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(self.model_name)

    def embed(
        self, objects: OrderedDict, metadata: OrderedDict
    ) -> Tuple[np.ndarray, np.array]:
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        if "image" in objects:
            images = []
            for i in range(len(objects["image"])):
                images.append(
                    Image.fromarray(
                        np.reshape(objects["image"][i], objects["shape"][i])
                    )
                )
            dataloader = DataLoader(
                images,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=lambda x: self.processor.process_images(x),
            )
        elif "text" in objects:
            dataloader = DataLoader(
                objects["text"],
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=lambda x: self.processor.process_queries(x),
            )

        embeddings = None
        external_ids = None
        id = 0
        for batch in tqdm(dataloader):
            with torch.no_grad():
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                batch_embeddings = list(torch.unbind(self.model(**batch).to("cpu")))
            for object_embeddings in batch_embeddings:
                object_embeddings_np = object_embeddings.to(torch.float32).cpu().numpy()
                ext_ids = metadata["external_id"][id] * np.ones(
                    object_embeddings_np.shape[0], dtype=np.uint64
                )
                if embeddings is None:
                    external_ids = ext_ids
                    embeddings = object_embeddings_np
                else:
                    external_ids = np.concatenate((external_ids, ext_ids))
                    embeddings = np.vstack((embeddings, object_embeddings_np))
                id += 1
        return (embeddings, external_ids)
