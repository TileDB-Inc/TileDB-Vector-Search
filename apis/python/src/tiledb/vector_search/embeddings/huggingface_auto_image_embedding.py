from typing import Dict, Optional, OrderedDict

import numpy as np


class HuggingfaceAutoImageEmbedding:
    def __init__(
        self,
        model_name_or_path: str,
        dimensions: int,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        batch_size: int = 64,
    ):
        self.model_name_or_path = model_name_or_path
        self.dim_num = dimensions
        self.device = device
        self.cache_folder = cache_folder
        self.batch_size = batch_size
        self.processor = None
        self.model = None

    def init_kwargs(self) -> Dict:
        return {
            "model_name_or_path": self.model_name_or_path,
            "dimensions": self.dim_num,
            "device": self.device,
            "cache_folder": self.cache_folder,
            "batch_size": self.batch_size,
        }

    def dimensions(self) -> int:
        return self.dim_num

    def vector_type(self) -> np.dtype:
        return np.float32

    def load(self) -> None:
        from transformers import AutoImageProcessor
        from transformers import AutoModel

        self.processor = AutoImageProcessor.from_pretrained(self.model_name_or_path)
        self.model = AutoModel.from_pretrained(self.model_name_or_path)

    def embed(self, objects: OrderedDict, metadata: OrderedDict) -> np.ndarray:
        from PIL import Image

        write_id = 0
        count = 0
        image_batch = []
        size = len(objects["image"])
        embeddings = np.zeros((size, self.dim_num), dtype=np.float32)
        for image_id in range(len(objects["image"])):
            image_batch.append(
                Image.fromarray(
                    np.reshape(objects["image"][image_id], objects["shape"][image_id])
                )
            )
            count += 1
            if count >= self.batch_size:
                print(image_id)
                inputs = self.processor(images=image_batch, return_tensors="pt")
                batch_embeddings = (
                    self.model(**inputs).last_hidden_state[:, 0].cpu().detach().numpy()
                )
                embeddings[write_id : write_id + count] = batch_embeddings
                count = 0
                image_batch = []

        if count > 0:
            inputs = self.processor(images=image_batch, return_tensors="pt")
            batch_embeddings = (
                self.model(**inputs).last_hidden_state[:, 0].cpu().detach().numpy()
            )
            embeddings[write_id : write_id + count] = batch_embeddings
        return embeddings
