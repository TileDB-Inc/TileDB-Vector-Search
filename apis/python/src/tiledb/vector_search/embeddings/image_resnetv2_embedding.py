from typing import Dict, OrderedDict

import numpy as np

# from tiledb.vector_search.embeddings import ObjectEmbedding

EMBED_DIM = 2048


# class ImageResNetV2Embedding(ObjectEmbedding):
class ImageResNetV2Embedding:
    def __init__(
        self,
    ):
        self.model = None

    def init_kwargs(self) -> Dict:
        return {}

    def dimensions(self) -> int:
        return EMBED_DIM

    def vector_type(self) -> np.dtype:
        return np.float32

    def load(self) -> None:
        import tensorflow as tf

        self.model = tf.keras.applications.ResNet50V2(include_top=False)

    def embed(self, objects: OrderedDict, metadata: OrderedDict) -> np.ndarray:
        from efficientnet.preprocessing import center_crop_and_resize
        from tensorflow.keras.applications.resnet_v2 import preprocess_input

        size = len(objects["image"])
        crop_size = 224
        images = np.zeros((size, crop_size, crop_size, 3), dtype=np.uint8)
        for image_id in range(len(objects["image"])):
            images[image_id] = center_crop_and_resize(
                np.reshape(objects["image"][image_id], objects["shape"][image_id]),
                crop_size,
            ).astype(np.uint8)
        maps = self.model.predict(preprocess_input(images))
        if np.prod(maps.shape) == maps.shape[-1] * len(objects):
            return np.squeeze(maps)
        else:
            return maps.mean(axis=1).mean(axis=1)
