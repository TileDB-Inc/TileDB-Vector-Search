from typing import Any, Dict, List, Mapping, Optional, OrderedDict, Tuple

import numpy as np

import tiledb

# from tiledb.vector_search.object_readers import ObjectPartition, ObjectReader

MAX_IMAGE_CROPS_PER_IMAGE = 10000


# class BioImagePartition(ObjectPartition):
class BioImagePartition:
    def __init__(
        self,
        partition_id: int,
        image_uris: List[str],
        image_id_start: int,
    ):
        self.partition_id = partition_id
        self.image_uris = image_uris
        self.image_id_start = image_id_start

    def init_kwargs(self) -> Dict:
        return {
            "partition_id": self.partition_id,
            "image_uris": self.image_uris,
            "image_id_start": self.image_id_start,
        }

    def id(self) -> int:
        return self.partition_id


# class BioImageReader(ObjectReader):
class BioImageReader:
    def __init__(
        self,
        uri: str,
        level: int = -1,
        object_crop_shape: Tuple[int, int] = None,
        config: Optional[Mapping[str, Any]] = None,
        timestamp=None,
    ):
        self.uri = uri
        self.level = level
        self.object_crop_shape = object_crop_shape
        self.config = config
        self.timestamp = timestamp
        self.images = None

    def init_kwargs(self) -> Dict:
        return {
            "uri": self.uri,
            "level": self.level,
            "object_crop_shape": self.object_crop_shape,
            "config": self.config,
            "timestamp": self.timestamp,
        }

    def partition_class_name(self) -> str:
        return "BioImagePartition"

    def metadata_array_uri(self) -> str:
        return None

    def metadata_attributes(self) -> List[tiledb.Attr]:
        image_uri_attr = tiledb.Attr(
            name="image_uri",
            dtype=str,
        )
        location_attr = tiledb.Attr(
            name="location",
            dtype=np.uint32,
            var=True,
        )
        return [image_uri_attr, location_attr]

    def get_partitions(
        self, images_per_partitions: int = -1, **kwargs
    ) -> List[BioImagePartition]:
        if images_per_partitions == -1:
            images_per_partitions = 1
        if self.images is None:
            vfs = tiledb.VFS(config=self.config)
            self.images = vfs.ls(self.uri)[1:]
        num_images = len(self.images)
        partitions = []
        partition_id = 0
        for start in range(0, num_images, images_per_partitions):
            end = start + images_per_partitions
            if end > num_images:
                end = num_images
            partitions.append(
                BioImagePartition(
                    partition_id,
                    image_uris=self.images[start:end],
                    image_id_start=start,
                )
            )
            partition_id += 1
        return partitions

    def read_objects(
        self, partition: BioImagePartition
    ) -> Tuple[OrderedDict, OrderedDict]:
        from tiledb.bioimg.openslide import TileDBOpenSlide

        def compute_external_id() -> int:
            id = image_id * MAX_IMAGE_CROPS_PER_IMAGE + image_iter_id
            return id

        def crop_image(dim_0_start, dim_0_end, dim_1_start, dim_1_end):
            cropped_image = image[dim_0_start:dim_0_end, dim_1_start:dim_1_end]
            images[write_id] = cropped_image.flatten()
            shapes[write_id] = np.array(cropped_image.shape, dtype=np.uint32)
            image_uris[write_id] = image_uri
            locations[write_id] = np.array(
                [dim_0_start, dim_0_end, dim_1_start, dim_1_end], dtype=np.uint32
            )
            external_ids[write_id] = compute_external_id()

        with tiledb.scope_ctx(ctx_or_config=self.config):
            max_size = MAX_IMAGE_CROPS_PER_IMAGE * len(partition.image_uris)
            images = np.empty(max_size, dtype="O")
            shapes = np.empty(max_size, dtype="O")
            external_ids = np.zeros(max_size, dtype=np.uint64)
            image_uris = np.empty(max_size, dtype="O")
            locations = np.empty(max_size, dtype="O")
            write_id = 0
            image_id = partition.image_id_start
            for image_uri in partition.image_uris:
                image_iter_id = 0
                slide = TileDBOpenSlide(image_uri)
                level_dimensions = slide.level_dimensions[self.level]
                image = slide.read_region((0, 0), self.level, level_dimensions)
                if self.object_crop_shape is None:
                    crop_image(0, level_dimensions[1], 0, level_dimensions[0])
                    write_id += 1
                else:
                    for dim_0_start in range(
                        0, level_dimensions[1], self.object_crop_shape[0]
                    ):
                        for dim_1_start in range(
                            0, level_dimensions[0], self.object_crop_shape[1]
                        ):
                            dim_0_end = min(
                                dim_0_start + self.object_crop_shape[0],
                                level_dimensions[1],
                            )
                            dim_1_end = min(
                                dim_1_start + self.object_crop_shape[1],
                                level_dimensions[0],
                            )
                            crop_image(dim_0_start, dim_0_end, dim_1_start, dim_1_end)
                            write_id += 1
                            image_iter_id += 1
                image_id += 1
            return (
                {
                    "image": images[0:write_id],
                    "shape": shapes[0:write_id],
                    "external_id": external_ids[0:write_id],
                },
                {
                    "image_uri": image_uris[0:write_id],
                    "location": locations[0:write_id],
                    "external_id": external_ids[0:write_id],
                },
            )

    def read_objects_by_external_ids(self, ids: List[int]) -> OrderedDict:
        from tiledb.bioimg.openslide import TileDBOpenSlide

        def crop_image():
            i = 0
            if self.object_crop_shape is None:
                if image_iter_id == i:
                    images[write_id] = image.flatten()
                    shapes[write_id] = np.array(image.shape, dtype=np.uint32)
                    external_ids[write_id] = external_id
                    return
            else:
                for dim_0_start in range(
                    0, level_dimensions[1], self.object_crop_shape[0]
                ):
                    for dim_1_start in range(
                        0, level_dimensions[0], self.object_crop_shape[1]
                    ):
                        if image_iter_id == i:
                            dim_0_end = min(
                                dim_0_start + self.object_crop_shape[0],
                                level_dimensions[1],
                            )
                            dim_1_end = min(
                                dim_1_start + self.object_crop_shape[1],
                                level_dimensions[0],
                            )
                            cropped_image = image[
                                dim_0_start:dim_0_end, dim_1_start:dim_1_end
                            ]
                            images[write_id] = cropped_image.flatten()
                            shapes[write_id] = np.array(
                                cropped_image.shape, dtype=np.uint32
                            )
                            external_ids[write_id] = external_id
                            return
                        i += 1

        with tiledb.scope_ctx(ctx_or_config=self.config):
            size = len(ids)
            images = np.empty(size, dtype="O")
            shapes = np.empty(size, dtype="O")
            external_ids = np.zeros(size, dtype=np.uint64)
            if self.images is None:
                vfs = tiledb.VFS(config=self.config)
                self.images = vfs.ls(self.uri)[1:]

            image_id = -1
            write_id = 0
            for external_id in ids:
                new_image_id = external_id // MAX_IMAGE_CROPS_PER_IMAGE
                image_iter_id = external_id % MAX_IMAGE_CROPS_PER_IMAGE
                if new_image_id != image_id:
                    # Load image
                    image_id = new_image_id
                    slide = TileDBOpenSlide(self.images[image_id])
                    level_dimensions = slide.level_dimensions[self.level]
                    image = slide.read_region((0, 0), self.level, level_dimensions)
                crop_image()
                write_id += 1
            return {"image": images, "shape": shapes, "external_id": external_ids}
