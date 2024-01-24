from typing import Any, Mapping, Optional, List, Tuple, Dict, OrderedDict
from tiledb.vector_search.object_readers import ObjectPartition, ObjectReader
from tiledb import Attr


class BioImagePartition(ObjectPartition):
    def __init__(
        self,
        partition_id: str,
        crop_config: List[Tuple[str, Tuple[Tuple[int, int], Tuple[int, int]]]],
        level: int,
        id_start: int,
        id_end: int,
    ):
        self.partition_id = partition_id
        self.crop_config = crop_config
        self.level =level
        self.id_start = id_start
        self.id_end = id_end
        self.size = id_end - id_start

    def init_kwargs(self) -> Dict:
        return {
            "partition_id": self.partition_id,
            "crop_config": self.crop_config,
            "level": self.level,
            "id_start": self.id_start,
            "id_end": self.id_end,
        }

    def num_vectors(self) -> int:
        return self.size

    def index_slice(self) -> Tuple[int,int]:
        return (self.id_start, self.id_end)


class BioImageReader(ObjectReader):
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
        self.crop_config = None

    def create_crop_config(self):
        import tiledb
        import math
        from tiledb.bioimg.openslide import TileDBOpenSlide

        vfs = tiledb.VFS(config=self.config)
        self.images = vfs.ls(self.uri)[1:]
        self.crop_config = []
        with tiledb.scope_ctx(ctx_or_config=self.config):
            for image in self.images:
                slide = TileDBOpenSlide(image)
                level_dimensions = slide.level_dimensions[self.level]
                if self.object_crop_shape is None:
                    self.crop_config.append((image, ((0, 0), self.object_crop_shape)))
                else:
                    for dim_0 in range(0, level_dimensions[1], self.object_crop_shape[0]):
                        for dim_1 in range(0, level_dimensions[0], self.object_crop_shape[1]):
                            self.crop_config.append((image, ((dim_0, dim_1), self.object_crop_shape)))
        self.num_objects = len(self.crop_config)

    def init_kwargs(self) -> Dict:
        return {
            "uri": self.uri,
            "level": self.level,
            "object_crop_shape": self.object_crop_shape,
            "config": self.config,
            "timestamp": self.timestamp,
        }

    def num_vectors(self) -> int:
        if self.crop_config is None:
            self.create_crop_config()
        return self.num_objects

    def partition_class_name(self) -> str:
        return "BioImagePartition"

    def metadata_array_uri(self) -> str:
        return None

    def metadata_attributes(self) -> List[Attr]:
        import numpy as np
        image_uri_attr = Attr(
            name="image_uri",
            dtype=str,
        )
        location_attr = Attr(
            name="location",
            dtype=np.uint32,
            var=True,
        )
        return [image_uri_attr, location_attr]

    def get_partitions(self, partition_size: int = -1) -> List[BioImagePartition]:
        if partition_size == -1:
            partition_size = 1

        if self.crop_config is None:
            self.create_crop_config()

        partitions = []
        partition_id = 0
        for start in range(0, self.num_vectors(), partition_size):
            end = start + partition_size
            if end > self.num_vectors():
                end = self.num_vectors()
            partitions.append(BioImagePartition(str(partition_id), crop_config=self.crop_config[start:end], level=self.level, id_start=start, id_end=end))
            partition_id += 1

        return partitions

    def read_objects(self, partition: BioImagePartition) -> Tuple[OrderedDict, OrderedDict]:
        import tiledb
        import numpy as np
        from tiledb.bioimg.openslide import TileDBOpenSlide


        if self.crop_config is None:
            self.create_crop_config()
        ids_per_image = {}
        i = 0
        for id in range(partition.id_start, partition.id_end):
            image_uri = partition.crop_config[i][0]
            if image_uri in ids_per_image:
                ids_per_image[image_uri].append(id)
            else:
                ids_per_image[image_uri] = [id]
            i += 1

        size = len(partition.crop_config)
        images = np.empty(size, dtype="O")
        shapes = np.empty(size, dtype="O")
        external_ids = np.zeros(size, dtype=np.uint64)
        i = 0
        image_uris = np.empty(size, dtype="O")
        locations = np.empty(size, dtype="O")
        with tiledb.scope_ctx(ctx_or_config=self.config):
            for image_uri, ids in ids_per_image.items():
                slide = TileDBOpenSlide(image_uri)
                level_dimensions = slide.level_dimensions[self.level]
                image = slide.read_region((0, 0), self.level, level_dimensions)
                for id in ids:
                    dim_0_start = self.crop_config[id][1][0][0]
                    dim_0_end = min(dim_0_start + self.object_crop_shape[0], level_dimensions[1])
                    dim_1_start = self.crop_config[id][1][0][1]
                    dim_1_end = min(dim_1_start + self.object_crop_shape[1], level_dimensions[0])
                    cropped_image = image[dim_0_start:dim_0_end, dim_1_start:dim_1_end]
                    images[i] = cropped_image.flatten()
                    shapes[i] = np.array(cropped_image.shape, dtype=np.uint32)
                    external_ids[i] = id
                    image_uris[i] = image_uri
                    locations[i] = np.array([dim_0_start, dim_0_end, dim_1_start, dim_1_end], dtype=np.uint32)
                    i += 1
        
        return {"image": images, "shape": shapes, "external_id": external_ids}, {"image_uri": image_uris, "location": locations, "external_id": external_ids}

    def read_objects_by_external_ids(self, ids: List[int]) -> OrderedDict:
        import tiledb
        import numpy as np
        from tiledb.bioimg.openslide import TileDBOpenSlide

        if self.crop_config is None:
            self.create_crop_config()
        ids_per_image = {}
        for id in ids:
            image_uri = self.crop_config[id][0]
            if image_uri in ids_per_image:
                ids_per_image[image_uri].append(id)
            else:
                ids_per_image[image_uri] = [id]

        size = len(ids)
        images = np.empty(size, dtype="O")
        shapes = np.empty(size, dtype="O")
        external_ids = np.zeros(size, dtype=np.uint64)
        i = 0
        with tiledb.scope_ctx(ctx_or_config=self.config):
            for image_uri, ids in ids_per_image.items():
                slide = TileDBOpenSlide(image_uri)
                level_dimensions = slide.level_dimensions[self.level]
                image = slide.read_region((0, 0), self.level, level_dimensions)
                for id in ids:
                    dim_0_start = self.crop_config[id][1][0][0]
                    dim_0_end = min(dim_0_start + self.object_crop_shape[0], level_dimensions[1])
                    dim_1_start = self.crop_config[id][1][0][1]
                    dim_1_end = min(dim_1_start + self.object_crop_shape[1], level_dimensions[0])
                    cropped_image = image[dim_0_start:dim_0_end, dim_1_start:dim_1_end]
                    images[i] = cropped_image.flatten()
                    shapes[i] = np.array(cropped_image.shape, dtype=np.uint32)
                    external_ids[i] = id
                    i += 1
        return {"image": images, "shape": shapes, "external_id": external_ids}

