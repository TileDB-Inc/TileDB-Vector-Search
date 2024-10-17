from typing import Dict, List, Optional, OrderedDict, Sequence, Tuple

import numpy as np

import tiledb
from tiledb.vector_search.object_readers.directory_reader import DirectoryImageReader
from tiledb.vector_search.object_readers.directory_reader import DirectoryPartition

MAX_IMAGE_CROPS_PER_IMAGE = 10000


class BioImageDirectoryReader(DirectoryImageReader):
    def __init__(
        self,
        search_uri: str,
        include: str = "*",
        exclude: Sequence[str] = ["[.]*", "*/[.]*"],
        suffixes: Optional[Sequence[str]] = None,
        max_files: Optional[int] = None,
        level: int = -1,
        object_crop_shape: Tuple[int, int] = None,
        timestamp=None,
    ):
        super().__init__(
            search_uri=search_uri,
            include=include,
            exclude=exclude,
            suffixes=suffixes,
            max_files=max_files,
        )
        self.level = level
        self.object_crop_shape = object_crop_shape
        self.timestamp = timestamp
        self.images = None

    def init_kwargs(self) -> Dict:
        return {
            "search_uri": self.search_uri,
            "include": self.include,
            "exclude": self.exclude,
            "suffixes": self.suffixes,
            "max_files": self.max_files,
            "level": self.level,
            "object_crop_shape": self.object_crop_shape,
            "timestamp": self.timestamp,
        }

    def partition_class_name(self) -> str:
        return "DirectoryPartition"

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

    def read_objects(
        self, partition: DirectoryPartition
    ) -> Tuple[OrderedDict, OrderedDict]:
        from tiledb.bioimg.openslide import TileDBOpenSlide

        def crop_image(path, dim_0_start, dim_0_end, dim_1_start, dim_1_end):
            cropped_image = image[dim_0_start:dim_0_end, dim_1_start:dim_1_end]
            images[write_id] = cropped_image.flatten()
            shapes[write_id] = np.array(cropped_image.shape, dtype=np.uint32)
            image_uris[write_id] = path
            locations[write_id] = np.array(
                [dim_0_start, dim_0_end, dim_1_start, dim_1_end], dtype=np.uint32
            )
            external_ids[write_id] = abs(hash(f"{path}_{dim_0_start}_{dim_1_start}"))

        max_size = MAX_IMAGE_CROPS_PER_IMAGE * len(partition.paths)
        images = np.empty(max_size, dtype="O")
        shapes = np.empty(max_size, dtype="O")
        external_ids = np.zeros(max_size, dtype=np.uint64)
        image_uris = np.empty(max_size, dtype="O")
        locations = np.empty(max_size, dtype="O")
        write_id = 0
        for path in partition.paths:
            slide = TileDBOpenSlide(path)
            level_dimensions = slide.level_dimensions[self.level]
            image = slide.read_region((0, 0), self.level, level_dimensions)
            if self.object_crop_shape is None:
                crop_image(path, 0, level_dimensions[1], 0, level_dimensions[0])
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
                        crop_image(path, dim_0_start, dim_0_end, dim_1_start, dim_1_end)
                        write_id += 1
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
