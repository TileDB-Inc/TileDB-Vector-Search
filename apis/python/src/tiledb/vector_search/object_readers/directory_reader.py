from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, OrderedDict, Sequence, Tuple

import numpy as np

import tiledb

# from tiledb.vector_search.object_readers import ObjectPartition, ObjectReader


# class DirectoryPartition(ObjectPartition):
class DirectoryPartition:
    def __init__(
        self,
        partition_id: int,
        paths: List[str],
        object_id_start: int,
    ):
        self.partition_id = partition_id
        self.paths = paths
        self.object_id_start = object_id_start

    def init_kwargs(self) -> Dict:
        return {
            "partition_id": self.partition_id,
            "paths": self.paths,
            "object_id_start": self.object_id_start,
        }

    def id(self) -> int:
        return self.partition_id


# class DirectoryReader(ObjectReader):
class DirectoryReader:
    def __init__(
        self,
        uri: str,
        glob: str = "**/[!.]*",
        exclude: Sequence[str] = (),
        suffixes: Optional[Sequence[str]] = None,
        config: Optional[Mapping[str, Any]] = None,
    ):
        """Initialize with a path to directory and how to glob over it.

        Args:
            uri: Path to directory to load from or path to file to load.
                  If a path to a file is provided, glob/exclude/suffixes are ignored.
            glob: Glob pattern relative to the specified path
                  by default set to pick up all non-hidden files
            exclude: patterns to exclude from results, use glob syntax
            suffixes: Provide to keep only files with these suffixes
                      Useful when wanting to keep files with different suffixes
                      Suffixes must include the dot, e.g. ".txt"
            config: TileDB config
        """
        self.uri = uri
        self.glob = glob
        self.exclude = exclude
        self.suffixes = suffixes
        self.config = config
        self.paths = None

    def partition_class_name(self) -> str:
        return "DirectoryPartition"

    def metadata_array_uri(self) -> str:
        return None

    def list_paths(self, vfs: tiledb.VFS, uri):
        children = vfs.ls(uri)
        uri_path = Path(uri)
        for child in children:
            child_path = Path(child)
            if child_path == uri_path:
                continue
            if self.exclude:
                if any(child_path.match(exclude_glob) for exclude_glob in self.exclude):
                    continue
            if vfs.is_file(child):
                if not child_path.match(self.glob):
                    continue

                if self.suffixes and child_path.suffix not in self.suffixes:
                    continue
                self.paths.append(child)
            else:
                self.list_paths(vfs, child)

    def get_partitions(
        self, files_per_partition: int = -1, **kwargs
    ) -> List[DirectoryPartition]:
        if files_per_partition == -1:
            files_per_partition = 1

        if self.paths is None:
            self.paths = []
            vfs = tiledb.VFS(config=self.config)
            self.list_paths(vfs=vfs, uri=self.uri)
        num_files = len(self.paths)
        partitions = []
        partition_id = 0
        for start in range(0, num_files, files_per_partition):
            end = start + files_per_partition
            if end > num_files:
                end = num_files
            partitions.append(
                DirectoryPartition(
                    partition_id,
                    paths=self.paths[start:end],
                    object_id_start=start,
                )
            )
            partition_id += 1
        return partitions


class DirectoryTextReader(DirectoryReader):
    MAX_OBJECTS_PER_FILE = 10000

    def __init__(
        self,
        uri: str,
        glob: str = "**/[!.]*",
        exclude: Sequence[str] = (),
        suffixes: Optional[Sequence[str]] = None,
        config: Optional[Mapping[str, Any]] = None,
        text_splitter: str = "RecursiveCharacterTextSplitter",
        text_splitter_kwargs: Optional[Dict] = {
            "chunk_size": 1000,
            "chunk_overlap": 100,
        },
    ):
        super().__init__(
            uri=uri,
            glob=glob,
            exclude=exclude,
            suffixes=suffixes,
            config=config,
        )
        self.text_splitter = text_splitter
        self.text_splitter_kwargs = text_splitter_kwargs

    def init_kwargs(self) -> Dict:
        return {
            "uri": self.uri,
            "glob": self.glob,
            "exclude": self.exclude,
            "suffixes": self.suffixes,
            "config": self.config,
            "text_splitter": self.text_splitter,
            "text_splitter_kwargs": self.text_splitter_kwargs,
        }

    def metadata_attributes(self) -> List[tiledb.Attr]:
        text_attr = tiledb.Attr(name="text", dtype=str)
        file_path_attr = tiledb.Attr(name="file_path", dtype=str)
        location_attr = tiledb.Attr(name="page", dtype=np.int32)
        return [text_attr, file_path_attr, location_attr]

    def read_objects(
        self, partition: DirectoryPartition
    ) -> Tuple[OrderedDict, OrderedDict]:
        import os
        import tempfile
        import traceback
        from urllib.parse import urlparse

        max_size = DirectoryTextReader.MAX_OBJECTS_PER_FILE * len(partition.paths)
        texts = np.empty(max_size, dtype="O")
        file_paths = np.empty(max_size, dtype="O")
        pages = np.zeros(max_size, dtype=np.int32)
        external_ids = np.zeros(max_size, dtype=np.uint64)
        write_id = 0
        vfs = tiledb.VFS(config=self.config)
        with tempfile.TemporaryDirectory() as temp_dir:
            for path in partition.paths:
                try:
                    local_temp_path = ""
                    parsed_path = urlparse(path, allow_fragments=False)
                    if parsed_path.scheme != "file":
                        # Copy remote file to a local temp file
                        key = parsed_path.path.lstrip("/")
                        local_temp_path = f"{temp_dir}/{key}"
                        os.makedirs(os.path.dirname(local_temp_path), exist_ok=True)
                        with vfs.open(path, "rb") as src, open(
                            local_temp_path, "wb"
                        ) as dst:
                            dst.write(src.read())
                        parsed_path = urlparse(local_temp_path, allow_fragments=False)
                    if parsed_path.path.endswith(".jpg") or parsed_path.path.endswith(
                        ".png"
                    ):
                        from langchain_community.document_loaders import (
                            UnstructuredFileLoader,
                        )

                        loader = UnstructuredFileLoader(file_path=parsed_path.path)
                    else:
                        from langchain_community.document_loaders.generic import (
                            GenericLoader,
                        )
                        from langchain_community.document_loaders.parsers.generic import (
                            MimeTypeBasedParser,
                        )
                        from langchain_community.document_loaders.parsers.html import (
                            BS4HTMLParser,
                        )
                        from langchain_community.document_loaders.parsers.msword import (
                            MsWordParser,
                        )
                        from langchain_community.document_loaders.parsers.pdf import (
                            PyMuPDFParser,
                        )
                        from langchain_community.document_loaders.parsers.txt import (
                            TextParser,
                        )

                        loader = GenericLoader.from_filesystem(
                            path=parsed_path.path,
                            parser=MimeTypeBasedParser(
                                handlers={
                                    "application/pdf": PyMuPDFParser(),
                                    "text/plain": TextParser(),
                                    "text/html": BS4HTMLParser(),
                                    "application/msword": MsWordParser(),
                                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
                                        MsWordParser()
                                    ),
                                },
                                fallback_parser=None,
                            ),
                        )
                    documents = loader.load()
                    import importlib

                    text_splitters_module = importlib.import_module(
                        "langchain.text_splitter"
                    )
                    splitter_class_ = getattr(text_splitters_module, self.text_splitter)
                    splitter = splitter_class_(**self.text_splitter_kwargs)
                    documents = splitter.split_documents(documents)
                    len(documents)
                    for d in documents:
                        file_paths[write_id] = str(path)
                        if "page" in d.metadata:
                            pages[write_id] = int(d.metadata["page"])
                        texts[write_id] = d.page_content
                        external_ids[write_id] = (
                            partition.object_id_start
                            * DirectoryTextReader.MAX_OBJECTS_PER_FILE
                            + write_id
                        )
                        write_id += 1
                    if local_temp_path != "":
                        os.remove(local_temp_path)
                except Exception:
                    traceback.print_exc()
        return (
            {"text": texts[0:write_id], "external_id": external_ids[0:write_id]},
            {
                "text": texts[0:write_id],
                "file_path": file_paths[0:write_id],
                "page": pages[0:write_id],
                "external_id": external_ids[0:write_id],
            },
        )

    def read_objects_by_external_ids(self, ids: List[int]) -> OrderedDict:
        # TODO implement
        return None


class DirectoryImageReader(DirectoryReader):
    def __init__(
        self,
        uri: str,
        glob: str = "**/[!.]*",
        exclude: Sequence[str] = (),
        suffixes: Optional[Sequence[str]] = None,
        config: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__(
            uri=uri,
            glob=glob,
            exclude=exclude,
            suffixes=suffixes,
            config=config,
        )

    def init_kwargs(self) -> Dict:
        return {
            "uri": self.uri,
            "glob": self.glob,
            "exclude": self.exclude,
            "suffixes": self.suffixes,
            "config": self.config,
        }

    def metadata_attributes(self) -> List[tiledb.Attr]:
        image_shape_attr = tiledb.Attr(name="shape", dtype=np.uint32, var=True)
        file_path_attr = tiledb.Attr(name="file_path", dtype=str)
        return [image_shape_attr, file_path_attr]

    def read_objects(
        self, partition: DirectoryPartition
    ) -> Tuple[OrderedDict, OrderedDict]:
        from PIL import Image

        import tiledb

        size = len(partition.paths)
        images = np.empty(size, dtype="O")
        shapes = np.empty(size, dtype="O")
        file_paths = np.empty(size, dtype="O")
        external_ids = np.zeros(size, dtype=np.uint64)
        write_id = 0
        vfs = tiledb.VFS(config=self.config)
        for path in partition.paths:
            with vfs.open(path) as fp:
                image = np.array(Image.open(fp))[:, :, :3]
                images[write_id] = image.flatten()
                shapes[write_id] = np.array(image.shape, dtype=np.uint32)
                external_ids[write_id] = partition.object_id_start + write_id
                file_paths[write_id] = str(path)
                write_id += 1
        return (
            {
                "image": images[0:write_id],
                "shape": shapes[0:write_id],
                "external_id": external_ids[0:write_id],
            },
            {
                "shape": shapes[0:write_id],
                "file_path": file_paths[0:write_id],
                "external_id": external_ids[0:write_id],
            },
        )

    def read_objects_by_external_ids(self, ids: List[int]) -> OrderedDict:
        from PIL import Image

        import tiledb

        if self.paths is None:
            self.paths = []
            vfs = tiledb.VFS(config=self.config)
            self.list_paths(vfs=vfs, uri=self.uri)
        size = len(ids)
        images = np.empty(size, dtype="O")
        shapes = np.empty(size, dtype="O")
        file_paths = np.empty(size, dtype="O")
        external_ids = np.zeros(size, dtype=np.uint64)
        write_id = 0
        vfs = tiledb.VFS(config=self.config)
        for id in ids:
            path = self.paths[id]
            with vfs.open(path) as fp:
                image = np.array(Image.open(fp))[:, :, :3]
                images[write_id] = image.flatten()
                shapes[write_id] = np.array(image.shape, dtype=np.uint32)
                external_ids[write_id] = id
                file_paths[write_id] = str(path)
                write_id += 1
        return {
            "image": images[0:write_id],
            "shape": shapes[0:write_id],
            "external_id": external_ids[0:write_id],
        }
