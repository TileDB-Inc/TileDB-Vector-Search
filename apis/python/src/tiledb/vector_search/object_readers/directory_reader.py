from pathlib import Path
from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    OrderedDict,
    Sequence,
    Tuple,
)

import numpy as np

import tiledb

# from tiledb.vector_search.object_readers import ObjectPartition, ObjectReader


# Util functions for file matching and recursive listing
def match_uri(
    search_uri: str,
    file_uri: str,
    include: str = "*",
    exclude: Sequence[str] = ["[.]*", "*/[.]*"],
    suffixes: Optional[Sequence[str]] = None,
) -> bool:
    search_uri = search_uri.rstrip("/") + "/"
    file_split = file_uri.split(search_uri)
    file_name = "/" if len(file_split) < 2 else file_split[1]
    file_path = Path(file_name)
    if not file_path.match(include):
        return False
    if any(file_path.match(e) for e in exclude):
        return False
    if suffixes:
        if file_path.suffix not in suffixes:
            return False
        else:
            return True
    return True


def find_uris_vfs(
    search_uri: str,
    include: str = "*",
    exclude: Sequence[str] = ["[.]*", "*/[.]*"],
    suffixes: Optional[Sequence[str]] = None,
    max_files: Optional[int] = None,
) -> Sequence[str]:
    vfs = tiledb.VFS()

    def find(
        uri: str,
        *,
        include: str = "*",
        exclude: Sequence[str] = (),
        suffixes: Optional[Sequence[str]] = None,
        max_files: Optional[int] = None,
    ):
        # Stop if we have found max_count files
        if max_files is not None and len(results) >= max_files:
            return
        listing = vfs.ls(uri)
        for file_uri in listing:
            # Avoid infinite recursion
            if file_uri == uri:
                continue
            if max_files is not None and len(results) >= max_files:
                return

            if vfs.is_dir(file_uri):
                find(
                    file_uri,
                    include=include,
                    exclude=exclude,
                    suffixes=suffixes,
                    max_files=max_files,
                )

            else:
                match = match_uri(
                    search_uri=search_uri,
                    file_uri=file_uri,
                    include=include,
                    exclude=exclude,
                    suffixes=suffixes,
                )
                if match:
                    results.append(file_uri)

    # Add one trailing slash to search_uri
    search_uri = search_uri.rstrip("/") + "/"
    results = []
    find(
        search_uri,
        include=include,
        exclude=exclude,
        suffixes=suffixes,
        max_files=max_files,
    )
    if max_files is not None:
        return results[:max_files]
    else:
        return results


def find_uris_tiledb_group(
    search_uri: str,
    include: str = "*",
    exclude: Sequence[str] = ["[.]*", "*/[.]*"],
    suffixes: Optional[Sequence[str]] = None,
    max_files: Optional[int] = None,
    verbose: bool = False,
) -> Sequence[str]:
    def find(
        uri: str,
        path: str,
        include: str = "*",
        exclude: Sequence[str] = (),
        suffixes: Optional[Sequence[str]] = None,
        max_files: Optional[int] = None,
    ):
        # Stop if we have found max_count files
        if max_files is not None and len(results) >= max_files:
            return
        obj_type = tiledb.object_type(uri)
        if obj_type == "array":
            match = match_uri(
                search_uri="/",
                file_uri=path,
                include=include,
                exclude=exclude,
                suffixes=suffixes,
            )
            if match:
                results.append(uri)
        elif obj_type == "group":
            with tiledb.Group(uri) as grp:
                for child in list(grp):
                    find(
                        uri=child.uri,
                        path=f"{path}/{child.name}",
                        include=include,
                        exclude=exclude,
                        suffixes=suffixes,
                        max_files=max_files,
                    )

    results = []
    find(
        uri=search_uri,
        path="",
        include=include,
        exclude=exclude,
        suffixes=suffixes,
        max_files=max_files,
    )
    if max_files is not None:
        return results[:max_files]
    else:
        return results


def find_uris_aws(
    search_uri: str,
    include: str = "*",
    exclude: Sequence[str] = ["[.]*", "*/[.]*"],
    suffixes: Optional[Sequence[str]] = None,
    max_files: Optional[int] = None,
    verbose: bool = False,
) -> Sequence[str]:
    from urllib.parse import urlparse

    try:
        import boto3
    except ImportError:
        raise ImportError(
            "Could not import boto3 python package. "
            "Please install it with `pip install boto3`."
        )

    search_uri = search_uri.rstrip("/") + "/"
    parsed_path = urlparse(search_uri, allow_fragments=False)
    config = tiledb.default_ctx().config().dict()
    s3_kwargs = {}
    if (
        "vfs.s3.aws_access_key_id" in config
        and config["vfs.s3.aws_access_key_id"] != ""
    ):
        s3_kwargs["aws_access_key_id"] = config["vfs.s3.aws_access_key_id"]
    if (
        "vfs.s3.aws_secret_access_key" in config
        and config["vfs.s3.aws_secret_access_key"] != ""
    ):
        s3_kwargs["aws_secret_access_key"] = config["vfs.s3.aws_secret_access_key"]
    if (
        "vfs.s3.aws_session_token" in config
        and config["vfs.s3.aws_session_token"] != ""
    ):
        s3_kwargs["aws_session_token"] = config["vfs.s3.aws_session_token"]
    if "vfs.s3.aws_role_arn" in config and config["vfs.s3.aws_role_arn"] != "":
        s3_kwargs["aws_role_arn"] = config["vfs.s3.aws_role_arn"]
    if "vfs.s3.aws_external_id" in config and config["vfs.s3.aws_external_id"] != "":
        s3_kwargs["aws_external_id"] = config["vfs.s3.aws_external_id"]
    if "vfs.s3.aws_session_name" in config and config["vfs.s3.aws_session_name"] != "":
        s3_kwargs["aws_session_name"] = config["vfs.s3.aws_session_name"]
    if "vfs.s3.region" in config and config["vfs.s3.region"] != "":
        s3_kwargs["region_name"] = config["vfs.s3.region"]
    s3 = boto3.resource("s3", **s3_kwargs)
    bucket = s3.Bucket(parsed_path.netloc)
    prefix = parsed_path.path.lstrip("/")
    results = []
    for obj in bucket.objects.filter(Prefix=prefix):
        # Skip directories
        if obj.size == 0 and obj.key.endswith("/"):
            continue
        match = match_uri(
            search_uri=prefix,
            file_uri=obj.key,
            include=include,
            exclude=exclude,
            suffixes=suffixes,
        )
        if match:
            file_name = obj.key.split(prefix)[1]
            file_uri = f"{search_uri}{file_name}"
            results.append(file_uri)
            if max_files is not None and len(results) >= max_files:
                return results
    return results


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
        search_uri: str,
        include: str = "*",
        exclude: Sequence[str] = ["[.]*", "*/[.]*"],
        suffixes: Optional[Sequence[str]] = None,
        max_files: Optional[int] = None,
    ):
        """Initialize with a path to directory and how to glob over it.

        Args:
            search_uri: Path of directory to load files from.
            include: File pattern to iclude relative to `search_uri`. By default set
                to include all files.
            exclude: File patterns to exclude relative to `search_uri`. By default set
                to ignore all hidden files.
            suffixes: Provide to keep only files with these suffixes
                Useful when wanting to keep files with different suffixes
                Suffixes must include the dot, e.g. ".txt"
            max_files: Maximum number of files to include.
        """
        self.search_uri = search_uri
        self.include = include
        self.exclude = exclude
        self.suffixes = suffixes
        self.max_files = max_files
        self.paths = None

    def partition_class_name(self) -> str:
        return "DirectoryPartition"

    def metadata_array_uri(self) -> str:
        return None

    def list_paths(self):
        obj_type = tiledb.object_type(self.search_uri)
        if obj_type == "group":
            self.paths = find_uris_tiledb_group(
                search_uri=self.search_uri,
                include=self.include,
                exclude=self.exclude,
                suffixes=self.suffixes,
                max_files=self.max_files,
            )
        elif obj_type == "array":
            self.paths = [self.search_uri]
        else:
            self.paths = find_uris_vfs(
                search_uri=self.search_uri,
                include=self.include,
                exclude=self.exclude,
                suffixes=self.suffixes,
                max_files=self.max_files,
            )

    def get_partitions(
        self, files_per_partition: int = -1, **kwargs
    ) -> List[DirectoryPartition]:
        if files_per_partition == -1:
            files_per_partition = 1

        if self.paths is None:
            self.paths = []
            self.list_paths()
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


class TileDBLoader:
    def __init__(
        self,
        uri: str,
    ) -> None:
        from langchain_community.document_loaders.parsers.generic import (
            MimeTypeBasedParser,
        )
        from langchain_community.document_loaders.parsers.html import BS4HTMLParser
        from langchain_community.document_loaders.parsers.msword import MsWordParser
        from langchain_community.document_loaders.parsers.pdf import PyMuPDFParser
        from langchain_community.document_loaders.parsers.txt import TextParser

        self.uri = uri
        self.parser = MimeTypeBasedParser(
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
        )

    def load(self) -> List:
        """Load data into Document objects."""
        return list(self.lazy_load())

    def lazy_load(
        self,
    ) -> Iterator:
        """A lazy loader for Documents."""
        import mimetypes

        from langchain_community.document_loaders.blob_loaders import Blob

        vfs = tiledb.VFS()
        if tiledb.array_exists(self.uri):
            with tiledb.open(self.uri, "r") as a:
                mime_type = a.meta.get("mime_type", None)
            f = tiledb.filestore.Filestore(self.uri)
        else:
            mime_type = mimetypes.guess_type(self.uri)[0]
            f = vfs.open(self.uri)

        if mime_type.startswith("image/"):
            from langchain_community.document_loaders import UnstructuredFileIOLoader

            unstructured_loader = UnstructuredFileIOLoader(f)
            yield from unstructured_loader.load()

        else:
            blob = Blob.from_data(data=f.read(), mime_type=mime_type)
            yield from self.parser.parse(blob)

    def load_and_split(self, text_splitter) -> List:
        """Load all documents and split them into sentences."""
        return text_splitter.split_documents(self.load())


class DirectoryTextReader(DirectoryReader):
    MAX_OBJECTS_PER_FILE = 10000

    def __init__(
        self,
        search_uri: str,
        include: str = "*",
        exclude: Sequence[str] = ["[.]*", "*/[.]*"],
        suffixes: Optional[Sequence[str]] = None,
        max_files: Optional[int] = None,
        text_splitter: str = "RecursiveCharacterTextSplitter",
        text_splitter_kwargs: Optional[Dict] = {
            "chunk_size": 1000,
            "chunk_overlap": 100,
        },
    ):
        super().__init__(
            search_uri=search_uri,
            include=include,
            exclude=exclude,
            suffixes=suffixes,
            max_files=max_files,
        )
        self.text_splitter = text_splitter
        self.text_splitter_kwargs = text_splitter_kwargs

    def init_kwargs(self) -> Dict:
        return {
            "search_uri": self.search_uri,
            "include": self.include,
            "exclude": self.exclude,
            "suffixes": self.suffixes,
            "max_files": self.max_files,
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
        import importlib
        import traceback

        max_size = DirectoryTextReader.MAX_OBJECTS_PER_FILE * len(partition.paths)
        texts = np.empty(max_size, dtype="O")
        file_paths = np.empty(max_size, dtype="O")
        pages = np.zeros(max_size, dtype=np.int32)
        external_ids = np.zeros(max_size, dtype=np.uint64)
        write_id = 0
        text_splitters_module = importlib.import_module("langchain.text_splitter")
        text_splitter_class_ = getattr(text_splitters_module, self.text_splitter)
        text_splitter = text_splitter_class_(**self.text_splitter_kwargs)
        for uri in partition.paths:
            try:
                loader = TileDBLoader(uri=uri)
                documents = loader.load_and_split(text_splitter=text_splitter)
                text_chunk_id = 0
                for d in documents:
                    file_paths[write_id] = str(uri)
                    if "page" in d.metadata:
                        pages[write_id] = int(d.metadata["page"])
                    texts[write_id] = d.page_content
                    external_ids[write_id] = abs(hash(f"{uri}_{text_chunk_id}"))
                    write_id += 1
                    text_chunk_id += 1
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
        search_uri: str,
        include: str = "*",
        exclude: Sequence[str] = ["[.]*", "*/[.]*"],
        suffixes: Optional[Sequence[str]] = None,
        max_files: Optional[int] = None,
    ):
        super().__init__(
            search_uri=search_uri,
            include=include,
            exclude=exclude,
            suffixes=suffixes,
            max_files=max_files,
        )

    def init_kwargs(self) -> Dict:
        return {
            "search_uri": self.search_uri,
            "include": self.include,
            "exclude": self.exclude,
            "suffixes": self.suffixes,
            "max_files": self.max_files,
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
        vfs = tiledb.VFS()
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
            self.list_paths()
        size = len(ids)
        images = np.empty(size, dtype="O")
        shapes = np.empty(size, dtype="O")
        file_paths = np.empty(size, dtype="O")
        external_ids = np.zeros(size, dtype=np.uint64)
        write_id = 0
        vfs = tiledb.VFS()
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
