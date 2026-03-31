from __future__ import annotations

from collections import Counter
from pathlib import PurePosixPath

import click

from tiledb.vector_search.cli.eval_commands import eval_cli
from tiledb.vector_search.cli.progress import progress_bar as _progress_bar
from tiledb.vector_search.cli.progress import progress_done as _progress_done

# Match TileDBLoader parsers in directory_reader.py (PDF, HTML, plain text, Word).
SUPPORTED_SUFFIXES: set[str] = {
    ".md",
    ".qmd",
    ".html",
    ".htm",
    ".pdf",
    ".txt",
    ".rst",
    ".doc",
    ".docx",
}

_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

# 7x7 glyphs using █ (full block). Lines are written with formatter.write(), not
# write_text(), so Click does not reflow; monospace fonts show aligned pixels.
_PIXEL_FONT: dict[str, list[str]] = {
    " ": ["       ", "       ", "       ", "       ", "       ", "       ", "       "],
    "A": ["  ███  ", " █   █ ", "█     █", "███████", "█     █", "█     █", "       "],
    "B": ["██████ ", "█     █", "██████ ", "█     █", "█     █", "██████ ", "       "],
    "C": [" █████ ", "█     █", "█      ", "█      ", "█     █", " █████ ", "       "],
    "D": ["██████ ", "█     █", "█     █", "█     █", "█     █", "██████ ", "       "],
    "E": ["███████", "█      ", "██████ ", "█      ", "█      ", "███████", "       "],
    "F": ["███████", "█      ", "██████ ", "█      ", "█      ", "█      ", "       "],
    "G": [" █████ ", "█     █", "█      ", "█  ████", "█     █", " █████ ", "       "],
    "H": ["█     █", "█     █", "███████", "█     █", "█     █", "█     █", "       "],
    "I": ["███████", "   █   ", "   █   ", "   █   ", "   █   ", "███████", "       "],
    "L": ["█      ", "█      ", "█      ", "█      ", "█      ", "███████", "       "],
    "N": ["█     █", "██    █", "█ █   █", "█  █  █", "█   █ █", "█    ██", "       "],
    "O": [" █████ ", "█     █", "█     █", "█     █", "█     █", " █████ ", "       "],
    "R": ["██████ ", "█     █", "██████ ", "█   █  ", "█    █ ", "█     █", "       "],
    "S": [" ██████", "█      ", "██████ ", "      █", "      █", "██████ ", "       "],
    "T": ["███████", "   █   ", "   █   ", "   █   ", "   █   ", "   █   ", "       "],
    "V": ["█     █", "█     █", "█     █", " █   █ ", "  █ █  ", "   █   ", "       "],
}


def _paint_pixel_word(word: str) -> list[str]:
    w = word.upper()
    rows = [""] * 7
    for i, ch in enumerate(w):
        g = _PIXEL_FONT.get(ch, _PIXEL_FONT[" "])
        gap = " " if i < len(w) - 1 else ""
        for r in range(7):
            rows[r] += g[r] + gap
    return rows


def _pixel_banner_block() -> str:
    """Colored block-pixel banner (no literal title lines)."""
    pad = "  "
    lines: list[str] = []

    def add_section(word: str, fg: str) -> None:
        for row in _paint_pixel_word(word):
            lines.append(pad + click.style(row, fg=fg))
        lines.append("")

    add_section("TILEDB", "blue")
    add_section("VECTOR", "cyan")
    add_section("SEARCH", "bright_blue")
    return "\n".join(lines).rstrip() + "\n"


class _VectorSearchGroup(click.Group):
    """Pixel banner above `vs` help (no subcommand or --help)."""

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        # write_text() runs wrap_text() and destroys fixed-width ASCII art — write raw lines.
        indent = " " * formatter.current_indent
        for line in _pixel_banner_block().splitlines():
            formatter.write(f"{indent}{line}\n")
        formatter.write_paragraph()
        super().format_help(ctx, formatter)


def _run_with_spinner(label: str, func: callable) -> None:
    """Run *func* in a background thread while showing an animated spinner."""
    import sys
    import threading
    import time

    error: BaseException | None = None

    def _target() -> None:
        nonlocal error
        try:
            func()
        except BaseException as exc:
            error = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

    start = time.monotonic()
    idx = 0
    try:
        while thread.is_alive():
            elapsed = time.monotonic() - start
            frame = _SPINNER_FRAMES[idx % len(_SPINNER_FRAMES)]
            msg = f"\r  {frame} {label} … {elapsed:.0f}s"
            sys.stderr.write(msg)
            sys.stderr.flush()
            idx += 1
            thread.join(timeout=0.1)
    finally:
        elapsed = time.monotonic() - start
        sys.stderr.write(f"\r  ✓ {label} ({elapsed:.1f}s)          \n")
        sys.stderr.flush()

    if error is not None:
        raise error


def _silence_transformers_logging() -> None:
    import logging
    import os
    import warnings

    warnings.filterwarnings("ignore", module=r"sentence_transformers|transformers|huggingface_hub")
    for name in (
        "sentence_transformers",
        "transformers",
        "huggingface_hub",
        "tokenizers",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["SAFETENSORS_FAST_GPU"] = "0"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    try:
        from tqdm import tqdm
        from functools import partialmethod
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    except ImportError:
        pass


@click.group(cls=_VectorSearchGroup, invoke_without_command=True)
@click.pass_context
def vs(ctx: click.Context) -> None:
    """Vector search commands."""
    _silence_transformers_logging()
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help(), color=ctx.color)


vs.add_command(eval_cli)


@vs.command()
@click.argument("source_dir")
@click.argument("output_uri")
@click.option(
    "--index-type",
    default="FLAT",
    type=click.Choice(["FLAT", "IVF_FLAT", "VAMANA"]),
    show_default=True,
    help="Vector index algorithm.",
)
@click.option(
    "--embedding-model",
    default="all-MiniLM-L6-v2",
    show_default=True,
    help="Sentence-transformers model name or path.",
)
@click.option(
    "--chunk-size",
    default=500,
    show_default=True,
    type=int,
    help="Text chunk size in characters.",
)
@click.option(
    "--chunk-overlap",
    default=50,
    show_default=True,
    type=int,
    help="Overlap between consecutive text chunks.",
)
@click.option(
    "--s3-region",
    default="us-east-1",
    show_default=True,
    help="AWS region for S3-backed indexes.",
)
def build(
    source_dir: str,
    output_uri: str,
    index_type: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    s3_region: str,
) -> None:
    """Build a vector index from a directory of files.

    Recursively discovers files under SOURCE_DIR.  Supported extensions
    (.md, .qmd, .html, .htm, .pdf, .txt, .rst, .doc, .docx) are chunked and
    embedded via langchain parsers; all other extensions are logged and skipped.

    If an index already exists at OUTPUT_URI, only new files (not yet indexed)
    are embedded and added.

    SOURCE_DIR is the path (local or s3://) to the directory of source files.

    OUTPUT_URI is the destination (local path or s3:// prefix) for the index.
    """
    import time
    from typing import Tuple

    import numpy as np

    import tiledb
    from tiledb.cloud.dag import Mode

    from tiledb.vector_search.embeddings.sentence_transformers_embedding import (
        SentenceTransformersEmbedding,
    )
    from tiledb.vector_search.ingestion import ingest
    from tiledb.vector_search.object_api.object_index import (
        ObjectIndex,
        create as create_index,
    )
    from tiledb.vector_search.object_readers.directory_reader import (
        DirectoryTextReader,
        find_uris_vfs,
    )

    config = {"vfs.s3.region": s3_region}

    # -- Discover files -----------------------------------------------------
    click.echo(f"Scanning {source_dir} …")
    all_paths = find_uris_vfs(source_dir)

    supported: list[str] = []
    ignored_exts: Counter[str] = Counter()
    for path in all_paths:
        ext = PurePosixPath(path).suffix.lower()
        if ext in SUPPORTED_SUFFIXES:
            supported.append(path)
        else:
            ignored_exts[ext] += 1

    if not supported:
        raise click.ClickException(
            f"No supported files ({', '.join(sorted(SUPPORTED_SUFFIXES))}) "
            f"found in {source_dir}"
        )

    ext_counts = Counter(PurePosixPath(p).suffix.lower() for p in supported)
    for ext, count in sorted(ext_counts.items()):
        click.echo(f"  {ext}: {count} file(s)")

    if ignored_exts:
        for ext, count in sorted(ignored_exts.items()):
            label = ext if ext else "(no extension)"
            click.secho(f"  Skipping {label}: {count} file(s)", fg="yellow")

    click.echo(f"Total: {len(supported)} file(s) discovered")

    # -- Detect existing index ----------------------------------------------
    incremental = False
    try:
        existing_type = tiledb.object_type(output_uri, ctx=tiledb.Ctx(config))
        if existing_type == "group":
            incremental = True
    except Exception:
        pass

    if incremental:
        click.echo(f"Existing index found at {output_uri} — checking for new files …")
        obj_index = ObjectIndex(
            uri=output_uri,
            config=config,
            load_embedding=False,
            load_metadata_in_memory=True,
        )
        indexed_paths: set[str] = set()
        if (
            obj_index.object_metadata_array_uri is not None
            and hasattr(obj_index, "metadata_df")
            and "file_path" in obj_index.metadata_df.columns
        ):
            indexed_paths = set(obj_index.metadata_df["file_path"].unique())

        new_paths = [p for p in supported if p not in indexed_paths]
        n_skipped = len(supported) - len(new_paths)
        if n_skipped:
            click.echo(f"  Already indexed: {n_skipped} file(s)")
        if not new_paths:
            click.secho("All files already indexed — nothing to do.", fg="green")
            return
        click.echo(f"  New files to add: {len(new_paths)}")
        supported = new_paths
    else:
        obj_index = None

    # -- Reader & embedding model -------------------------------------------
    reader = DirectoryTextReader(
        search_uri=source_dir,
        suffixes=sorted(SUPPORTED_SUFFIXES),
        text_splitter_kwargs={
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
    )
    reader.paths = supported

    click.echo(f"Loading embedding model: {embedding_model} …")
    embedding = SentenceTransformersEmbedding(model_name_or_path=embedding_model)
    click.echo(f"Embedding dimensions: {embedding.dimensions()}")

    # -- Create index if needed ---------------------------------------------
    if not incremental:
        click.echo(f"Creating {index_type} index at {output_uri} …")
        obj_index = create_index(
            uri=output_uri,
            index_type=index_type,
            object_reader=reader,
            embedding=embedding,
            config=config,
        )

    with tiledb.scope_ctx(ctx_or_config=config):
        if not obj_index.embedding_loaded:
            obj_index.embedding.load()
            obj_index.embedding_loaded = True
        vector_type = obj_index.embedding.vector_type()

        partitions = reader.get_partitions()
        n_partitions = len(partitions)

        if incremental:
            # -- Incremental: use update_batch per partition ----------------
            metadata_array_uri = (
                obj_index.object_metadata_array_uri
                if obj_index.materialize_object_metadata
                else None
            )
            index_timestamp = int(time.time() * 1000)
            metadata_array = (
                tiledb.open(metadata_array_uri, "w", timestamp=index_timestamp)
                if metadata_array_uri
                else None
            )

            t0 = time.monotonic()
            total_chunks = 0
            try:
                for i, partition in enumerate(partitions):
                    _progress_bar(i, n_partitions, "Embedding new files", t0)

                    objects, metadata = reader.read_objects(partition)
                    embeddings_out = obj_index.embedding.embed(objects, metadata)
                    if isinstance(embeddings_out, Tuple):
                        external_ids = embeddings_out[1]
                        embeddings_out = embeddings_out[0]
                    else:
                        external_ids = objects["external_id"].astype(np.uint64)

                    total_chunks += embeddings_out.shape[0]

                    vectors = np.empty(embeddings_out.shape[0], dtype="O")
                    for vi in range(embeddings_out.shape[0]):
                        vectors[vi] = embeddings_out[vi].astype(vector_type)
                    obj_index.index.update_batch(
                        vectors=vectors,
                        external_ids=external_ids.astype(np.uint64),
                    )

                    if metadata_array is not None:
                        meta_ext_ids = metadata.pop("external_id", None)
                        metadata_array[meta_ext_ids] = metadata
            finally:
                if metadata_array is not None:
                    metadata_array.close()

            _progress_done(n_partitions, n_partitions, "Embedding new files", t0)
            click.echo(f"  Generated {total_chunks} chunk embedding(s)")

            _run_with_spinner(
                "Consolidating index",
                lambda: obj_index.index.consolidate_updates(mode=Mode.LOCAL),
            )
        else:
            # -- Fresh build: partitioned array + ingest --------------------
            temp_dir_name, embeddings_array_uri = (
                obj_index._create_embeddings_partitioned_array()
            )
            metadata_array_uri = (
                obj_index.object_metadata_array_uri
                if obj_index.materialize_object_metadata
                else None
            )
            index_timestamp = int(time.time() * 1000)

            embeddings_array = tiledb.open(
                embeddings_array_uri, "w", timestamp=index_timestamp
            )
            metadata_array = (
                tiledb.open(metadata_array_uri, "w", timestamp=index_timestamp)
                if metadata_array_uri
                else None
            )

            t0 = time.monotonic()
            total_chunks = 0
            try:
                for i, partition in enumerate(partitions):
                    _progress_bar(i, n_partitions, "Embedding files", t0)

                    objects, metadata = obj_index.object_reader.read_objects(partition)
                    embeddings_out = obj_index.embedding.embed(objects, metadata)
                    if isinstance(embeddings_out, Tuple):
                        external_ids = embeddings_out[1]
                        embeddings_out = embeddings_out[0]
                    else:
                        external_ids = objects["external_id"].astype(np.uint64)

                    total_chunks += embeddings_out.shape[0]
                    pid = partition.id()

                    vec_flat = np.empty(1, dtype="O")
                    vec_flat[0] = embeddings_out.astype(vector_type).flatten()
                    vec_shape = np.empty(1, dtype="O")
                    vec_shape[0] = np.array(embeddings_out.shape, dtype=np.uint32)
                    ext_ids = np.empty(1, dtype="O")
                    ext_ids[0] = external_ids.astype(np.uint64)

                    embeddings_array[pid] = {
                        "vectors": vec_flat,
                        "vectors_shape": vec_shape,
                        "external_ids": ext_ids,
                    }
                    if metadata_array is not None:
                        meta_ext_ids = metadata.pop("external_id", None)
                        metadata_array[meta_ext_ids] = metadata
            finally:
                embeddings_array.close()
                if metadata_array is not None:
                    metadata_array.close()

            _progress_done(n_partitions, n_partitions, "Embedding files", t0)
            click.echo(f"  Generated {total_chunks} chunk embedding(s)")

            _run_with_spinner(
                "Building index",
                lambda: ingest(
                    index_type=obj_index.index_type,
                    index_uri=obj_index.uri,
                    source_uri=embeddings_array_uri,
                    source_type="TILEDB_PARTITIONED_ARRAY",
                    external_ids_uri=embeddings_array_uri,
                    external_ids_type="TILEDB_PARTITIONED_ARRAY",
                    index_timestamp=index_timestamp,
                    storage_version=obj_index.index.storage_version,
                    config=config,
                    mode=Mode.LOCAL,
                ),
            )

            with tiledb.Group(obj_index.uri, "w") as grp:
                grp.remove(temp_dir_name)
            temp_dir_uri = f"{obj_index.uri}/{temp_dir_name}"
            with tiledb.Group(temp_dir_uri, "m") as temp_grp:
                temp_grp.delete(recursive=True)

    mode_label = "Updated" if incremental else "Done"
    click.echo(f"{mode_label}.")
    click.echo(f"  Index URI : {output_uri}")
    click.echo(f"  Index type: {obj_index.index_type}")
    click.echo(f"  New files : {len(supported)}")
    click.echo(f"  New chunks: {total_chunks}")


@vs.command()
@click.argument("index_uri")
@click.argument("query")
@click.option(
    "--topk",
    default=10,
    show_default=True,
    type=int,
    help="Number of results to return.",
)
def search(index_uri: str, query: str, topk: int) -> None:
    """Search a vector index with a natural-language query.

    INDEX_URI is the path (local or s3://) to an existing vector index.

    QUERY is the search text.
    """
    import shutil
    import textwrap
    from collections import OrderedDict

    import numpy as np

    from tiledb.vector_search.object_api.object_index import ObjectIndex

    term_width = shutil.get_terminal_size((80, 24)).columns

    click.echo(f"Opening index at {index_uri} …")
    obj_index = ObjectIndex(uri=index_uri, load_metadata_in_memory=True)

    click.echo(f"Searching for: {click.style(query, bold=True)}")
    click.echo()

    query_objects = OrderedDict({"text": np.array([query])})
    distances, _ids, metadata = obj_index.query(
        query_objects=query_objects,
        k=topk,
        driver_mode=None,
        return_objects=False,
        return_metadata=True,
    )

    n_results = 0
    for i in range(distances.shape[1]):
        dist = float(distances[0, i])
        if dist == 0.0 and i > 0:
            break
        n_results += 1

    if n_results == 0:
        click.secho("No results found.", fg="yellow")
        return

    separator = click.style("─" * term_width, dim=True)

    for i in range(n_results):
        dist = float(distances[0, i])
        file_path = str(metadata["file_path"][0, i]) if metadata and "file_path" in metadata else ""
        text = str(metadata["text"][0, i]) if metadata and "text" in metadata else ""

        text_snippet = " ".join(text.split())
        wrapped = textwrap.fill(
            text_snippet,
            width=min(term_width, 100),
            initial_indent="    ",
            subsequent_indent="    ",
        )

        rank = click.style(f"[{i + 1}]", bold=True)
        score_label = click.style(f"score={dist:.4f}", fg="cyan")
        file_label = click.style(file_path, fg="green") if file_path else ""

        click.echo(separator)
        click.echo(f"  {rank}  {score_label}  {file_label}")
        click.echo(wrapped)

    click.echo(separator)
    click.echo()
    click.echo(f"Returned {click.style(str(n_results), bold=True)} result(s)")
