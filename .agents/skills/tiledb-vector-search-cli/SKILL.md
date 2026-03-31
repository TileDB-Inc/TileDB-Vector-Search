---
name: tiledb-vector-search-cli
description: >-
  TileDB Vector Search CLI (`tiledb vs`): build indexes from markdown/Quarto/HTML
  directories and run semantic search. Use when the user mentions tiledb vs,
  vector search CLI, building a vector index from docs, or searching an index
  built with this repo.
---

# TileDB Vector Search CLI

## Install

From the TileDB-Vector-Search repo root (native extension must build):

```bash
pip install -e ".[cli]"
```

For **LangGraph + Anthropic evals** (`tiledb vs eval`), `langgraph` / `langchain-anthropic` / `langchain-core` ship with the base package; use **`[eval]`** if you want an explicit `anthropic` dependency. Set `ANTHROPIC_API_KEY`:

```bash
pip install -e ".[cli,eval]"
```

Entry point: `tiledb` (Click). Subcommand group: `vs` (vector search).

Implementation lives under `apis/python/src/tiledb/vector_search/cli/` (`main.py`, `vs.py`, `eval_commands.py`). Eval logic lives under `apis/python/src/tiledb/vector_search/evals/`.

## Commands

### `tiledb vs build SOURCE_DIR OUTPUT_URI`

Recursively indexes docs under `SOURCE_DIR` (local path or `s3://`): `.md`, `.qmd`, `.html`, `.htm`, `.pdf`, `.txt`, `.rst`, `.doc`, `.docx`. Other extensions are skipped (logged in yellow). Parsing uses the same `TileDBLoader` stack as `DirectoryTextReader` (PyMuPDF for PDF, BS4 for HTML, etc.).

**Incremental updates:** If `OUTPUT_URI` already exists as a TileDB group, only files whose paths are **not** already present in index metadata (`file_path`) are embedded and merged. If every file is already indexed, the command exits with nothing to do.

**Caveat:** Updates are keyed by **file URI/path**. Editing a file in place does **not** re-embed it; only new paths are added.

| Option | Default | Notes |
|--------|---------|--------|
| `--index-type` | `FLAT` | `FLAT`, `IVF_FLAT`, `VAMANA` |
| `--embedding-model` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `--chunk-size` | `500` | Characters per chunk |
| `--chunk-overlap` | `50` | Chunk overlap |
| `--s3-region` | `us-east-1` | Passed as TileDB config `vfs.s3.region` |

**Phases (fresh build):** progress bar while embedding per file/partition, then spinner while building the index.

**Phases (incremental):** progress bar for new files, then spinner for `consolidate_updates`.

Examples:

```bash
tiledb vs build ./docs ./out/my_index
tiledb vs build ./docs s3://bucket/prefix/index --s3-region us-west-2
tiledb vs build ./docs ./out/index --index-type IVF_FLAT --embedding-model all-mpnet-base-v2
```

### `tiledb vs search INDEX_URI QUERY`

Semantic search over an existing index (local or `s3://`). Uses the same embedding model stored in the index.

| Option | Default |
|--------|---------|
| `--topk` | `10` |

Quote multi-word queries in the shell:

```bash
tiledb vs search ./out/my_index "how do I configure auth"
tiledb vs search s3://bucket/prefix/index "deployment" --topk 5
```

### `tiledb vs eval agent DATASET.jsonl`

Runs a small **LangGraph** pipeline (retrieve → generate) with **ChatAnthropic**, using your eval JSONL file. Each line is one object with at least `question` and `gold_answer`; optional `id` and `relevant_file_paths` (for recall@k when an index is used).

**With an index (default):** runs **both** passes per example—**with** retrieval from `--index-uri` and **without**—and prints JSON with `with_index` / `without_index` aggregates (exact match rate, mean token F1, optional LLM judge).

**Baseline only:** `--no-index-only` (no `--index-uri`).

| Option | Notes |
|--------|--------|
| `--index-uri` | Vector index URI (required unless `--no-index-only`) |
| `--top-k` | Chunks passed to the model (default 10) |
| `--model` | Anthropic model id (default `claude-3-5-haiku-20241022`) |
| `--llm-judge` | Extra model call to score prediction vs gold (0–1) |
| `--skip-without-index` / `--skip-with-index` | Run only one side when comparing |
| `--out` | Write JSON report to a file |

Example:

```bash
export ANTHROPIC_API_KEY=...
tiledb vs eval agent ./examples/evals/sample.jsonl --index-uri ./out/my_index --out report.json
```

Example datasets live under `examples/evals/`; see `examples/evals/README.md` for **`academy_vector_embeddings_foundation.jsonl`** (build the index from the TileDB-Documentation `academy` directory so `relevant_file_paths` align with metadata).

### `tiledb vs eval compare-indexes DATASET.jsonl`

Compares **two** indexes (e.g. different `--embedding-model` builds) on rows that include `relevant_file_paths`: reports **recall@k** per index (whether any top-`k` result path matches a labeled path; matching uses normalized equality, path suffix overlap, or same parent-dir + filename—not basename alone, so shared names like `index.qmd` are not all treated as the same file). With `--answer-metrics`, also runs the full agent per index and scores answers vs `gold_answer`.

```bash
tiledb vs eval compare-indexes ./eval.jsonl --index-a ./idx_minilm --index-b ./idx_mpnet --top-k 10
tiledb vs eval compare-indexes ./eval.jsonl --index-a ./idx_a --index-b ./idx_b --answer-metrics --llm-judge
```

## Scores in search output

Default index metric is L2-related distance on normalized embeddings: **lower scores are better** (0 = closest). Results are ordered best-first.

## Dependencies and runtime

- **CLI extras:** `sentence-transformers`, `langchain-text-splitters`, `langchain-community`, `beautifulsoup4`, `pymupdf`, `python-docx` (`pyproject.toml` → `[project.optional-dependencies].cli`).
- **Eval stack (core):** `langgraph`, `langchain-anthropic`, and `langchain-core` are **main** dependencies. **`[eval]`** adds an explicit `anthropic` pin for API clients; `pip install .[eval]` is still recommended for eval workflows alongside `[cli]`.
- **S3:** Standard TileDB/AWS credential and endpoint configuration; `--s3-region` sets `vfs.s3.region` only.
- **Sentence-transformers:** Model weights cache under the usual Hugging Face / sentence-transformers cache dirs; no API key for public models.

## Extending file types

Supported suffixes are defined in `SUPPORTED_SUFFIXES` in `apis/python/src/tiledb/vector_search/cli/vs.py`. New types must be parseable by `TileDBLoader` in `object_readers/directory_reader.py` (add a MIME handler there if needed).
