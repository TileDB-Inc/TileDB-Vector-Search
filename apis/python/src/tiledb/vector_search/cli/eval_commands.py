from __future__ import annotations

import json
from pathlib import Path

import click


def _ensure_eval_deps() -> None:
    try:
        import langchain_anthropic  # noqa: F401
        import langgraph.graph  # noqa: F401
    except ImportError as e:
        raise click.ClickException(
            "Eval commands need optional dependencies. Install with:\n"
            "  pip install 'tiledb-vector-search[eval]'\n"
            "(Use [cli] as well to build indexes from documents.)"
        ) from e


@click.group(name="eval", invoke_without_command=True)
@click.pass_context
def eval_cli(ctx: click.Context) -> None:
    """LangGraph + Anthropic evals and index comparison."""
    _ensure_eval_deps()
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help(), color=ctx.color)


@eval_cli.command("agent")
@click.argument(
    "dataset",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--index-uri",
    default=None,
    help="TileDB vector index URI. With the default flags, runs both with and without retrieval.",
)
@click.option(
    "--no-index-only",
    is_flag=True,
    help="Only run the baseline (no retrieval). Does not require --index-uri.",
)
@click.option(
    "--skip-without-index",
    is_flag=True,
    help="When using --index-uri, do not run the no-retrieval pass.",
)
@click.option(
    "--skip-with-index",
    is_flag=True,
    help="When using --index-uri, do not run the retrieval pass.",
)
@click.option("--top-k", default=10, show_default=True, type=int)
@click.option(
    "--model",
    default="claude-haiku-4-5-20251001",
    show_default=True,
    help="Anthropic model id (langchain-anthropic).",
)
@click.option(
    "--llm-judge",
    is_flag=True,
    help="Also call the model to score each answer vs gold (0–1).",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Write JSON report to this path.",
)
def eval_agent(
    dataset: Path,
    index_uri: str | None,
    no_index_only: bool,
    skip_without_index: bool,
    skip_with_index: bool,
    top_k: int,
    model: str,
    llm_judge: bool,
    out: Path | None,
) -> None:
    """Run eval examples: compare answers with vs without vector retrieval."""
    from tiledb.vector_search.evals.runner import run_agent_eval_report

    if no_index_only:
        run_with = False
        run_without = True
        if index_uri and (skip_with_index or skip_without_index):
            raise click.UsageError(
                "--no-index-only cannot be combined with --index-uri skip flags meaningfully."
            )
    else:
        if not index_uri:
            raise click.UsageError(
                "Provide --index-uri, or use --no-index-only for baseline-only."
            )
        run_with = not skip_with_index
        run_without = not skip_without_index
        if not run_with and not run_without:
            raise click.UsageError(
                "Choose at least one of retrieval or baseline (check skip flags)."
            )

    report = run_agent_eval_report(
        dataset_path=dataset,
        index_uri=index_uri,
        top_k=top_k,
        anthropic_model=model,
        llm_judge=llm_judge,
        run_without_index=run_without,
        run_with_index=run_with,
        show_progress=True,
    )
    text = json.dumps(report, indent=2)
    click.echo(text)
    if out is not None:
        out.write_text(text, encoding="utf-8")
        click.echo(f"Wrote {out}", err=True)


@eval_cli.command("compare-indexes")
@click.argument(
    "dataset",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option("--index-a", required=True, help="First index URI.")
@click.option("--index-b", required=True, help="Second index URI.")
@click.option("--top-k", default=10, show_default=True, type=int)
@click.option(
    "--answer-metrics",
    is_flag=True,
    help="Run the full agent per index and score answers vs gold (calls the API).",
)
@click.option(
    "--model",
    default="claude-haiku-4-5-20251001",
    show_default=True,
)
@click.option("--llm-judge", is_flag=True)
@click.option(
    "--out",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
)
def eval_compare_indexes(
    dataset: Path,
    index_a: str,
    index_b: str,
    top_k: int,
    answer_metrics: bool,
    model: str,
    llm_judge: bool,
    out: Path | None,
) -> None:
    """Compare two indexes: retrieval recall@k (labeled rows) and optional answer metrics."""
    from tiledb.vector_search.evals.runner import run_compare_indexes_report

    report = run_compare_indexes_report(
        dataset_path=dataset,
        index_uri_a=index_a,
        index_uri_b=index_b,
        top_k=top_k,
        anthropic_model=model,
        answer_metrics=answer_metrics,
        llm_judge=llm_judge,
        show_progress=True,
    )
    r = report.get("retrieval") or {}
    if r.get("per_example") == []:
        click.secho(
            "No rows with relevant_file_paths; retrieval section is empty. "
            "Add labels or use eval agent for answer-only runs.",
            fg="yellow",
            err=True,
        )
    text = json.dumps(report, indent=2)
    click.echo(text)
    if out is not None:
        out.write_text(text, encoding="utf-8")
        click.echo(f"Wrote {out}", err=True)
