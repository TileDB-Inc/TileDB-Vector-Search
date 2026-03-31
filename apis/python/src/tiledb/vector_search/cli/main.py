import click

from tiledb.vector_search.cli.vs import vs


@click.group()
@click.version_option(package_name="tiledb-vector-search")
def cli():
    """TileDB command-line interface."""


cli.add_command(vs)
