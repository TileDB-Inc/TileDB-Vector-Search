from skbuild import setup  # This line replaces 'from setuptools import setup'

setup(
    name="tiledb-vector-search",
    version="0.1",
    description="Vector Search with TileDB",
    author='TileDB',
    license="MIT",
    packages=['tiledb.vector_search'],
    python_requires=">=3.7",
)
