from skbuild import setup  # This line replaces 'from setuptools import setup'

setup(
    name="tiledb-vector-search",
    description="Vector Search with TileDB",
    author='TileDB',
    license="MIT",
    packages=['tiledb.vector_search'],
    python_requires=">=3.7",
)
