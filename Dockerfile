FROM continuumio/miniconda3:4.10.3

WORKDIR /tmp/

RUN apt-get update && apt-get install -y wget build-essential libarchive-dev
RUN conda config --prepend channels conda-forge

# Install mamba for faster installations
RUN conda install mamba

# RUN mamba install -y -c tiledb tiledb==2.15.3 cmake pybind11 pytest c-compiler cxx-compiler ninja openblas-devel "pip>22"
RUN mamba install -y -c tiledb tiledb==2.16.0 cmake pybind11 pytest c-compiler cxx-compiler ninja openblas-devel "pip>22"

COPY . TileDB-Vector-Search/

RUN . ~/.bashrc && cd TileDB-Vector-Search/apis/python && pip install .
