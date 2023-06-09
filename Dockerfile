FROM continuumio/miniconda3:4.10.3

WORKDIR /tmp/

RUN apt-get update && apt-get install -y wget build-essential libarchive-dev
RUN conda config --prepend channels conda-forge

# Install mamba for faster installations
RUN conda install mamba

RUN mamba install -y -c tiledb tiledb==2.15.3 cmake pybind11 pytest gcc gxx openblas-devel

COPY . feature-vector-prototype/

RUN mkdir build
RUN cd build && cmake -DTILEDB_VS_PYTHON=ON /tmp/feature-vector-prototype/experimental
RUN cd build && make -j4
ENV PYTHONPATH /tmp/build/libtiledbvectorsearch/python/src/tiledb/vector_search
RUN cd feature-vector-prototype/apis/python && pip install -e .
