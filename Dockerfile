#FROM continuumio/miniconda3:4.10.3
FROM ubuntu:22.04

WORKDIR /tmp/

RUN apt-get update && apt-get install -y wget build-essential libarchive-dev
RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
RUN bash Anaconda3-2020.11-Linux-x86_64.sh -b
ENV PATH /root/anaconda3/bin:$PATH

RUN conda config --prepend channels conda-forge

# Install mamba for faster installations
RUN conda install mamba

# This Dockerfile currently works for multiple Python versions. If it becomes
# impossible to maintain one Dockerfile for all the Python versions we support
# server-side, split off the offending version(s) into their own Dockerfile.
ARG PYTHON_VERSION

#RUN mamba install -y python=3.9
RUN mamba install -y -c tiledb tiledb==2.15.3 cmake pybind11 pytest gcc gxx openblas-devel

COPY . feature-vector-prototype/

RUN mkdir build
RUN cd build && cmake -DTILEDB_VS_PYTHON=ON /tmp/feature-vector-prototype/experimental
RUN cd build && make -j4
ENV PYTHONPATH /tmp/build/libtiledbvectorsearch/python/src/tiledb/vector_search
RUN cd feature-vector-prototype/apis/python && pip install .
