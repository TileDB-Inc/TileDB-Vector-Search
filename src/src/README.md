# Command Line Drivers for the TileDB-Vector-Search Library

This subdirectory contains command-line driver programs for accessing functionality of the TileDB-Vector-Search library: `ivf_flat`, `flat_l2`, and (WIP) `index`.
Much of their functionality is for evaluating performance of different algorithmic approaches within the library.
A wealth of internal performance information can be dumped from the programs.

## Building

```bash
  cd < project root >
  mkdir build
  cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DTileDB_DIR=/path/to/installed/libtiledb
```

**Note:** The CLI programs build against `libtiledb`. You will need to point your the `TileDB_DIR` to the
directory where `libtiledb` is installed (along with associated headers, etc) .

That is, if the `libtiledb` you want to use is in `/usr/local/tiledb/lib/libtiledb.so` then you would set `TileDB_DIR` to `/usr/local/tiledb/lib/cmake`

```
  % cmake .. -DTileDB_DIR=/usr/local/tiledb/
```

**If you don't specify a value for `TileDB_DIR`** the system default will be used (an installation of TileDB done with a package manager such as `apt` or `homebrew`). That is, you do not have to specify a value for `TileDB_DIR` if the system defaults are good enough.

The CLI do require a fairly recent version of `libtiledb`. If you get compilation errors along the lines of

```
In file included from /home/user/tiledb-vector-search/src/test/unit_sift_array.cpp:5:
/home/user/tiledb-vector-search/src/test/../sift_array.h:67:21: error: expected ';' after expression
    tiledb::Subarray subarray(ctx_, array_);
                    ^
                    ;
/home/user/tiledb-vector-search/src/test/../sift_array.h:67:13: error: no member named 'Subarray' in namespace 'tiledb'
    tiledb::Subarray subarray(ctx_, array_);
    ~~~~~~~~^
/home/user/tiledb-vector-search/src/test/../sift_array.h:68:5: error: use of undeclared identifier 'subarray'
    subarray.set_subarray(subarray_vals);
```

then you likely need a more recent version of `libtiledb`. To fix this, first try updating your installed version of `libtiledb` by invoking the appropriate "upgrade" or "update" command associated with your package manager (if you installed `libtiledb` using a package manager). Otherwise, obtain an up-to-date version of `libtiledb` from the TileDB GitHub repository at `https://github.com/TileDB-Inc/TileDB` and build and install it per the instructions there.

After installing TileDB, your setting for
`TileDB_DIR` should be the same as the value of `CMAKE_INSTALL_PREFIX` that was used when building and installing `libtiledb`.
That is, if you built `libtiledb` with

```
  % cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/tiledb
```

Then if you set `TileDB_DIR` to `/usr/local/tiledb`

```
  % cmake .. -DTileDB_DIR=/usr/local/tiledb
```

If you did not explicitly set `CMAKE_INSTALL_PREFIX`, you should be able to find its value in `CMakeCache.txt`.
(And, if you used the default setting, `cmake` for TileDB-Vector-Search may find it automatically.)
Also, the `fetch_content` call in `CMakeLists.txt` may also find the TileDB installation.
To check if the right path will be searched, look for the output line

```
-- TileDB include directories are <>
```

This will tell you the path that will be used when building `flat`. If it isn't the path you are expecting, e.g., if it is the system defaults when you expected something else, check the path you used when invoking `cmake`.

**Note:** If you build `libtiledb` yourself and are going to use S3 as a source for TileDB array input, your `libtiledb`
should be built with S3 support.

#### Building with MKL BLAS (Linux)

If you have MKL BLAS available on the machine where you are building and running the CLI programs, you should configure them to use MKL BLAS.

```bash
  cmake .. -DUSE_MKL_CBLAS=on
```

If you do not have MKL BLAS available, `cmake` will default to Open BLAS. In either case, for the moment, you must have at least one of Open BLAS or MKL BLAS installed.

If you use the `apt` package manager you should be able to install MKL with

```
  % sudo apt install intel-oneapi-mkl
```

You can also use

```
apt-cache pkgnames intel | grep intel-oneapi | grep -v intel-oneapi-runtime | fgrep mkl
```

to see other available packages variations.

#### Building with Open BLAS (Linux)

To install Open BLAS, you can install one of a number of different options. I am not sure which version offers best performance, so you may want to experiment with the `openmp` vs `pthreads` version. Depending on the compiler used, one or the other may be used. The Intel compilers generally provide the best `openmp` support (installing and using those is beyond the scope of this README however.)

```
  % apt install libopenblas64-dev
```

#### Building with MacOS

If you are using MacOS, the `Accelerate` framework will automatically be selected. You should not have to do anything yourself to use optimized BLAS with MacOS.

## `ivf_flat`: Inverted File Index with Stored Vectors

### Running `ivf_flat` with TileDB Arrays

`ivf_flat` performs a query for given queries against a specified dataset, using an inverted file index derived from applying kmeans partitioning on that dataset. It returns an array of the top `k` nearest neighbors for each of the given query vectors.
The program can also create extensive logging information reporting the accuracy of the search along with numerous other aspects of the program configuration and its execution. The configuration information is intended to enable reproducibility of the results while the execution information is intended to provide coarse-grained profiling information for performance evaluation and performance tuning.

`ivf_flat` currently uses Euclidean distance as its similarity measure. Additional measures (such as cosine similarity and Jaccard similarity) will be added in near-term future releases.

The full set of usage options for `ivf_flat` are the following:

```txt
ivf_flat: demo CLI program for performing feature vector search with kmeans index.
Usage:
    ivf_flat (-h | --help)
    ivf_flat centroids_uri URI parts_uri URI (index_uri URI | --sizes_uri URI)
             ids_uri URI query_uri URI [groundtruth_uri URI] [--output_uri URI]
            [--k NN][--nprobe NN] [--nqueries NN] [--alg ALGO] [--infinite] [--finite] [--blocksize NN]
            [--nthreads NN] [--ppt NN] [--vpt NN] [--nodes NN] [--region REGION] [--stats] [--log FILE] [-d] [-v]

Options:
    -h, --help            show this screen
    centroids_uri URI   URI with centroid vectors
    index_uri URI       URI with the paritioning index
    --sizes_uri URI       URI with the parition sizes
    parts_uri URI       URI with the partitioned data
    ids_uri URI         URI with original IDs of vectors
    query_uri URI       URI storing query vectors
    groundtruth_uri URI URI storing ground truth vectors
    --output_uri URI      URI to store search results
    --k NN                number of nearest neighbors to return for each query vector [default: 10]
    --nprobe NN           number of centroid partitions to use [default: 100]
    --nqueries NN         number of query vectors to use (0 = all) [default: 0]
    --alg ALGO            which algorithm to use for query [default: qv_heap]
    --infinite            Load the entire array into RAM for the search [default: false]
    --finite              For backward compatibility, load only required partitions into memory [default: true]
    --blocksize NN        number of vectors to process in an out of core block (0 = all) [default: 0]
    --nthreads NN         number of threads to use (0 = hardware concurrency) [default: 0]
    --ppt NN              minimum number of partitions to assign to a thread (0 = no min) [default: 0]
    --vpt NN              minimum number of vectors to assign to a thread (0 = no min) [default: 0]
    --nodes NN            number of nodes to use for (emulated) distributed query [default: 1]
    --region REGION       AWS S3 region [default: us-east-1]
    --log FILE            log info to FILE (- for stdout)
    --stats               log TileDB stats [default: false]
    -d, --debug           run in debug mode [default: false]
    -v, --verbose         run in verbose mode [default: false]
```

#### Specifying the query

`ivf_flat` operates on an inverted file index, generated by, say, a `kmeans` algorithm from an original set of vectors.
The inverted file index consists of data stored in multiple TileDB arrays, which can be located locally or in the cloud. (Any valid URI can be used to specify a TileDB array.) The arrays required for a query (and their corresponding option flags) are:

- An array of centroids (`centroids_uri`), which is a collection of `k` vectors, selected by, e.g., the `kmeans` algorithm. Each vector is the centroid of a partition of the original set of vectors (or conversely, each partition is the set of vectors closest to the corresponding centroid.)
- A vector database reordered into partitions, with one partition per centroid, such that each partition comprises the vectors closest to the corresponding centroid (`parts_uri`).
- An array containing partition indexes for the reordered vector database (`index_uri`), such that `index[i]` is the beginning of partition`i`, and `index[i+1]` is the end of partition `i`. (Note that the size of the index array is one more than the number of partitions; the last element of `index` is the total number of vectors in the database.)
- OR an array containing the sizes of each partition (`--sizes_uri`).
- An array of the ids of vectors in the original set of vectors (prior to partitioning) (`ids_uri`), and
- An array containing query vectors (`--query_id`).

The user can also optionally specify

- An array containing ground truth vectors (`groundtruth_uri`), i.e., the nearest-neighbors that would be returned from an exact (`flat L2)` search and/or
- An array for saving the results of the query. (`--output_uri`).

Example

```txt
  ivf_flat                                                                                    \
    centroids_uri s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/centroids.tdb  \
    parts_uri s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/parts.tdb          \
    index_uri s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/index.tdb          \
    ids_uri s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/ids.tdb              \
    query_uri s3://tiledb-andrew/kmeans/benchmark/query_public_10k                          \
    groundtruth_uri s3://tiledb-andrew/kmeans/benchmark/bigann_1B_GT_nnids                  \
    --output_uri file://vector_search/results/output.tdb
```

#### Search Options

The user can also specify a number of options related to the search.

- The number of nearest neighbors to return for each given query vector (`--k`)
- The number of centroids (and associated partitions) to use in performing the query (`--nprobe`). The centroids chosen will be the centroids closest to the given set of query vectors.
- The number of queries from the given query array to search for (`--nqueries`). The first `nqueries` vectors from the query array will be used if the specified value for `nqueries` is less than the total number of vectors in the query array.
  The default is to use all the queries in the query array, which can also be specified with the option value of `0`.
- Which search algorithm in the C++ library to use for performing the search (`--algo`). It is recommended to use the default (other algorithms are currently WIP).
- Whether to load the entire partitioned array into memory when performing the search or (if the `--infinite` option is given) whether to load only the necesary partitions, given the specified query. It is recommended to generally use the default value except in the case of large values of `nqueries` and `nprobe` and the availability of sufficient RAM to hold the entire partitioned array. (For backward compatibility, there is also a `--finite` flag which had the complementary behavior to `--infinite`). If `--blocksize` is specified with the finite-memory option, `ivf_flat` also operate in out-of-core fashion, loading subsets of partitions into memory, in the order they appear in the partitioned vector array.
- An upper bound to the number of vectors to be loaded during each batch when using the finite-memory case. `ivf_flat` will load complete partitions on each out-of-core iteration, so the number of vectors loaded will generally be fewer than the specified upper bound. Similarly, the specified upper bound must be larger than the largest partition in the partitioned array. Out of core operation is necessary if available RAM cannot hold all the index data (in general due to the size of the vector data to be searched). Even if available memory can accommodate the entire partitioned array, out of core operation can be useful for making more efficient use of hierarchical memory.
- The AWS region to use when accessing TileDB arrays stored in S3 (`--region`). The example array URIs provided with TileDB-Vector-Search are located in the `us-east-1` region, which is the default value.
- The name of a file to write logging information to (`--log`). The default is nil, meaning no logs will be written. If the value `-` is specified, the output will be written to `std::cout`.
- Whether to run in debug mode (`-d` or `--debug`). This will print copious information that is useful only to the library developers. End users should always use the default.
- Verbose output, which prints a mild amount of diagnostic information (`--verbose`).

Example:

```txt
  ivf_flat                                                                                    \
    centroids_uri s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/centroids.tdb  \
    parts_uri s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/parts.tdb          \
    index_uri s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/index.tdb          \
    --sizes_uri s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/index_size.tdb     \
    ids_uri s3://tiledb-nikos/vector-search/andrew/sift-base-1b-10000p/ids.tdb              \
    query_uri s3://tiledb-andrew/kmeans/benchmark/query_public_10k                          \
    groundtruth_uri s3://tiledb-andrew/kmeans/benchmark/bigann_1B_GT_nnids                  \
    --output_uri file://vector_search/results/output.tdb                                    \
    --blocksize 1000000 --nqueries 1000 --nprobe 128 --log -v
```

Since there are a large number of options, particular the long set of of array URIs, it is recommended that
you use the setup scripts in the `src/benchmarks` subdirectory. The setup script defines bash functions that
automatically set the various options (notably the URIs) for `ivf_flat` and invoke the executable. There is
some customization you may need to do to make the scripts work in your local environment. The README and comments
in the scripts are intended to help you do that.

See the section [Benchmarking](#benchmarking-with-provided-examples) below on generating your own benchmark results.

### Recommendations

If you use the CLI programs for your own queries, you will need to set the URIs to the arrays containing your own data.
For the other options,
many of the defaults are reasonable choices for attaining good performance in most use cases. For your own queries, the main
options that you might want to change are `--k`, `--nprobe`, `--nqueries`, and `--blocksize`.

## `index` (WIP)

The `index` driver creates an inverted file index, given an input array of vectors to be indexed (`inputs_uri`).
If the `--kmeans` flag is specified, `index` will generate a centroids array using the kmeans algorithm. If the
`--centroids` option is specified, `index` will use the specified centroids array, which may be generated by another program such as Faiss or scikit-learn. The `--kmeans` option and `--centroids` option are mutually exclusive.

The options used by `index` are

- The name of the database to be indexed (`inputs_uri`)
- The name of the centroids array to be written, if `--kmeans` is specified (`centroids_uri`)
- The name of the centroids array to be used for indexing if `--kmeans` is not specified (`--centroids`)
- The name of the array of vectors specified by `inputs_uri`, partitioned according to the generated (or provided) centroids
- The name of the index array to be written (`index_uri`)
- The name of a partition sizes array to be written (`--sizes_uri`). The `--sizes_uri` option and the `index_uri` options are mutually exclusive.
- The name of the ids array to be written (`ids_uri`)
- The initialization algorithm to be used by kmeans (`--init`). Current options are `kmeanspp` and `random`. The default is `random`.

Example:

```
  ivf_index  --kmeans                                                       \
             inputs_uri s3://tiledb-lums/sift/sift_base                       \
             ids_uri s3://tiledb-lums/kmeans/ivf_flat/ids                  \
             index_uri s3://tiledb-lums/kmeans/ivf_flat/index             \
             --part_uri s3://tiledb-lums/kmeans/ivf_flat/parts              \
             centroids_uri s3://tiledb-lums/kmeans/ivf_flat/centroids

```

## The `flat_l2` Search Driver

The `flat_l2` program performs an
exhaustive vector-by-vector comparison between
a given set of query vectors and a given set of vectors.

### Running flat

The program currently
performs search based on L2 similarity. Future releases will include
other similarity measures such as
cosine similarity and Jaccard similarity. If a groundtruth URI is supplied, the
program will check its results against the given set of ground truth vectors.

### Usage

```
  Usage:
      flat_l2 (-h | --help)
      flat_l2 inputs_uri URI query_uri URI [groundtruth_uri URI] [--output_uri URI]
          [--k NN] [--nqueries NN] [--alg ALGO] [--finite] [--blocksize NN]
          [--nthreads N] [--region REGION] [--log FILE] [--stats] [-d] [-v]

  Options:
      -h, --help              show this screen
      inputs_uri URI            database URI with feature vectors
      query_uri URI         query URI with feature vectors to search for
      groundtruth_uri URI   ground truth URI
      --output_uri URI        output URI for results
      --k NN                  number of nearest neighbors to find [default: 10]
      --nqueries NN           size of queries subset to compare (0 = all) [default: 0]
      --alg ALGO              which algorithm to use for comparisons [default: vq_heap]
      --finite                use finite RAM (out of core) algorithm [default: false]
      --blocksize NN          number of vectors to process in an out of core block (0 = all) [default: 0]
      --nthreads N            number of threads to use in parallel loops (0 = all) [default: 0]
      --region REGION         AWS region [default: us-east-1]
      --log FILE              log info to FILE (- for stdout)
      --stats                 log TileDB stats [default: false]
      -d, --debug             run in debug mode [default: false]
      -v, --verbose           run in verbose mode [default: false]
```

The `flat_l2` driver uses a subset of the options used by `ivf_flat` (see above) but instead of `parts_uri`, the
input array of (unpartitioned) vectors is given by `inputs_uri`. In addition,
for experimenting with different approaches to vector similarity search, `flat_l2` has a larger set of options
for `--algo`.

- `vq_nth`
- `vq_heap`
- `qv_nth`
- `qv_heap`
- `gemm`

The algorithm name specifies the ordering of the loops over queries and vectors (`qv` has queries on the outer loop and vectors on the inner loop, `vq` is the opposite). The `nth` algorithms take the `nth` flag to switch between `nth_element` ranking or heap-based ranking. The `heap` option specifies to use an algorithm specialized for heap-based ranking. Finally, the `gemm` option specifies to use a linear algebra based approach for computing scores.

Example:

```txt
  flat_l2                                                                                     \
    inputs_uri s3://tiledb-nikos/vector-search/datasets/arrays/sift-1b-col-major                \
    query_uri s3://tiledb-andrew/kmeans/benchmark/query_public_10k                          \
    groundtruth_uri s3://tiledb-andrew/kmeans/benchmark/bigann_1B_GT_nnids                  \
    --output_uri file://vector_search/results/output.tdb                                    \
    --blocksize 1000000 --nqueries 1000 --nprobe 128 --log - -v
```

As with `ivf_flat`, it is recommended that you run `flat_l2` using the setup scripts in `src/benchmarks` (or that you use your own scripts).

### Observations

For many values of the search options, the `gemm` approach appears to be the fastest,
but you may want to experiment with the example problems. If you have Intel's MKL BLAS
available, you should use those as they are significantly faster than almost all open-source
versions of BLAS.

For the other approaches, different parameter values may result
in different performance. Significant experimentation would need
to be done to find those, however, and it isn't clear that the
performance of `gemm` could be matched at any rate.

## Benchmarking With Provided Examples

Example. Run the `ivf_flat` CLI program for the `bigann1M` example problem. We assume the appropriate arrays have
been installed locally in the `gp3` subdirectory (see [Example Datasets](#example-datasets) below).

### Setup

Prior to runnning any benchmakrs, we first set up the benchmarking environment

```
  cd src/benchmarks
  . ./setup.bash
```

This only needs to be once per shell session -- you can run various benchmark configurations without having to rerun the setup script.

Once `setup.bash` has been sourced, there are two steps to running a benchmark. First, you select which dataset you
want to run with. Second you invoke the benchmarking function.

In choosing a benchmark, you run a defined bash function of the form:

```
init_<dataset>_<datasource>
```

For example

```
init_1M_gp3
```

selects the `bigann1M` dataset from the `gp3` datasource.
Once an `init` function has been invoked you can invoke a benchmark function, either `ivf_query` (which runs the `ivf_flat` CLI program) or `flat_query` (which runs the `flat_l2`).

### Examples

Running the benchmarks

#### Basic invocation.

Run the `ivf_query` CLI program for the `bigann1M` example problem. We assume the appropriate arrays have
been installed under the `gp3` subdirectory per [Example Datasets](#example-datasets) below. Here we just run the query and report recall.

```
  cd src/benchmarks
  . ./setup.bash
  init_1M_gp3
  ivf_flat --nqueries 16 --nprobe 16 --finite
```

Output:

```txt
# total intersected = 148 of 160 = R@10 of 0.925
```

Example with S3. Running with arrays from S3 is an almost identical invocation

```txt
  init_1M_s3
  ivf_flat --nqueries 16 --nprobe 16 --finite
```

#### Basic invocation.

To get complete information about a run, including various pieces of timing information, we add `--log -` to the command line.

```txt
  cd src/benchmarks
  . ./setup.bash
  init_1M_gp3
  ivf_flat --nqueries 16 --nprobe 16 --finite --log -
```

This run will generate copious output (with sufficient information to enable duplication of the results).

```txt
# total intersected = 148 of 160 = R@10 of 0.925
# [ Repo ]: TileDB-Vector-Search @ lums/tmp/release_prep / 6b83bd2
# [cmake source directory]: /Users/lums/TileDB/TileDB-Vector-Search/src
# [cmake build type]: Release
# [compiler]: /Library/Developer/CommandLineTools/usr/bin/c++
# [compiler id]: AppleClang
# [compiler version ]: 14.0.3.14030022
# [c++ flags]: -Wall -Wno-unused-variable
# [c++ debug flags ]: -O0 -g -fno-elide-constructors
# [c++ release flags ]: -Ofast -DNDEBUG
# [c++ relwithdebinfo flags]: -Ofast -g -DNDEBUG
  -|-   Algorithm  Queries  nprobe    k_nn   thrds  recall         [A]         [B]         [C]         [D]         [E]         [F]         [G]         [H]         [I]         [J]         [K]         [L]         [M]         [N]         [O]         [P]         [Q]         [R]         [S]         [T]         [U]
  -|-     qv_heap       16      16      10      16   0.925     0.00114     0.00188      0.0462     0.00174      0.0657      0.0504     0.00170    0.000004    0.000513    0.000380     0.00200    0.000004    0.000015    0.000007     0.00201    0.000259    0.000838        27.5        26.8       0.905       0.008
[A]: load /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/bigann_1M_GT_nnids (s)
[B]: load /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/centroids.tdb (s)
[C]: load /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/parts.tdb (s)
[D]: load /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/query_public_10k (s)
[E]: mainall inclusive query time (s)
[F]: nuv_query_heap_finite_ram /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/parts.tdb (s)
[G]: nuv_query_heap_finite_ram in RAM (s)
[H]: nuv_query_heap_finite_ram_top_k (s)
[I]: partition_ivf_index (s)
[J]: qv_query_nth (s)
[K]: read_vector /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/index_size.tdb (s)
[L]: tdbBlockedMatrix /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/bigann_1M_GT_nnids (s)
[M]: tdbBlockedMatrix /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/centroids.tdb (s)
[N]: tdbBlockedMatrix /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/query_public_10k (s)
[O]: tdbBlockedMatrix constructor (s)
[P]: tdbPartitionedMatrix /Users/lums/TileDB/TileDB-Vector-Search/external/data/gp3/1M/parts.tdb (s)
[Q]: tdbPartitionedMatrix constructor (s)
[R]: load (MiB)
[S]: nuv_query_heap_finite_ram (predicted) (MiB)
[T]: nuv_query_heap_finite_ram (upper bound) (MiB)
[U]: read_vector (MiB)
```

The line beginning with `-|-` contains fine-grained timing and memory usage information. The columns are keyed with
a letter, which are explained in the index that follows. One time of interest will be the time with the text "in RAM", which
measures the amount of wall-clock time spent in the indicated query. Another time of interest is the total query time (including all array loads), which is indicated by the key containing `main`. The number of queries is also displayed; QPS can be computed by dividing the number of queries by the time in RAM, or by the total query time, depending on what you are reporting.

**NB:** The timers are keyed on the name of the function, as generated by the compiler, in which they are contained. Since different compilers generate different functions names, the column of interest for time in RAM will differ. As of the writing of this document, `clang` is in column `[G]`, while `g++` is in column `[D]`.

## Example Datasets

### S3

A number of example datasets based on `bigann` have been provided in S3. These can be accessed in the `s3://tiledb-vector-search` S3 bucket. The different examples are available in "subdirectories" in that bucket
named according to the example problem:

```
  s3://tiledb-vector-search/bigann1B
  s3://tiledb-vector-search/bigann100M
  s3://tiledb-vector-search/bigann10M
  s3://tiledb-vector-search/bigann1M
```

In each subdirectory are the arrays needed for `ivf_flat` and `flat_l2`. The arrays use a naming
scheme based on the subdirectory containing them. For example

```
  s3://tiledb-vector-search/bigann1B/bigann_base
  s3://tiledb-vector-search/bigann1B/bigann_train
  s3://tiledb-vector-search/bigann1B/bigann1B_query
  s3://tiledb-vector-search/bigann1B/bigann1B_groundtruth

  s3://tiledb-vector-search/bigann1B/bigann1B_centroids
  s3://tiledb-vector-search/bigann1B/bigann1B_parts
  s3://tiledb-vector-search/bigann1B/bigann1B_index
  s3://tiledb-vector-search/bigann1B/bigann1B_index_size
  s3://tiledb-vector-search/bigann1B/bigann1B_ids
```

Similarly, the `bigann100M` example files have the form

```
s3://tiledb-vector-search/bigann100M/bigann100M_base
```

and so on.

These arrays all store the `sift_inputs_uri` and `sift_parts_uri` arrays as `uint8_t`. The arrays are
derived from those found on the `bigann` web page `http://corpus-texmex.irisa.fr`. The
arrays named simply `bigann` are the 1B array example (`ANN_SIFT1B` on the site), converted
into a TileDB array. The `100M`, `10M`, and `1M` examples consist of the first 100M elements
of `ANN_SIFT1B`, the first 10M elements, and the first 1M elements, respectively.
The queries and corresponding ground truth arrays are the same for all examples and
are converted to TileDB arrays from

```
https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin

https://comp21storage.blob.core.windows.net/publiccontainer/comp21/bigann/public_query_gt100.bin
```

**NB:** You should not run the larger examples using arrays stored in S3 from your local machine (desktop or laptop) as there are substantial charges for data egress from S3 (in addition, the bandwidth is extremely throttled, so loading them could take hours). It is recommended that

- If you want to run problems on your local desktop or laptop that you copy the arrays from S3 to your local machine (but see below)
- If you want to run problems on the larger examples that you do so using a suitable EC2 instance or that you do so using TileDB Cloud (recommended)

In general, the 1M or 10M examples are good for experimenting on your local machine with the C++ CLI programs.

### Local Filesystem

If you would prefer to have the arrays in local storage, you can download them from S3 using the `aws s3 sync` command (you will need to have AWS command line utilities installed).
The most basic usage of `aws s3 sync` is

```
aws s3 sync <S3Uri> <LocalPath>
```

which will copy the file at `<S3URI>` to the file `<LocalPath>`. For TileDB arrays, which are actually directories, you
need to issue the command with the `--recursive option`

```
aws s3 sync <S3Uri> <LocalPath> --recursive
```

So, for example, to download the array `s3://tiledb-vector-search/bigann10M/bigann10M_base` you would issue the command

```
aws s3 sync s3://tiledb-vector-search/bigann10M/bigann10M_base ./bigann10M_base --recursive
```

**Warning** The `sift_inputs_uri` and `sift_parts_uri` arrays for `bigann1B` are more than 120GB each, meaning downloading the full set of arrays for this problem will consume a quarter of a TB of storage (and incur the corresponding egress chages).
It is recommended that you copy the 1B and 100M examples to local storage only if you know your machine has sufficient available storage.

#### Using arrays in your local filesystem

To use the downloaded arrays in your local filesystem with the scripts in `src/benchmarks`, you will need to
customize some of the paths in `setup.bash`. It is recommended that, regardless of the path to it, that you
structure storage of your local arrays as

```
.
└── gp3
    ├── 10M
    │   ├── bigann10M_base
    │   ├── bigann_10M_GT_nnids
    │   ├── centroids.tdb
    │   ├── get.bash
    │   ├── ids.tdb
    │   ├── index.tdb
    │   ├── index_size.tdb
    │   ├── parts.tdb
    │   └── query_public_10k
    ├── 1M
    │   ├── bigann1M_base
    │   ├── bigann_1M_GT_nnids
    │   ├── centroids.tdb
    │   ├── get.bash
    │   ├── ids.tdb
    │   ├── index.tdb
    │   ├── index_size.tdb
    │   ├── parts.tdb
    │   └── query_public_10k
    ├── etc
```

If you have the local arrays in this structure, you need only update the `gp3_root` variable in `setup.bash`.

To download a single set of arrays

```
aws s3 sync s3://tiledb-vector-search/bigann10M/ ./bigann10M --recursive
```

Note that the `bigann100M` and `bigann1B` array sets are quite large and will tax both the disk storage of a typical desktop or laptop. It is recommended that you copy those to local storage only if you know your machine has sufficient available storage. You should also use the `--finite` option with a sutable value of `--blocksize` if you intend to perform queries on
a desktop or laptop agains the 100M or 1B vector datasets.

_If you have sufficient storage and only if you have sufficient storage_ you can download the entire corpus with
To download a single set of arrays

```
aws s3 sync s3://tiledb-vector-search/ ./tiledb-vector-search --recursive
```

### License

All original problem data and the TileDB arrays derived therefrom
are licensed under the CC0 license.
