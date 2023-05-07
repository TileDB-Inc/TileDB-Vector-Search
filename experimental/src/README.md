# C++ Support Library for Similarity Search with TileDB



## The Partitioned Search Driver

The `main` program for the driver performing partitioned search is  
called `ivf_flat` which does an approximate search based on a given partition (an index).

### Building ivf_hack

`ivf_hack` is built in the same manner as `flat` (below).



### Running ivf_hack

```
Usage:
    ivf_hack (-h | --help)
    ivf_hack --db_uri URI --centroids_uri URI --index_uri URI --part_uri URI --id_uri URI
            [--output_uri URI] [--query_uri URI] [--groundtruth_uri URI] [--ndb NN] [--nqueries NN] [--blocksize NN]
            [--k NN] [--cluster NN] [--nthreads N] [--region REGION] [--nth] [--log FILE] [-d | -v]

Options:
    -h, --help            show this screen
    --db_uri URI          database URI with feature vectors, to query against
    --centroids_uri URI   URI with centroid vectors from kmeans applied to the feature vector set
    --index_uri URI       URI with the paritioning index obtained from kmeans
    --part_uri URI        URI with the data vectors reordered and partitioned by kmeans
    --id_uri URI          URI with IDs of the partitioned vectors, as given in the original dataset  
    --output_uri URI      URI to store search results
    --query_uri URI       Optional URI containing a set of vectors to query the database against
    --groundtruth_uri URI Optionsal URI containing the ground truth vectors for the query vectors
    --nqueries NN         number of query vectors to use (0 = all) [default: 0]
    --ndb NN              number of database vectors to use (0 = all) [default: 0]
    --nthreads N          number of threads to use in parallel loops (0 = all) [default: 0]
    --k NN                number of nearest neighbors to search for [default: 10]
    --cluster NN          aka nprobe -- number of kmeanse clusters to use in the search [default: 100]
    --blocksize NN        number of vectors to process in memory from the vector dataset (0 = all) [default: 0]
    --nth                 use nth_element for finding top k rather than heap-based [default: false]
    --log FILE            log program stats to FILE (- for stdout)
    --region REGION       AWS S3 region for the arrays used by this program [default: us-east-1]
    -d, --debug           emit and log debugging level information [default: false]
    -v, --verbose         emit and log verbose level information [default: false]
```

### Running `ivf_hack` with TileDB Arrays

#### Query mode
`ivf_hack` performs a query for given queries against a specified dataset, using an inverted file index derived from applying kmeans partitioning against that dataset.

`ivf_hack`reads its data from TileDB arrays.  Reading from local data files is no longer supported. Previous functionality for writing the IVF index files (partitioned vectors, index, vector IDs) has been moved to another driver program (`index`).

`ivf_hack` takes as input a vector database, an array centroids, 
a reordered vector database, an array containing partition indexes for the reordered vector database, and an array of original ids of the shuffled vectors.
The user can also optionally specify
*  an array containing query vectors 
*  an array containing ground truth vectors

The user can also specify values for how many nearest neighbors to return for
each search vector, and how many partitions to use in searching for each search vector.

Example:
```
  ivf_hack --id_uri s3://tiledb-andrew/kmeans/ivf_hack/ids              \
           --index_uri s3://tiledb-andrew/kmeans/ivf_hack/index         \
           --part_uri s3://tiledb-andrew/kmeans/ivf_hack/parts          \ 
           --centroids_uri s3://tiledb-andrew/kmeans/ivf_hack/centroids \
           --db_uri s3://tiledb-andrew/sift/sift_base 
```

#### index_ivf

The `index_ivf` driver is complementary to `ivf_hack` and writes the arrays necesarry for indexing, given a vector database and centroid vectors produces by kmeans. This functionality was previously also included as part of `ivf_hack` but has been moved to the `index_ivf` driver.

The three arrays produced by `index_ivf` are the partitioned (shuffled) vector database, the partitioning indexes, and the original vector database ids for the shuffled vectors. 

Example:
```
  ivf_index --id_uri s3://tiledb-lums/kmeans/ivf_hack/ids              \
            --index_uri s3://tiledb-lums/kmeans/ivf_hack/index         \
            --part_uri s3://tiledb-lums/kmeans/ivf_hack/parts          \ 
            --centroids_uri s3://tiledb-lums/kmeans/ivf_hack/centroids \
            --db_uri s3://tiledb-lums/sift/sift_base                   
```

Note that the program will **not** overwrite any existing arrays.  It is the responsibility of the user to make sure that the arrays to be written do not exist when the program is executed.

**NB:** The reason write mode exists is because the available partitioning data created by kmeans programs only seem to include the centroids rather than the full set of data needed by `ivf_hack`.  
This functionality will also be used in conjunction with TileDB's own kmeans centroid generation (currently WIP).

## The Flat Search Driver

The driver program for 
performing flat search is  
called `flat` which does an exhaustive vector-by-vector comparison between a given
set of vectors and a given set of query vectors.


### Running flat

The `flat` program is a basic program that reads the `sift` feature vector data, either
from a TileDB array or from a file.  Both `flat` and `ivf_hack` use a simple (naive) array schema for the arrays that they use.  Future releases will include support for other pre-defined schema, as well as offer an API for user-defined schema.

The program currently 
performs search based on L2  similarity.  Future releases will include
cosine similarity, and Jaccard similarity.  The program can also check its results against a supplied set of ground truth vectors.  The `sift` dataset includes ground truth files computed with L2 similarity.

If the program is running in verbose mode, any vectors not mathing the ground truth will be printed to the console and logged.
**Note** A correct computation is not
unique.  That is, multiple vectors may return the same results, because of ties in their similarity scores.
Thus, the index values computed by `flat` may differ from the supplied
ground truth.  You should examine any differences logged to the output for evidence
of ties.  Most methods in `flat` seem to process ties in the same order as
the ground truth, so this only comes up in one or two cases.

### Usage
```
  Usage:
      flat (-h | --help)
      flat --db_uri URI --q_uri URI [--g_uri URI] [--output_uri URI] [--order ORDER] [--k NN]
          [--block N] [--nqueries N] [--nthreads N] [-d ] [-v]

  Options:
      -h, --help            show this screen
      --db_uri URI          database URI with feature vectors
      --q_uri URI           query URI with feature vectors to search for
      --g_uri URI           ground true URI
      --output_uri URI      output URI for results
      --order ORDER         which ordering to do comparisons [default: gemm]
      --k NN                number of nearest neighbors to find [default: 10]
      --block N             block database with size N (0 = no blocking) [default: 0]
      --nqueries N          size of queries subset to compare (0 = all) [default: 0]
      --nthreads N          number of threads to use in parallel loops (0 = all) [default: 0]
      --nth                 use nth_element for top k [default: false]
      -d, --debug           run in debug mode [default: false]
      -v, --verbose         run in verbose mode [default: false]
```


### Build 'flat'
```bash
  cd < project root >
  mkdir build
  cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DTileDB_DIR=/path/to/TileDB/cmake
```
**Note:** `flat` builds against `libtiledb`.  You will need to point your `cmake` to the directory that has the cmake configuration files.  That is, if the `libtiledb` you want to use is in `/usr/local/tiledb/lib/libtiledb.so` then you would set `TileDB_DIR` to `/usr/local/tiledb/lib/cmake`
```
  % cmake .. -DTileDB_DIR=/usr/local/tiledb/lib/cmake
```

Note that the following also appears to work.  You can set `TileDB_DIR` to the value of `CMAKE_INSTALL_PREFIX` that was used when building and installing `libtiledb`.  That is, if you built `libtiledb` with
```
  % cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/tiledb
```
Then if you set `TileDB_DIR` to `/usr/local/tiledb`
```
  % cmake .. -DTileDB_DIR=/usr/local/tiledb
```
the `fetch_content` call in `CMakeLists.txt` will also find the TileDB `cmake` configuration files.
To check if the right path will be searched, look for the output line
```
-- TileDB include directories are <>
```
This will tell you the path that will be used when building `flat`.  If it isn't the path you are expecting, e.g., if it is the system defaults when you expected something else, check the path you used when invoking `cmake`.

**If you don't specify a value for `TileDB_DIR`** the system default will be used.  That is, you do not have to specify a value for `TileDB_DIR` if the system defaults are good enough.

`flat` does require a fairly recent version of `libtiledb`.  If you get compilation errors along the lines of
```
In file included from /home/user/feature-vector-prototype/src/test/unit_sift_array.cpp:5:
/home/user/feature-vector-prototype/src/test/../sift_array.h:67:21: error: expected ';' after expression
    tiledb::Subarray subarray(ctx_, array_);
                    ^
                    ;
/home/user/feature-vector-prototype/src/test/../sift_array.h:67:13: error: no member named 'Subarray' in namespace 'tiledb'
    tiledb::Subarray subarray(ctx_, array_);
    ~~~~~~~~^
/home/user/feature-vector-prototype/src/test/../sift_array.h:68:5: error: use of undeclared identifier 'subarray'
    subarray.set_subarray(subarray_vals);
```
then you likely need a more recent version of `libtiledb`.  To fix this, first try updating your instaleld version of `libtiledb` by invoking the appropriate "upgrade" or "update" command associated with your package manager (if you installed `libtiledb` using a package manager).  Otherwise, obtain an up-to-date version of `libtiledb` from the TileDB github repository at `https://github.com/TileDB-Inc/TileDB` and build and install that per the instructions there.

**Node:** If you are going to use S3 as a source for TileDB array input, your `libtiledb`
should be built with S3 support.



### Run `flat` with TileDB arrays

Basic invocation of flat requires specifying at least the base set of vectors, the query vectors, and
the ground truth vectors.  Other options

Example, run with small sized data set
```bash
% ./src/flat --db_uri s3://tiledb-lums/sift/siftsmall_base       \
             --q_uri s3://tiledb-lums/sift/siftsmall_query       \
	     --g_uri s3://tiledb-lums/sift/siftsmall_groundtruth \
             --order gemm
```


Example, run with medium sized data set.
```bash
% ./src/flat --db_uri s3://tiledb-lums/sift/sift_base       \
             --q_uri s3://tiledb-lums/sift/sift_query       \
	     --g_uri s3://tiledb-lums/sift/sift_groundtruth \
             --order gemm
```


Output
```text
# [ Load database, query, and ground truth arrays ]: 11960 ms
# [ Allocating score array ]: 350 ms
# [ L2 comparison (gemm) ]: 1190 ms
# [ L2 comparison colsum ]: 322 ms
# [ L2 comparison outer product ]: 320 ms
# [ L2 comparison finish ]: 140 ms
# [ Get top k ]: 1180 ms
# [ Checking results ]: 0 ms
# [ Total time gemm ]: 4470 ms
```

### Options

The `order` options specify which approach to use for computing the
similarity.  The `qv` and `vq` options compare the feature vectors
to the query vectors one by one.  The `qv` order has queries on
the outer loop and feature vectors on the inner loop.  The `vq`
oder has vectors on the outer loop and queries on the inner loop.
The `gemm` order uses the optimized linear algebra matrix-matrix
produce algorithm to do the comparisons.  (Presentation of details
of this TBD.)

`flat` is parallelized using `std::async`.  The `--nthreads` option
specifies how many threads to use.  In environments where `gemm` is
provided by MKL, the `OMP_NUM_THREADS` or `MKL_NUM_THREADS` option
will set the number of threads to be used by `gemm`.  The default
in both cases is the number of available hardware threads.  **Note:**
This problem is largely memory-bound.  Threading will help, but
you should not expect linear speedup.

The `--nqueries` option specifies how many of the query vectors to
use.  The default is to use all of them.  Setting this to a
smaller number can be useful for testing and will also allow
the medium dataset to fit into memory of a desktop machine.

The `--ndb` option specifies how much of the data set to
use.  Note that when using this that the computed result will
almost surely differ from the ground truth because all
potential neighbors have not been considered.  This may be more
useful as a performance testing mechanism (e.g.) but we
would need to silence error reporting (perhaps simply by
not supplying a ground truth file/array).

The `--k` option specifies how many neighbors to keep (i.e.,
for "top k").

The `--hardway` option specifies how to collect the top k
data.  If this option is set, A complete set of scores is
computed and then filtered for the top k vectors.  If
the value is not set, a heap is used to filter the scores
on the fly.  This only affects the `qv` and `vq` options.
The `gemm` option computes all scores.

The `-v` and `-d` options turn on verbosity and debugging,
respectively.  (There is not currently a large amount
of output for this.)

### Observations

The `gemm` approach appears to be by far the fastest, and is
about a factor of four faster than the open source `faiss`
project for the flat L2 query.

For the other approaches, different parameter values may result
in different performance.  Significant experimentation would need
to be done to find those, however, and it isn't clear that the
performance of `gemm` could be matched at any rate.



## About the datasets

`flat` is set up to run with the sift reference arrays available on `http://corpus-texmex.irisa.fr`
and complete info about those arrays can be found there.  The basic characteristics of
the problems are:

| Vector Set  | name      | dimension | nb base       | nb query | nb learn     | format  |
|-------------|-----------|-----------|---------------|----------|--------------|---------|
| ANN_SIFT10K | siftsmall | 128       | 10,000	       | 100	  | 25,000	     | fvecs   |
| ANN_SIFT1M  | sift      | 128	   | 1,000,000	   | 10,000	  | 100,000	     | fvecs   |
| ANN_GIST1M  | gist      | 960       | 1,000,000	   | 1,000	  | 500,000      | 	fvecs  |
|  ANN_SIFT1B | sift1b    | 128	   | 1,000,000,000 | 10,000	  | 100,000,000	 | bvecs   |


These have been ingested as TileDB arrays and can be found at
```text
s3://tiledb-lums/sift/${name}_{base,query,learn,ground_truth}
```
Read permissions are set for all of TileDB.


### Get file data (optional)
```bash
  cd < project root >
  mkdir -p external/data
  cd external/data
  wget ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
  tar zxf siftsmall.tar.gz
  wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
  tar zxf sift.tar.gz
```

### Ingesting the data
The python notebook `python/ingest_eq.ipynb` was used to
convert the original data files to TileDB arrays in S3.  **Important:**
Do not re-run this unless you change the paths to avoid
overwriting the existing arrays.


### Run `flat` with local files

`flat` can also be run with the original files from the data repository.
```bash
  cd < project root >
  cd build

  # Get help
  ./src/flat -h

  # Run with small data
  ./src/flat  --db_file ../external/data/siftsmall/siftsmall_base.fvecs \
              --q_file ../external/data/siftsmall/siftsmall_query.fvecs \
              --g_file ../external/data/siftsmall/siftsmall_groundtruth.ivecs \
              --k 100
```
This specifies to run `flat` on the `siftsmall` data, checking the first 100 neighbors.  If all of the checks pass, the program will simply return 0.  Otherwise, it will print error messages for each of its computed neighbors that does not pass (does not match ground truth).

You can also run on the medium data set:
```c++
  # Run with small data
  time ./src/flat  --db_file ../external/data/sift/sift_base.fvecs \
                    --q_file ../external/data/sift/sift_query.fvecs \
                    --g_file ../external/data/sift/sift_groundtruth.ivecs \
                    --k 100
```

The memory and CPU requirements for the 1B dataset become prohibitive and probably can't be run on a typical desktop or typical server.

## CHANGELOG

* **Use streaming approach to handle arrays/files that won't fit in local memory of one machine.**  This could be done before moving into core or afterwords.  It is probably better to do this with the prototype.  This shouldn't take more than two days.
* **Improve performance with better blocking for memory use** In conjunction with reorganizing for out-of-core operation, we should also arrange the in-memory computation to make better use of the memory hierarchy.
* **Make ground truth comparison an optional argument**
* **Perform ground truth comparison only when requested**


## Main TODO items

(**Note:** We can now say that TileDB can be used for similarity search.)

* **Use OpenBLAS instead of MKL for gemm** and incorporate into the build process.  This should take less than a day.
* **Move prototype into core as a query.** If I can work with Luc this shouldn't take more than a week (and would not take up anywhere close to a week for Luc).  This shouldn't take more than a week, depending on how fancy we want to be.  Probably need to add a day or two to deal with designing the API.
* **Provide --id argument** to allow selection of a single vector to query
* **Use parallel/distributed approach** to handle arrays/files that won't fit in local memory of one machine.  This is doable if we want to just parallelize an application using libtiledb.  However, if we want to do the similarity search "in the cloud" it might be better to orchestrate the distributed computation at the Python task graph level.
* **Finish implementation of cosine similarity** This should be fairly straightforward to implement. However, we have not ground truth to verify it with the sift benchmark dataset.
* **Implement parallelism using task graph** This should not be too difficult, a couple of days, as the `std::async` parallelism is similar to very basic task graph usage.
