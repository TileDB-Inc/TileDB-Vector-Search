# feature-vector-prototype
Directory for prototyping feature vector / similarity search

## Very basic steps


### Running flat

The `flat` program is a basic program that reads the texmex sift feature vector data, 
performs either L2 (default) or cosine similarity and checks the result against the ground truth.

```
flat: feature vector search with flat index.
  Usage:
      tdb (-h | --help)
      tdb (--db_file FILE | --db_uri URI) (--q_file FILE | --q_uri URI) (--g_file FILE | --g_uri URI) 
          [--k NN] [--L2 | --cosine] [--order ORDER] [--hardway] [--nthreads N] [--nqueries N] [--ndb N] [-d | -v]

  Options:
      -h, --help            show this screen
      --db_file FILE        database file with feature vectors
      --db_uri URI          database URI with feature vectors
      --q_file FILE         query file with feature vectors to search for
      --q_uri URI           query URI with feature vectors to search for
      --g_file FILE         ground truth file
      --g_uri URI           ground true URI
      --k NN                number of nearest neighbors to find [default: 10]
      --L2                  use L2 distance (Euclidean)
      --cosine              use cosine distance [default]
      --order ORDER         which ordering to do comparisons [default: gemm]
      --hardway             use hard way to compute distances [default: false]
      --nthreads N          number of threads to use in parallel loops [default: 8]
      --nqueries N          size of queries subset to compare (0 = all) [default: 0]
      --ndb N               size of vectors subset to compare (0 = all) [default: 0]
      -d, --debug           run in debug mode [default: false]
      -v, --verbose         run in verbose mode [default: false]
```


### Build 'flat'
```bash
  cd < project root >
  mkdir build
  cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release
```


### Run `flat` with TileDB arrays

Basic invocation of flat requires specifying at least the base set of vectors, the query vectors, and
the ground truth vectors.  Other options 

Example:
```bash
% ./src/flat --db_uri s3://tiledb-lums/sift/sift_base       \
             --q_uri s3://tiledb-lums/sift/sift_query       \
	     --g_uri s3://tiledb-lums/sift/sift_groundtruth \
             --order gemm
```
This invokes flat on the medium sized array stored in S3, using the gemm-based method for comparison.

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
in both cases is the number of available hardware threads.


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

## Main TODO items

* **Use MKL or OpenBLAS to compute similarities.**  Not more than a day to implement.  Will need to make MKL and/or OpenBLAS an external package when the prototype code is merged into core.
* ***Implement similarity search on TileDB arrays** (accessed via `--db_uri`, &c.  Should not take more than a day (if used as an external program calling into libtiledb).
* **Use streaming approach to handle arrays/files that won't fit in local memory of one machine.**  This could be done before moving into core or afterwords.  It is probably better to do this with the prototype.  This shouldn't take more than two days.
* **Move prototype into core as a query.** If I can work with Luc this shouldn't take more than a week (and would not take up anywhere close to a week for Luc).  This shouldn't take more than a week, depending on how fancy we want to be.  Probably need to add a day or two to deal with designing the API.
* **Use parallel/distributed approach** to handle arrays/files that won't fit in local memory of one machine.  This is doable if we want to just parallelize an application using libtiledb.  However, if we want to do the similarity search "in the cloud" it might be better to orchestrate the distributed computation at the Python task graph level.

