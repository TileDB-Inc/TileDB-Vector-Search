# feature-vector-prototype
Directory for prototyping feature vector / similarity search

## Very basic steps

### Get data
```bash
  cd < project root >
  mkdir -p external/data
  cd external/data
  wget ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
  tar zxf siftsmall.tar.gz
  wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
  tar zxf sift.tar.gz
```


### Build 'flat'
```bash
  cd < project root >
  mkdir build
  cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release
```


### Run `flat`
`flat` is a basic program that reads the texmex sift feature vector data, performs either L2 (default) or cosine similarity and checks the result against the ground truth.
```bash
  cd < project root >
  cd build

  # Get help
  ./src/flat -h

  # Run with small data
  time ./src/flat  --db_file ../external/data/siftsmall/siftsmall_base.fvecs \
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

