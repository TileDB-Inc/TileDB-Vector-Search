## C++ CLI for index and query with ivf flat index

### Usage

```c++
    ivf_index (-h | --help)
    ivf_index --db_uri URI --index_uri URI [--ftype TYPE] [--idtype TYPE] [--pxtype TYPE]
                 [--init TYPE] [--num_clusters NN] [--max_iter NN] [--tol NN]
                 [--nthreads NN] [--log FILE] [--stats] [-d] [-v] [--dump NN]

Options:
    -h, --help              show this screen
    --db_uri URI            database URI with feature vectors
    --index_uri URI         group URI for storing ivf index
    --ftype TYPE            data type of feature vectors [default: float]
    --idtype TYPE           data type of ids [default: uint64]
    --pxtype TYPE           data type of partition index [default: uint64]
    -i, --init TYPE         initialization type, kmeans++ or random [default: random]
    --num_clusters NN       number of clusters/partitions, 0 = sqrt(N) [default: 0]
    --max_iter NN           max number of iterations for kmeans [default: 10]
    --tol NN                tolerance for kmeans [default: 1e-4]
    --nthreads N            number of threads to use in parallel loops (0 = all) [default: 0]
    --log FILE              log info to FILE (- for stdout)
    --stats                 log TileDB stats [default: false]
    -d, --debug             run in debug mode [default: false]
    -v, --verbose           run in verbose mode [default: false]
```

### Example

```bash
ivf_index --db_uri siftsmall_base --ftype float --index_uri flatIVF_index_siftsmall_base --pxtype uint64 --idtype uint32  -v -d --log -
```
