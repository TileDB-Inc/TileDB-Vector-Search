# `IVFFlatIndex`

There are four primary classes in the `IVFFlatIndex` design:

1. `IVFFlatIndex`: The type-erased C++ class interface to an IVF index.
2. `ivf_flat_index`: The typed C++ class that implements the IVF index.
3. `ivf_flat_group`: The C++ class that represents the TileDB comprising the arrays that make up the IVF index (centroids, partititioned vectors, partitioned ids, index, etc.).
4. `ivf_flat_index_metadata`: The C++ class that represents the metadata for the group (types of the arrays, history of the group, etc.)

## Design Principles

The `ivf_flat_index` class is assumed to hold the complete and self-consistent
information necessary to perform a query at a particular point in time. Its
primary member data are a feature vectory array of `centroids_` and a partitioned
feature vector array `partitioned_vectors_` comprising an array of partitioned
vectors, a vector of partitioned ids `partitioned_ids_`, and an index vector
demarcating the partitions in the vectors and the ids. Note that this partitioned
feature vector array encapsulates a complete and self-consistent set of partitioned
vectors and ids. That is, its indices refer only to the vectors and ids
contained in `partitioned_vectors_`. An indexed query can be performed directly
on a partitioned feature vector array.

The `ivf_flat_index` is simply an inverted index. It is not "aware" of anything
having to do with TileDB arrays, including URIs, groups, arrays, metadata, and the
like. That is all handled by the `ivf_flat_group` class.

The `ivf_flat_group` is the interface between the `ivf_flat_index` and the
TileDB index arrays. It contains all of the information necessary to fill the
arrays in an `ivf_flat_index` at a given timestamp.

The `ivf_flat_index_metadata` class encapsulates metadata in a class separate from
`ivf_flat_group` for the sake of convenience.

## Functionality

The `ivf_flat_index` class provides three classes of functionality:

1. Creating an inverted file index and writing it to a group; and
2. Searching the index for the nearest neighbors of the vectors in a given query.
3. Updating the index, i.e., insert or delete vectors from the index.

Note that in some sense creating an inverted index is the special case of
adding a (potentially very large) set of vectors to an empty group.

## High-Level API

Below are the primary interfaces for each of these classes. Note that
`FeatureVectorArray` is a type-erased class for arrays of feature vectors
(i.e., for matrices).

### `IVFFlatIndex`

This is a type-erased wrapper around the C++ `ivf_flat_index` class. It is this class that
is intended to be called directly from Python.

```c++
/**
 * Type-erased interface for an IVF Flat Index
 */
class IVFFlatIndex {

  /**
   * Open the index at `group_uri`.  The group metadata is assumed
   * to contain type information for the different constituents of
   * the index.  The constructor peeks at those and creates a corresponding
   * typed C++ `ivf_flat_index` object.  If the type information does
   * not exist in the metadata, it will peek at the constituent arrays.
   *
   * The `mode` parameter can be set to its default or set to `create`.
   * If it is set to its default and the group at `group_uri` does not
   * exist, an error will be thrown.  If `mode` is set to `create` and
   * the group at `group_uri` does not exist, a group with empty arrays
   * with default parameters will be created.  If the group exists, `mode`
   * (currently) has no effect.
   *
   * If the group exists, the data for the index will be loaded according
   * to the specified `timestamp`.
   */
  IVFFLatIndex(const tiledb::Context& ctx, const std::string& group_uri,
               size_t timestamp, Mode mode);

  /**
   * Initializes a new `ivf_flat_index` based on the vectors in `input_vectors_uri`.
   * An exception will be thrown if the group at `group_uri` already exists,
   * unless `mode` is set to `create`.  After the index is created, it may be
   * queried, just as if it had been read from an already existing group.
   */
  IVFFLatIndex(const tiledb::Context& ctx, std::string& group_uri,
               std::string& input_vectors_uri, Mode mode);

  IVFFLatIndex(const tiledb::Context& ctx, std::string& group_uri,
               std::string& input_vectors_uri, Mode mode);

  /**
   * Equivalent to above, but taking data from a given `FeatureVectorArray`
   */
  IVFFLatIndex(const tiledb::Context& ctx, std::string& group_uri,
               const FeatureVectorArray& input_vectors, Mode mode);


  /**
   * Insert/delete the vectors from `input_vectors_uri` into the index.  An exception
   * will be thrown if the type or the dimension of the vectors in `input_vectors_uri`
   * is not the same as the vectors in `partitioned_vectors_`.
   */
  void insert(const tiledb::Context& ctx, const std::string& input_vectors_uri);
  void delete(const tiledb::Context& ctx, const std::string& input_vectors_uri);

  /**
   * Insert/delete the vectors from `input_vectors` into the index.
   */
  void insert(const FeatureVectorArray& input_vectors);
  void delete(const FeatureVectorArray& input_vectors);

  /**
   * Reindex. The inserted and deleted vectors are not directly inserted into the
   * index but are rather kept in an auxiliary "flat" index.  As the size of the
   * flat index grows, the cost of applying queries against it will grow. Accordingly,
   * inverted file index will need to be recreated, incorporating all of the updates,
   * and removing all of the deletions.
   */
  void reindex();

  /**
   * We may (or may not) want to cache data rather than writing small amounts of data
   * to the group.  In that case, we may want to provide explicit control to force
   * writing of the group.
   */
  void flush();

  /**
   * Search for nearest neighbors of vectors in `query_uri`, using the first
   * `num_queries` of them if `num_queries` is specified.  An exception is thrown
   * if the vectors in `query_uri` are not the same dimension and type as the
   * vectors in `partitioned_vectors_`.
   *
   * Searches are performed against the index corresponding to the timestamp with
   * which the index was created.  Does not load more than `upper_bound` vectors
   * into memory at a time when performing the query.
   */
  std::tuple<FeatureVectorArray, FeatureVectorArray>
  query(const tiledb::Context& ctx, const std::string& query_uri,
        size_t num_queries, size_t nprobe, size_t k, size_t upper_bound);

  std::tuple<FeatureVectorArray, FeatureVectorArray>
  query(const FeatureVectorArray& query_vectors, size_t nprobe, size_t k, size_t upper_bound);

  /**
   * Perform a query against only those partitions numbered between
   * [first_part, last_part) and return a heap containing the k best
   * scores and ids.
   *
   * This query is intended to be called by a task graph.
   * It will compute the necessary information for active queries and
   * active partitions.  Most suited to the case where it is
   * expected most of the partitions will be active.
   */
  MinHeap query(const FeatureVectorArray& query, size_t nprobe, size_t k,
                size_t first_part, size_t last_part, size_t upper_bound);

  /**
   * Perform a query against the index, restricting the search to the
   * partitions specified in `active_partitions` with the queries
   * specified in `active_queries`.  Note that for each partition,
   * `active_queries` contains a list of the queries relevant for that
   * partition.  The set of active partitions is the set of partitions
   * for which there is at least one active query.
   *
   * `active_partitions` and `active_queries` can be computed with the
   * function `partition_for_active`.
   */
  MinHeap query(const FeatureVectorArray& query, size_t nprobe, size_t k,
                const FeatureVector& active_partitions,
                const FeatureVectorArray& activer_queries, size_t upper_bound);

  /**
   * Computes the active partitions and active queries for the given query.
   * Internally applies a query of the centroids against the queries.  This
   * can safely be done on the centralized instance since the number of
   * queries and the number of centroids is assumed to be modest and will fit
   * completely into memory.
   */
  std::tuple<FeatureVector, FeatureVectorArray>
  partition_for_active(const FeatureVector& query, size_t nprobe);
};
```

## Implementation

We seek to provide as uniform an interface as possible across different index
classes and to reuse as much code as possible.

- `vector_search_index` base class that provides common API

- `flat_l2_index` derived class supporting flat search with L2 norm distance. Note that the distance computation is parameterized so that `flat_l2_index` can also support dot, cosine, jaccard, etc.
- `flat_pq_index` derived class supporting flat search with L2 norm (et al) distance. Vectors are compressed with PQ encoding.
- `ivf_flat_index` derived class supporting inverted index search without compression, and with parameterized distance over uncompressed vectors.
- `ivf_pq_index` derived class supporting inverted index search with parameterized distance over PQ compressed vectors.
- `vamana_index` derived class supporting graph-based vamana indexing and querying with parameterized distance over uncompressed vectors.
- `vamana_pq_index` derived class supporting graph-based vamana indexing and querying with parameterized distance over PQ compressed vectors.

The `flat` and `ivf` indexes can all support additions and deletions in the same way -- by using an auxiliary array to hold the updates. Queries are performed by first querying the "main" index and then filtering those results using the contents of the query applied to the updates array. When the updates array has grown sufficiently large, the indexes are re-indexed.

In the non-cloud case, insertions into the `vamana` (graph-based) indexes is extrememly efficient and can be done immediately (without the use of a separate updates array).

### `ivf_flat_index`

### `ivf_flat_group`

### `ivf_flat_index_metadata`

Product quantization is an approach to compressing the vectors in a vector database.
The basic idea is to divide the vectors into `m` subvectors and to quantize each subvector
into `k` centroids. The centroids are stored in a codebook. The original vectors are
then replaced by the indices of the centroids in the codebook.
