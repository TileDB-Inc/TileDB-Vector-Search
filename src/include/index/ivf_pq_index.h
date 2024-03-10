/**
 * @file   ivf_pq_index.h
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2024 TileDB, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * @section DESCRIPTION
 *
 * @note As of now (2024-03-10), we are using a naive encoding rather than
 * the difference between the centroids and the vectors.
 * @todo  Implement the difference encoding
 *
 *
 * I. To compress, train, add, the steps would be:
 * - train_pq to get cluster_centroids_ and distance_tables_
 *   - uses sub_kmeans on uncompressed vectors
 *   - cluster_centroids_ is a feature_vector_array of feature_type, dimension_
 * x num_clusters_
 * - encode training_set using cluster_centroids_ to get
 * unpartitioned_pq_vectors_, which are a feature_vector_array of pq_code_type,
 * num_subspaces_ x num_vectors_
 * - train using unpartitioned_pq_vectors_ to get ivf_centroids_, which is a
 *   feature_vector_array of pq_code_type, num_subpaces x num_partitions_
 *   - uses kmeans -- which would have to be a new implementation that
 *     understands compressed vectors
 * - partition unpartitioned_pq_vectors_ using ivf_centroids_ to get
 * partitioned_pq_vectors_
 *   - use qv_partition with symmetric difference
 *
 * II. If we train, add, compress, the steps would be:
 * - train using training_set to get ivf_centroids_, which is a
 *   feature_vector_array of feature_type, dimension_ x num_partitions_
 *   - uses ivf_flat::train
 *     - uses kmeans (just as it is used today)
 * - add
 *   - partition training set using ivf_centroids_ to get partitioned_vectors_
 *     - uses ivf_flat::add
 *       - uses partitioned_matrix constructor, which just shuffles
 * - pq train with partitioned_vectors_ to get cluster_centroids_ and
 *   distance_tables_
 * - compress partitioned_vectors_ and ivf_centroids_
 * - query with symmetric distance
 *
 * III. OR, train , compress, add, the steps would be:
 * - train using training_set to get ivf_centroids_, which is a
 *   feature_vector_array of feature_type, dimension_ x num_partitions_
 *   - uses ivf_flat::train
 *     - uses kmeans (just as it is used today)
 * - add
 *   - pq_train with training set to get cluster_centroids_ and distance_tables_
 *     - uses sub_kmeans
 *   - compress training set and ivf_centroids_ to get pq_vectors_ and
 * pq_ivf_centroids_
 *     - uses flatpq::add
 *   - partition pq_vectors using pq_ivf_centroids_ to get
 * partitioned_pq_vectors_
 *     - uses qv_partition with symmetric distance
 * - query with symmetric distance
 *
 * we can experiment with II or II by putting add and compress into add()
 *
 * Summary:
 * II
 *   - use training vectors to get flat_ivf_centroids_ (ivf_flat::train)
 *   - use flat_ivf_centroids_ to get partitioned_flat_vectors_ (ivf_flat::add)
 *   - compress flat_ivf_centroids_ and partitioned_flat_vectors_
 * III
 *   - use training vectors to get flat_ivf_centroids_ (ivf_flat::train)
 *   - compress training vectors and flat_ivf_centroids_ to get pq_ivf_centroids
 *   - use pq_ivf_centroids to get partitioned_pq_vectors_ (ivf_pq::add)
 */

#ifndef TILEDB_ivf_pq_H
#define TILEDB_ivf_pq_H

#include "algorithm.h"
#include "concepts.h"
#include "cpos.h"
#include "index/index_defs.h"
#include "index/ivf_pq_group.h"
#include "index/kmeans.h"
#include "linalg.h"

#include "detail/flat/qv.h"
#include "detail/ivf/index.h"
#include "detail/ivf/partition.h"
#include "detail/ivf/qv.h"

#include <tiledb/group_experimental.h>
#include <tiledb/tiledb>
#include <type_traits>

/**
 * Class representing an inverted file (IVF) index for flat (non-compressed)
 * feature vectors.  The class simply holds the index data itself, it is
 * unaware of where the data comes from -- reading and writing data is done
 * via an ivf_pq_group.  Thus, this class does not hold information
 * about the group (neither the group members, nor the group metadata).
 *
 * @tparam partitioned_vectors_feature_type
 * @tparam partitioned_ids_type
 * @tparam partitioning_index_type
 */
template <
    class partitioned_vectors_feature_type,
    class partitioned_ids_type = uint64_t,
    class partitioning_index_type = uint64_t>
class ivf_pq_index {
 public:
  using feature_type = partitioned_vectors_feature_type;
  using id_type = partitioned_ids_type;
  using indices_type = partitioning_index_type;
  using score_type = float;  // @todo -- this should be a parameter?

  // @todo IMPORTANT: Use a uint64_t to store 8 bytes together -- should make
  // loads and other operations faster and SIMD friendly
  using pq_code_type = uint8_t;

  using pq_vector_feature_type = pq_code_type;

  // The pq_centroids store the (unencoded) centroids for each subspace
  using flat_vector_feature_type = score_type;

 private:
  using flat_storage_type = ColMajorMatrix<pq_code_type>;
  using tdb_flat_storage_type = ColMajorMatrix<pq_code_type>;

  using pq_storage_type = ColMajorPartitionedMatrix<  // was: storage_type
      pq_code_type,                                   // was: feature_type
      partitioned_ids_type,
      indices_type>;

  using tdb_pq_storage_type =
      tdbColMajorPartitionedMatrix<  // was: tdb storage_type
          pq_code_type,              // was: feature_type
          partitioned_ids_type,
          indices_type>;

  /*
   * We need to store three different sets of centroids.
   *   - The flat ivf centroids_ which are uncompressed and used to partition
   *     the entire training set with train / add+compress patterns.
   *   - The pq ivf centroids - compressed version of the ivf centroids.  These
   *     are used in queries with symmetric distance.  They are also used in
   *     the compress / train / add pattern.
   *   - The cluster_centroids_ that partition each subspace of the training set
   *     into another set of partitions. There are num_clusters of these, with
   *     each centroid being of size dimension_.  These are used to build the
   *     distance_tables_ and compress the training set and ivf centroids
   */
  using flat_ivf_centroid_storage_type =
      ColMajorMatrix<flat_vector_feature_type>;
  using tdb_flat_ivf_centroid_storage_type =
      tdbColMajorMatrix<flat_vector_feature_type>;

  using pq_ivf_centroid_storage_type = ColMajorMatrix<pq_vector_feature_type>;
  using tdb_pq_ivf_centroid_storage_type =
      tdbColMajorMatrix<pq_vector_feature_type>;

  using cluster_centroid_storage_type =
      ColMajorMatrix<flat_vector_feature_type>;
  using tdb_cluster_centroid_storage_type =
      tdbColMajorMatrix<flat_vector_feature_type>;

  constexpr static const IndexKind index_kind_ =
      IndexKind::IVFPQ;  // was: IVFFlat

  /****************************************************************************
   * Index group information
   ****************************************************************************/

  /** The timestamp at which the index was created */
  uint64_t timestamp_{0};

  std::unique_ptr<ivf_pq_group<ivf_pq_index>> group_;

  /****************************************************************************
   * Index representation
   ****************************************************************************/

  // Cached information about the partitioned vectors in the index
  uint64_t dimension_{0};
  uint64_t num_partitions_{0};

  // Cached information about the pq encoding
  uint64_t num_subspaces_{0};
  uint64_t sub_dimension_{0};
  constexpr static const uint64_t bits_per_subspace_{8};
  constexpr static const uint64_t num_clusters_{256};

  /*
   * We are going to use train / compress / add pattern, so we need to store
   * flat ivf_centroids, pq_ivf_centroids, pq_vectors, partitioned_pq_vectors
   */

  flat_ivf_centroid_storage_type flat_ivf_centroids_;

  cluster_centroid_storage_type cluster_centroids_;
  std::vector<ColMajorMatrix<score_type>> distance_tables_;

  pq_ivf_centroid_storage_type pq_ivf_centroids_;
  pq_storage_type partitioned_pq_vectors_;
  flat_storage_type unpartitioned_pq_vectors_;
  // Or should these be unique_ptrs?
  // std::unique_ptr<pq_storage_type> partitioned_pq_vectors_;
  // std::unique_ptr<flat_storage_type> unpartitioned_pq_vectors_;

  // Some parameters for performing kmeans clustering for ivf index
  uint64_t max_iter_{1};
  float tol_{1.e-4};
  float reassign_ratio_{0.075};

  // Some parameters for performing kmeans clustering for pq compression
  uint64_t pq_max_iter_{1};         // new for pq
  float pq_tol_{1.e-4};             // new for pq
  float pq_reassign_ratio_{0.075};  // new for pq

  // Some parameters for execution
  uint64_t num_threads_{std::thread::hardware_concurrency()};
  uint64_t seed_{std::random_device{}()};

 public:
  using value_type = feature_type;
  using index_type = partitioning_index_type;  // @todo This isn't quite right

  /****************************************************************************
   * Constructors (et al)
   ****************************************************************************/

  // ivf_pq_index() = delete;
  ivf_pq_index(const ivf_pq_index& index) = delete;
  ivf_pq_index& operator=(const ivf_pq_index& index) = delete;
  ivf_pq_index(ivf_pq_index&& index) = default;
  ivf_pq_index& operator=(ivf_pq_index&& index) = default;

  /**
   * @brief Construct a new `ivf_pq_index` object, setting a number of
   * parameters to be used subsequently in training.  To fully create an index
   * we will need to call `train()` and `add()`.
   *
   * @param dimension Dimension of the vectors comprising the training set and
   * the data set.
   * @param nlist Number of centroids / partitions to compute.
   * @param max_iter Maximum number of iterations for kmans algorithm.
   * @param tol Convergence tolerance for kmeans algorithm.
   * @param timestamp Timestamp for the index.
   * @param seed Random seed for kmeans algorithm.
   *
   * @param num_subspaces number of subspaces to use for pq compression
   * @param sub_dimension dimension of each subspace (dimension / num_subspaces)
   *
   * @note PQ encoding generally is described as having parameter nbits, how
   * many bits to use for indexing into the codebook.  In real implementations,
   * this seems to always be 8 -- it doesn't make sense to be anything other
   * than that, else indexing would be too slow.  Accordingly, we set it as
   * a constexpr value to 8 -- and correspondingly, we set num_clusters to 256.
   *
   * @todo Use chained parameter technique for arguments
   * @todo -- Need something equivalent to "None" since user could pass 0
   * @todo -- Or something equivalent to "Use current time" and something
   * to indicate "no time traveling"
   * @todo -- May also want start/stop?  Use a variant?  TemporalPolicy?
   */
  ivf_pq_index(
      // size_t dim,
      size_t nlist = 0,
      size_t num_subspaces = 16,  // new for pq
      size_t max_iter = 2,
      float tol = 0.000025,
      size_t timestamp = 0,
      uint64_t seed = std::random_device{}())
      :  // , dimension_(dim)
      timestamp_{
          (timestamp == 0) ?
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count() :
              timestamp}
      , num_partitions_(nlist)
      , num_subspaces_{num_subspaces}  // new for pq
      , max_iter_(max_iter)
      , tol_(tol)
      , seed_{seed} {
    gen_.seed(seed_);
  }

  /**
   * @brief Open a previously created index, stored as a TileDB group.  This
   * class does not deal with the group itself, but rather calls the group
   * constructor.  The group constructor will initialize itself with information
   * about the different constituent arrays needed for operation of this class,
   * but will not initialize any member data of the class.
   *
   * The group is opened with a timestamp, so the correct values of base_size
   * and num_partitions will be set.
   *
   * We go ahead and load the centroids here.  We defer reading anything else
   * because that will depend on the type of query we are doing as well as the
   * contents of the query itself.
   *
   * @todo Is this the right place to load the centroids?
   *
   * @param ctx
   * @param uri
   * @param timestamp
   *
   */
  ivf_pq_index(
      const tiledb::Context& ctx,
      const std::string& uri,
      uint64_t timestamp = 0)
      : group_{std::make_unique<ivf_pq_group<ivf_pq_index>>(
            *this, ctx, uri, TILEDB_READ, timestamp_)} {
    if (timestamp_ == 0) {
      timestamp_ = group_->get_previous_ingestion_timestamp();
    }

    /**
     * Read the centroids.  How the partitioned_vectors_ are read in will be
     * determined by the type of query we are doing.  But they will be read
     * in at this same timestamp.
     */
    dimension_ = group_->get_dimension();
    num_partitions_ = group_->get_num_partitions();

    pq_ivf_centroids_ =
        tdbPreLoadMatrix<pq_vector_feature_type, stdx::layout_left>(
            group_->cached_ctx(),
            group_->centroids_uri(),
            std::nullopt,
            num_partitions_,
            0,
            timestamp_);
  }

  /****************************************************************************
   * Methods for building, writing, and reading the complete index.  Includes:
   *   - Method for encoding the training set using pq compression to create
   *     the cluster_centroids_ and distance_tables_.
   *   - Method to initialize the centroids that we will use for building the
   *IVF index.
   *   - Method to partition the pq_vectors_ into a partitioned_matrix of pq
   *encoded vectors.
   * @note With this approach, we are partitioning based on cluster_centroids_
   *the stored centroids are also encoded using pq.  Thus we can do our search
   *using the symmetric distance function.
   *
   * @todo Create single function that trains and adds (ingests)
   * @todo Provide interface that takes URI rather than vectors
   * @todo Provide "kernel" interface for use in distributed computation
   * @todo Do we need an out-of-core version of this?
   ****************************************************************************/

  /**
   * @brief Create the `cluster_centroids_` (encoded from the training set) and
   * create `distance_tables_`. We measure the maximum number of iterations and
   * minimum convergence over all of the subspaces and return a tuple of those
   * values. We compute all of the distance_tables_ regarless of the values of
   * max_local_iters_taken or min_local_conv relative to max_iter_ and tol_.
   *
   * @tparam V type of the training vectors
   * @tparam SubDistance type of the distance function to use for encoding.
   * Must be a cached_sub_distance_function.
   * @param training_set The set of vectors to compress
   *
   * @return tuple of the maximum number of iterations taken and the minimum
   * convergence
   *
   * @note This is essentially the same as flat_pq_index::train
   * @note Recall that centroids_ are used for IVF indexing.  cluster_centroids_
   * are still centroids, but they are divided into subspaces, and each portion
   * of cluster_centroids_ is the centroid of the corresponding subspace of the
   * training set.
   */
  template <
      feature_vector_array V,
      class SubDistance = cached_sub_sum_of_squares_distance>
    requires cached_sub_distance_function<
        SubDistance,
        typename ColMajorMatrix<feature_type>::span_type,
        typename ColMajorMatrix<flat_vector_feature_type>::span_type>
  auto train_pq(const V& training_set) {
    cluster_centroids_ =
        ColMajorMatrix<flat_vector_feature_type>(dimension_, num_clusters_);

    // Lookup table for the distance between centroids of each subspace
    distance_tables_ = std::vector<ColMajorMatrix<score_type>>(num_subspaces_);
    for (size_t i = 0; i < num_subspaces_; ++i) {
      distance_tables_[i] =
          ColMajorMatrix<score_type>(num_clusters_, num_clusters_);
    }

    size_t max_local_iters_taken = 0;
    size_t min_local_conv = std::numeric_limits<double>::max();

    // This basically the same thing we do in ivf_flat, but we perform it
    // num_subspaces_ times, once for each subspace.
    // @todo IMPORTANT This is highly suboptimal and will make multiple passes
    // through the training set.  We need to move iteration over subspaces to
    // the inner loop -- and SIMDize it
    for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      auto sub_begin = subspace * dimension_ / num_subspaces_;
      auto sub_end = (subspace + 1) * dimension_ / num_subspaces_;

      auto local_sub_distance = SubDistance{sub_begin, sub_end};

      // @todo Make choice of kmeans init configurable
      sub_kmeans_random_init(
          training_set, cluster_centroids_, sub_begin, sub_end, 0xdeadbeef);

      // sub_kmeans will invoke the sub_distance function with centroids
      // against new_centroids, and will call flat::qv_partition with centroids
      // and training_set which will invoke the sub_distance function with
      // centroids against training_set (though that can perhaps be reversed,
      // but will have to make sure asymmetric distance gets passed in).
      // operator()() is a function template, so it should do the "right thing"
      // @note we are doing this for one subspace at a time
      auto&& [iters, conv] = sub_kmeans<
          std::remove_cvref_t<decltype(training_set)>,
          std::remove_cvref_t<decltype(cluster_centroids_)>,
          SubDistance>(
          training_set,
          cluster_centroids_,
          sub_begin,
          sub_end,
          num_clusters_,
          tol_,
          max_iter_,
          num_threads_);

      max_local_iters_taken = std::max(max_local_iters_taken, iters);
      min_local_conv = std::min(min_local_conv, conv);
    }

    // Create tables of distances storing distance between encoding keys,
    // one table for each subspace.  That is, distance_tables_[i](j, k) is
    // the distance between the jth and kth centroids in the ith subspace.
    // The distance between two encoded vectors is looked up using the
    // keys of the vectors in each subspace (summing up the results obtained
    // from each subspace).
    // @todo SIMDize with subspace iteration in inner loop
    for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      auto sub_begin = subspace * sub_dimension_;
      auto sub_end = (subspace + 1) * sub_dimension_;
      auto local_sub_distance = SubDistance{sub_begin, sub_end};

      for (size_t i = 0; i < num_clusters_; ++i) {
        for (size_t j = 0; j < num_clusters_; ++j) {
          auto sub_distance =
              local_sub_distance(cluster_centroids_[i], cluster_centroids_[j]);
          distance_tables_[subspace](i, j) = sub_distance;
        }
      }
    }

    return std::make_tuple(max_local_iters_taken, min_local_conv);
  }

  /***************************************************************************
   *
   * Distance functions for pq encoded vectors
   *
   ***************************************************************************/

  // @todo IMPORTANT: We need to do some abstraction penalty tests to make sure
  // that the distance functions are inlined.
  // @todo Make this SIMD friendly -- do multiple subspaces at a time
  // For each (i, j), distances should be stored contiguously
  float sub_distance_symmetric(auto&& a, auto&& b) const {
    float pq_distance = 0.0;
    for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      auto i = a[subspace];
      auto j = b[subspace];

      auto sub_distance = distance_tables_[subspace](i, j);
      pq_distance += sub_distance;
    }
    return pq_distance;
  }

  auto make_pq_distance_symmetric() const {
    using A = decltype(pq_storage_type{}[0]);
    using B = decltype(pq_storage_type{}[0]);
    struct pq_distance {
      const ivf_pq_index* outer_;
      inline float operator()(const A& a, const B& b) {
        return outer_->sub_distance_symmetric(a, b);
      }
    };
    return pq_distance{this};
  }

  /**
   * @brief Uncompress the b and compute the distance between a and b
   * @param a The uncompressed vector
   * @param b The compressed vector
   * @tparam U The type of a, a feature vector
   * @tparam V The type of b, a compressed feature vector, i.e., a vector of
   * code types
   * @return The distance between a and b
   * @todo There is likely a copy constructor of the Distance functor.  That
   * should be checked and possibly fixed so that there is just a reference to
   * an existing object.
   * @todo This also needs to be SIMDized.
   */
  template <
      feature_vector U,
      feature_vector V,
      class Distance = sub_sum_of_squares_distance>
  float sub_distance_asymmetric(const U& a, const V& b) const {
    float pq_distance = 0.0;
    auto local_distance = Distance{};

    for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      auto sub_begin = subspace * sub_dimension_;
      auto sub_end = (subspace + 1) * sub_dimension_;
      auto i = b[subspace];

      pq_distance +=
          local_distance(a, cluster_centroids_[i], sub_begin, sub_end);
    }

    return pq_distance;
  }

  auto make_pq_distance_asymmetric() {
    using A = std::span<feature_type>;  // @todo: Don't hardcode span
    using B = decltype(pq_storage_type{}[0]);

    // @todo Do we need to worry about function call overhead here?
    struct pq_distance {
      const ivf_pq_index* outer_;
      inline float operator()(const A& a, const B& b) {
        return outer_->sub_distance_asymmetric(a, b);
      }
    };
    return pq_distance{this};
  }

  /**
   * @brief Initialize the centroids that we will use for building IVF index.
   */
  template <feature_vector_array V>
  void kmeans_random_init(const V& training_set) {
    ::kmeans_random_init(training_set, cluster_centroids_, num_partitions_);
  }

  /**
   * @brief Initialize the centroids that we will use for building IVF index.
   */
  template <feature_vector_array V, class Distance = sum_of_squares_distance>
  void kmeans_pp(const V& training_set) {
    ::kmeans_pp<std::remove_cvref_t<V>, decltype(cluster_centroids_), Distance>(
        training_set, cluster_centroids_, num_partitions_, num_threads_);
  }

  /**
   * Compute `num_partitions` centroids of the training set data, using the
   * kmeans algorithm. The initialization algorithm used to generate the
   * starting centroids for kmeans is specified by the `init` parameter.
   * Either random initialization or kmeans++ initialization can be used.
   *
   * @param training_set Array of vectors to cluster.
   * @param init Specify which initialization algorithm to use,
   * random (`random`) or kmeans++ (`kmeanspp`).
   */
  template <feature_vector_array V, class Distance = sum_of_squares_distance>
  void train_ivf(
      const V& training_set, kmeans_init init = kmeans_init::random) {
    dimension_ = ::dimension(training_set);
    if (num_partitions_ == 0) {
      num_partitions_ = std::sqrt(num_vectors(training_set));
    }

    flat_ivf_centroids_ =
        flat_ivf_centroid_storage_type(dimension_, num_partitions_);

    switch (init) {
      case (kmeans_init::none):
        break;
      case (kmeans_init::kmeanspp):
        kmeans_pp<std::remove_cvref_t<decltype(training_set)>, Distance>(
            training_set);
        break;
      case (kmeans_init::random):
        kmeans_random_init(training_set);
        break;
    };

    train_no_init<
        std::remove_cvref_t<decltype(training_set)>,
        std::remove_cvref<decltype(flat_ivf_centroids_)>,
        Distance>(
        training_set,
        flat_ivf_centroids_,
        dimension_,
        num_partitions_,
        max_iter_,
        tol_,
        num_threads_,
        reassign_ratio_);
  }

  /**
   * @brief Build the index from a training set, given the centroids.  This
   * will partition the training set into a contiguous array, with one
   * partition per centroid.  It will also create an array to record the
   * original ids locations of each vector (their locations in the original
   * training set) as well as a partitioning index array demarcating the
   * boundaries of each partition (including the very end of the array).
   *
   * @param training_set Array of vectors to partition.
   *
   * @todo Create and write index that is larger than RAM
   */
  template <feature_vector_array V>
  void add(const V& training_set) {
    auto num_unique_labels = ::num_vectors(flat_ivf_centroids_);

#if ONE_WAY  // -- through training set three times
    // Partition based on unencoded centroids and unencoded training set
    auto partition_labels = detail::flat::qv_partition(
        flat_ivf_centroids_, training_set, num_threads_, distance);

    // PQ Encode: -> cluster_centroids_, distance_tables_
    // Encode: training_set -> unpartitioned_pq_vectors_
    encode();
#else  // ANOTHER WAY -- through training set twice -- maybe only once if we can
       // combine pq_encode and encode
    train_pq(training_set);   // cluster_centroids_, distance_tables_
    train_ivf(training_set);  // flat_ivf_centroids_
    encode();                 // unpartitioned_pq_vectors, pq_ivf_centroids
    auto partition_labels = detail::flat::qv_partition(
        pq_ivf_centroids_,
        unpartitioned_pq_vectors_,
        num_threads_,
        make_pq_distance_symmetric());
#endif
    // This just reorders based on partition_labels
    partitioned_pq_vectors_ = std::make_unique<pq_storage_type>(
        unpartitioned_pq_vectors_, partition_labels, num_unique_labels);
  }

  /**
   * @brief PQ encode the training set using the cluster_centroids_ to get
   * unpartitioned_pq_vectors_.  PQ encode the flat_ivf_centroids_ to get
   * pq_ivf_centroids_.
   *
   * @return
   */
  template <feature_vector_array V>
  auto encode(const V& training_set) {
    // unpartitioned_pq_vectors_ :
  }

  template <
      feature_vector V,
      feature_vector W,
      class SubDistance = sub_sum_of_squares_distance>
    requires uncached_sub_distance_function<
        SubDistance,
        V,
        decltype(cluster_centroids_[0])>
  inline auto encode(const V& v, W& pq) const {
    auto local_sub_distance = SubDistance{};

    for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      auto sub_begin = sub_dimension_ * subspace;
      auto sub_end = sub_begin + sub_dimension_;

      auto min_score = std::numeric_limits<score_type>::max();
      pq_code_type idx{0};
      for (size_t i = 0; i < num_vectors(cluster_centroids_); ++i) {
        auto score =
            local_sub_distance(v, cluster_centroids_[i], sub_begin, sub_end);
        if (score < min_score) {
          min_score = score;
          idx = i;
        }
      }
      pq[subspace] = idx;
    }
  }

  template <
      feature_vector_array V,
      class SubDistance = cached_sub_sum_of_squares_distance>
    requires cached_sub_distance_function<
        SubDistance,
        typename V::span_type,
        decltype(cluster_centroids_[0])>
  auto encode(const V& v) {
    /*
     * Encode the training set using the cluster_centroids_ to get the
     * unpartitioned_pq_vectors_.
     */
    unpartitioned_pq_vectors_ =
        flat_storage_type(num_subspaces_, num_vectors(v));
    for (size_t i = 0; i < num_vectors(v); ++i) {
      auto x = unpartitioned_pq_vectors_[i];
      encode<
          typename V::span_type,
          decltype(unpartitioned_pq_vectors_[0]),
          SubDistance>(v[i], x);
    }

    /*
     * Encode the flat_ivf_centroids_ to get the pq_ivf_centroids_.
     */
    pq_ivf_centroids_ =
        pq_ivf_centroid_storage_type(num_subspaces_, num_partitions_);
    for (size_t i = 0; i < num_partitions_; ++i) {
      auto x = pq_ivf_centroids_[i];
      encode<
          decltype(cluster_centroids_[0]),
          decltype(pq_ivf_centroids_[0]),
          SubDistance>(cluster_centroids_[i], x);
    }
  }

  /*****************************************************************************
   * Methods for reading and reading the index from a group.
   *****************************************************************************/

  /**
   * @brief Read the the complete index arrays into ("infinite") memory.
   * This will read the centroids, indices, partitioned_ids, and
   * and the complete set of partitioned_vectors, along with metadata
   * from a group_uri.
   *
   * @param group_uri
   * @return bool indicating success or failure of read
   */
  auto read_index_infinite() {
    if (partitioned_vectors_ &&
        (::num_vectors(*partitioned_vectors_) != 0 ||
         ::num_vectors(partitioned_vectors_->ids()) != 0)) {
      throw std::runtime_error("Index already loaded");
    }

    // Load all partitions for infinite query
    // Note that the constructor will move the infinite_parts vector
    auto infinite_parts = std::vector<indices_type>(::num_vectors(centroids_));
    std::iota(begin(infinite_parts), end(infinite_parts), 0);
    partitioned_vectors_ = std::make_unique<tdb_storage_type>(
        group_->cached_ctx(),
        group_->parts_uri(),
        group_->indices_uri(),
        group_->get_num_partitions() + 1,
        group_->ids_uri(),
        infinite_parts,
        0,
        timestamp_);

    partitioned_vectors_->load();

    assert(
        ::num_vectors(*partitioned_vectors_) ==
        size(partitioned_vectors_->ids()));
    assert(
        size(partitioned_vectors_->indices()) == ::num_vectors(centroids_) + 1);
  }

  /**
   * @brief Open the index from the arrays contained in the group_uri.
   * The "finite" queries only load as much data (ids and vectors) as are
   * necessary for a given query -- so we can't load any data until we
   * know what the query is.  So, here we would have read the centroids and
   * indices into memory, when creating the index but would not have read
   * the partitioned_ids or partitioned_vectors.
   *
   * @param group_uri
   * @return bool indicating success or failure of read
   */
  template <feature_vector_array Q>
  auto read_index_finite(
      const Q& query_vectors, size_t nprobe, size_t upper_bound) {
    if (partitioned_vectors_ &&
        (::num_vectors(*partitioned_vectors_) != 0 ||
         ::num_vectors(partitioned_vectors_->ids()) != 0)) {
      throw std::runtime_error("Index already loaded");
    }

    auto&& [active_partitions, active_queries] =
        detail::ivf::partition_ivf_pq_index<indices_type>(
            centroids_, query_vectors, nprobe, num_threads_);

    partitioned_vectors_ = std::make_unique<tdb_storage_type>(
        group_->cached_ctx(),
        group_->parts_uri(),
        group_->indices_uri(),
        group_->get_num_partitions() + 1,
        group_->ids_uri(),
        active_partitions,
        upper_bound,
        timestamp_);

    // NB: We don't load the partitioned_vectors here.  We will load them
    // when we do the query.
    return std::make_tuple(
        std::move(active_partitions), std::move(active_queries));
  }

  /**
   * @brief Write the index to storage.  This would typically be done after a
   * set of input vectors has been read and a new group is created.  Or after
   * consolidation.
   *
   * We assume we have all of the data in memory, and that we are writing
   * all of it to a TileDB group.  Since we have all of it in memory,
   * we write from the PartitionedMatrix base class.
   *
   * @param group_uri
   * @param overwrite
   * @return bool indicating success or failure
   */
  auto write_index(
      const tiledb::Context& ctx,
      const std::string& group_uri,
      bool overwrite) const {
    tiledb::VFS vfs(ctx);

    // @todo Deal with this in right way vis a vis timestamping
    if (vfs.is_dir(group_uri)) {
      if (overwrite == false) {
        return false;
      }
      vfs.remove_dir(group_uri);
    }

    // Write the group
    auto write_group = ivf_pq_group(*this, ctx, group_uri, TILEDB_WRITE);

    write_group.set_dimension(dimension_);

    write_group.append_ingestion_timestamp(timestamp_);
    write_group.append_base_size(::num_vectors(*partitioned_vectors_));
    write_group.append_num_partitions(num_partitions_);

    write_matrix(
        ctx, centroids_, write_group.centroids_uri(), 0, false, timestamp_);

    write_matrix(
        ctx,
        *partitioned_vectors_,
        write_group.parts_uri(),
        0,
        false,
        timestamp_);

    write_vector(
        ctx,
        partitioned_vectors_->ids(),
        write_group.ids_uri(),
        0,
        false,
        timestamp_);

    write_vector(
        ctx,
        partitioned_vectors_->indices(),
        write_group.indices_uri(),
        0,
        false,
        timestamp_);

    return true;
  }

  auto write_index_arrays(
      const tiledb::Context& ctx,
      const std::string& centroids_uri,
      const std::string& parts_uri,
      const std::string& ids_uri,
      const std::string& indices_uri,
      bool overwrite) const {
    tiledb::VFS vfs(ctx);

    write_matrix(ctx, centroids_, centroids_uri, 0, true);
    write_matrix(ctx, *partitioned_vectors_, parts_uri, 0, true);
    write_vector(ctx, partitioned_vectors_->ids(), ids_uri, 0, true);
    write_vector(ctx, partitioned_vectors_->indices(), indices_uri, 0, true);

    return true;
  }

  /*****************************************************************************
   *
   * Queries, infinite and finite.
   *
   * An "infinite" query assumes there is enough RAM to load the entire array
   * of partitioned vectors into memory.  The query function then searches in
   * the appropriate partitions of the array for the query vectors.
   *
   * A "finite" query, on the other hand, examines the query and only loads
   * the partitions that are necessary for that particular search.  A finite
   * query also supports out of core operation, meaning that only a subset of
   * the necessary partitions are loaded into memory at any one time.  The
   * query is applied to each subset until all of the necessary partitions to
   * satisfy the query have been read in . The number of partitions to be held
   * in memory is controlled by an upper bound parameter that the user can set.
   * The upper bound limits the total number of vectors that will he held in
   * memory as the partitions are loaded.  Only complete partitions are loaded,
   * so the actual number of vectors in memory at any one time will generally
   * be less than the upper bound.
   *
   * @todo Add vq and dist queries (should dist be its own index?)
   * @todo Order queries so that partitions are queried in order
   *
   ****************************************************************************/

  /**
   * @brief Perform a query on the index, returning the nearest neighbors
   * and distances.  The function returns a matrix containing k_nn nearest
   * neighbors for each given query and a matrix containing the distances
   * corresponding to each returned neighbor.
   *
   * This function searches for the nearest neighbors using "infinite RAM",
   * that is, it loads the entire IVF index into memory and then applies the
   * query.
   *
   * @tparam Q Type of query vectors.
   * @param query_vectors Array of vectors to query.
   * @param k_nn Number of nearest neighbors to return.
   * @param nprobe Number of centroids to search.
   *
   * @return A tuple containing a matrix of nearest neighbors and a matrix
   * of the corresponding distances.
   *
   */
  template <feature_vector_array Q, class Distance = sum_of_squares_distance>
  auto query_infinite_ram(
      const Q& query_vectors,
      size_t k_nn,
      size_t nprobe,
      Distance distance = Distance{}) {
    if (!partitioned_vectors_ || ::num_vectors(*partitioned_vectors_) == 0) {
      read_index_infinite();
    }
    auto&& [active_partitions, active_queries] =
        detail::ivf::partition_ivf_pq_index<indices_type>(
            centroids_, query_vectors, nprobe, num_threads_);
    return detail::ivf::query_infinite_ram(
        *partitioned_vectors_,
        active_partitions,
        query_vectors,
        active_queries,
        k_nn,
        num_threads_,
        distance);
  }

  /**
   * @brief Same as query_infinite_ram, but using the qv_query_heap_infinite_ram
   * function.
   * See the documentation for that function in detail/ivf/qv.h
   * for more details.
   */
  template <feature_vector_array Q, class Distance = sum_of_squares_distance>
  auto qv_query_heap_infinite_ram(
      const Q& query_vectors,
      size_t k_nn,
      size_t nprobe,
      Distance distance = Distance{}) {
    if (!partitioned_vectors_ || ::num_vectors(*partitioned_vectors_) == 0) {
      read_index_infinite();
    }

    auto top_centroids = detail::ivf::ivf_top_centroids(
        centroids_, query_vectors, nprobe, num_threads_);
    return detail::ivf::qv_query_heap_infinite_ram(
        top_centroids,
        *partitioned_vectors_,
        query_vectors,
        nprobe,
        k_nn,
        num_threads_,
        distance);
  }

  /**
   * @brief Same as query_infinite_ram, but using the
   * nuv_query_heap_infinite_ram function.
   * See the documentation for that function in detail/ivf/qv.h
   * for more details.
   */
  template <feature_vector_array Q, class Distance = sum_of_squares_distance>
  auto nuv_query_heap_infinite_ram(
      const Q& query_vectors,
      size_t k_nn,
      size_t nprobe,
      Distance distance = Distance{}) {
    if (!partitioned_vectors_ || ::num_vectors(*partitioned_vectors_) == 0) {
      read_index_infinite();
    }
    auto&& [active_partitions, active_queries] =
        detail::ivf::partition_ivf_pq_index<indices_type>(
            centroids_, query_vectors, nprobe, num_threads_);
    return detail::ivf::nuv_query_heap_infinite_ram(
        *partitioned_vectors_,
        active_partitions,
        query_vectors,
        active_queries,
        k_nn,
        num_threads_,
        distance);
  }

  /**
   * @brief Same as query_infinite_ram, but using the
   * nuv_query_heap_infinite_ram_reg_blocked function.
   * See the documentation for that function in detail/ivf/qv.h
   * for more details.
   */
  template <feature_vector_array Q, class Distance = sum_of_squares_distance>
  auto nuv_query_heap_infinite_ram_reg_blocked(
      const Q& query_vectors,
      size_t k_nn,
      size_t nprobe,
      Distance distance = Distance{}) {
    if (!partitioned_vectors_ || ::num_vectors(*partitioned_vectors_) == 0) {
      read_index_infinite();
    }
    auto&& [active_partitions, active_queries] =
        detail::ivf::partition_ivf_pq_index<indices_type>(
            centroids_, query_vectors, nprobe, num_threads_);
    return detail::ivf::nuv_query_heap_infinite_ram_reg_blocked(
        *partitioned_vectors_,
        active_partitions,
        query_vectors,
        active_queries,
        k_nn,
        num_threads_,
        distance);
  }

  // WIP
#if 0
  template <feature_vector_array Q, class Distance = sum_of_squares_distance>
  auto qv_query_heap_finite_ram(
      const Q& query_vectors,
      size_t k_nn,
      size_t nprobe,
      size_t upper_bound = 0, Distance distance = Distance{}) {
    if (partitioned_vectors_ && ::num_vectors(*partitioned_vectors_) != 0) {
      std::throw_with_nested(
          std::runtime_error("Vectors are already loaded. Cannot load twice. "
                             "Cannot do finite query on in-memory index."));
    }
    auto&& [active_partitions, active_queries] =
        read_index_finite(query_vectors, nprobe, upper_bound);

    return detail::ivf::qv_query_heap_finite_ram(
        centroids_,
        *partitioned_vectors_,
        query_vectors,
        active_queries,
        nprobe,
        k_nn,
        upper_bound,
        num_threads_, distance);
  }
#endif  // 0

  /**
   * @brief Perform a query on the index, returning the nearest neighbors
   * and distances.  The function returns a matrix containing k_nn nearest
   * neighbors for each given query and a matrix containing the distances
   * corresponding to each returned neighbor.
   *
   * This function searches for the nearest neighbors using "finite RAM",
   * that is, it only loads that portion of the IVF index into memory that
   * is necessary for the given query.  In addition, it supports out of core
   * operation, meaning that only a subset of the necessary partitions are
   * loaded into memory at any one time.
   *
   * See the documentation for that function in detail/ivf/qv.h
   * for more details.
   *
   * @param query_vectors Array of vectors to query.
   * @param k_nn Number of nearest neighbors to return.
   * @param nprobe Number of centroids to search.
   *
   * @return A tuple containing a matrix of nearest neighbors and a matrix
   * of the corresponding distances.
   *
   * @tparam Q
   * @param query_vectors
   * @param k_nn
   * @param nprobe
   * @return
   */
  template <feature_vector_array Q, class Distance = sum_of_squares_distance>
  auto query_finite_ram(
      const Q& query_vectors,
      size_t k_nn,
      size_t nprobe,
      size_t upper_bound = 0,
      Distance distance = Distance{}) {
    if (partitioned_vectors_ && ::num_vectors(*partitioned_vectors_) != 0) {
      throw std::runtime_error(
          "Vectors are already loaded. Cannot load twice. "
          "Cannot do finite query on in-memory index.");
    }
    auto&& [active_partitions, active_queries] =
        read_index_finite(query_vectors, nprobe, upper_bound);

    return detail::ivf::query_finite_ram(
        *partitioned_vectors_,
        query_vectors,
        active_queries,
        k_nn,
        upper_bound,
        num_threads_,
        distance);
  }

  /**
   * @brief Same as query_finite_ram, but using the
   * nuv_query_heap_infinite_ram function.
   * See the documentation for that function in detail/ivf/qv.h
   * for more details.
   */
  template <feature_vector_array Q, class Distance = sum_of_squares_distance>
  auto nuv_query_heap_finite_ram(
      const Q& query_vectors,
      size_t k_nn,
      size_t nprobe,
      size_t upper_bound = 0,
      Distance distance = Distance{}) {
    if (partitioned_vectors_ && ::num_vectors(*partitioned_vectors_) != 0) {
      std::throw_with_nested(
          std::runtime_error("Vectors are already loaded. Cannot load twice. "
                             "Cannot do finite query on in-memory index."));
    }
    auto&& [active_partitions, active_queries] =
        read_index_finite(query_vectors, nprobe, upper_bound);

    return detail::ivf::nuv_query_heap_finite_ram(
        *partitioned_vectors_,
        query_vectors,
        active_queries,
        k_nn,
        upper_bound,
        num_threads_,
        distance);
  }

  /**
   * @brief Same as query_finite_ram, but using the
   * nuv_query_heap_infinite_ram_reg_blocked function.
   * See the documentation for that function in detail/ivf/qv.h
   * for more details.
   */
  template <feature_vector_array Q, class Distance = sum_of_squares_distance>
  auto nuv_query_heap_finite_ram_reg_blocked(
      const Q& query_vectors,
      size_t k_nn,
      size_t nprobe,
      size_t upper_bound = 0,
      Distance distance = Distance{}) {
    if (partitioned_vectors_ && ::num_vectors(*partitioned_vectors_) != 0) {
      std::throw_with_nested(
          std::runtime_error("Vectors are already loaded. Cannot load twice. "
                             "Cannot do finite query on in-memory index."));
    }
    auto&& [active_partitions, active_queries] =
        read_index_finite(query_vectors, nprobe, upper_bound);

    return detail::ivf::nuv_query_heap_finite_ram_reg_blocked(
        *partitioned_vectors_,
        query_vectors,
        active_queries,
        k_nn,
        upper_bound,
        num_threads_,
        distance);
  }

  /***************************************************************************
   * Getters (copilot weirded me out again -- it suggested "Getters" based
   * only on the two character "/ *" that I typed to begin a comment,
   * and with the two functions below.)
   * Note that we don't have a `num_vectors` because it isn't clear what
   * that means for a partitioned (possibly out-of-core) index.
   ***************************************************************************/
  auto dimension() const {
    return dimension_;
  }

  auto num_partitions() const {
    return ::num_vectors(centroids_);
  }

  /***************************************************************************
   * Methods to aid Testing and Debugging
   *
   * @todo -- As elsewhere in this class, there is huge code duplication here
   *
   **************************************************************************/

  /**
   * @brief Compare groups associated with two ivf_pq_index objects for
   * equality.  Note that both indexes will have had to perform a read or
   * a write.  An index created from partitioning will not yet have a group
   * associated with it.
   *
   * Comparing groups will also compare metadata associated with each group.
   *
   * @param rhs the index against which to compare
   * @return bool indicating equality of the groups
   */
  bool compare_group(const ivf_pq_index& rhs) const {
    return group_->compare_group(*(rhs.group_));
  }

  /**
   * @brief Compare metadata associated with two ivf_pq_index objects for
   * equality.  Thi is not the same as the metadata associated with the index
   * group.  Rather, it is the metadata associated with the index itself and is
   * only a small number of cached quantities.
   *
   * Note that `max_iter` et al are only relevant for partitioning an index
   * and are not stored (and would not be meaningful to compare at any rate).
   *
   * @param rhs the index against which to compare
   * @return bool indicating equality of the index metadata
   */
  bool compare_cached_metadata(const ivf_pq_index& rhs) {
    if (dimension_ != rhs.dimension_) {
      return false;
    }
    if (num_partitions_ != rhs.num_partitions_) {
      return false;
    }

    return true;
  }

  /**
   * @brief Compare two `feature_vector_arrays` for equality
   *
   * @tparam L Type of the lhs `feature_vector_array`
   * @tparam R Type of the rhs `feature_vector_array`
   * @param rhs the index against which to compare
   * @param lhs The lhs `feature_vector_array`
   * @return bool indicating equality of the `feature_vector_arrays`
   */
  template <feature_vector_array L, feature_vector_array R>
  auto compare_feature_vector_arrays(const L& lhs, const R& rhs) const {
    if (::num_vectors(lhs) != ::num_vectors(rhs) ||
        ::dimension(lhs) != ::dimension(rhs)) {
      std::cout << "num_vectors(lhs) != num_vectors(rhs) || dimension(lhs) != "
                   "dimension(rhs)n"
                << std::endl;
      std::cout << "num_vectors(lhs): " << ::num_vectors(lhs)
                << " num_vectors(rhs): " << ::num_vectors(rhs) << std::endl;
      std::cout << "dimension(lhs): " << ::dimension(lhs)
                << " dimension(rhs): " << ::dimension(rhs) << std::endl;
      return false;
    }
    for (size_t i = 0; i < ::num_vectors(lhs); ++i) {
      if (!std::equal(begin(lhs[i]), end(lhs[i]), begin(rhs[i]))) {
        std::cout << "lhs[" << i << "] != rhs[" << i << "]" << std::endl;
        std::cout << "lhs[" << i << "]: ";
        for (size_t j = 0; j < ::dimension(lhs); ++j) {
          std::cout << lhs[i][j] << " ";
        }
        std::cout << std::endl;
        std::cout << "rhs[" << i << "]: ";
        for (size_t j = 0; j < ::dimension(rhs); ++j) {
          std::cout << rhs[i][j] << " ";
        }
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Compare two `feature_vectors` for equality
   * @tparam L Type of the lhs `feature_vector`
   * @tparam R Type of the rhs `feature_vector`
   * @param lhs The lhs `feature_vector`
   * @param rhs The rhs `feature_vector`
   * @return
   */
  template <feature_vector L, feature_vector R>
  auto compare_vectors(const L& lhs, const R& rhs) const {
    if (::dimension(lhs) != ::dimension(rhs)) {
      std::cout << "dimension(lhs) != dimension(rhs) (" << ::dimension(lhs)
                << " != " << ::dimension(rhs) << ")" << std::endl;
      return false;
    }
    return std::equal(begin(lhs), end(lhs), begin(rhs));
  }

  /**
   * @brief Compare `centroids_` against another index
   * @param rhs The index against which to compare
   * @return bool indicating equality of the centroids
   */
  auto compare_centroids(const ivf_pq_index& rhs) {
    return compare_feature_vector_arrays(centroids_, rhs.centroids_);
  }

  /**
   * @brief Compare all of the `feature_vector_arrays` against another index
   * @param rhs The other index
   * @return bool indicating equality of the `feature_vector_arrays`
   */
  auto compare_feature_vectors(const ivf_pq_index& rhs) {
    if (partitioned_vectors_->num_vectors() !=
        rhs.partitioned_vectors_->num_vectors()) {
      std::cout << "partitioned_vectors_->num_vectors() != "
                   "rhs.partitioned_vectors_->num_vectors() ("
                << partitioned_vectors_->num_vectors()
                << " != " << rhs.partitioned_vectors_->num_vectors() << ")"
                << std::endl;
      return false;
    }
    if (partitioned_vectors_->num_partitions() !=
        rhs.partitioned_vectors_->num_partitions()) {
      std::cout << "partitioned_vectors_->num_parts() != "
                   "rhs.partitioned_vectors_->num_parts() ("
                << partitioned_vectors_->num_partitions()
                << " != " << rhs.partitioned_vectors_->num_partitions() << ")"
                << std::endl;
      return false;
    }

    return compare_feature_vector_arrays(
        *partitioned_vectors_, *(rhs.partitioned_vectors_));
  }

  /** Compare the stored `indices_` vector */
  auto compare_indices(const ivf_pq_index& rhs) {
    return compare_vectors(
        partitioned_vectors_->indices(), rhs.partitioned_vectors_->indices());
  }

  /** Compare the stored `partitioned_ids_` vector */
  auto compare_partitioned_ids(const ivf_pq_index& rhs) {
    return compare_vectors(
        partitioned_vectors_->ids(), rhs.partitioned_vectors_->ids());
  }

  auto set_centroids(const ColMajorMatrix<feature_type>& centroids) {
    centroids_ = ColMajorMatrix<centroid_feature_type>(
        ::dimension(centroids), ::num_vectors(centroids));
    std::copy(
        centroids.data(),
        centroids.data() + centroids.num_rows() * centroids.num_cols(),
        centroids_.data());
  }

  auto& get_centroids() {
    return centroids_;
  }

  /**
   * @brief Used for evaluating quality of partitioning
   * @param centroids
   * @param vectors
   * @return
   */
  static std::vector<indices_type> predict(
      const ColMajorMatrix<feature_type>& centroids,
      const ColMajorMatrix<feature_type>& vectors) {
    // Return a vector of indices of the nearest centroid for each vector in
    // the matrix. Write the code below:
    auto nClusters = centroids.num_cols();
    std::vector<indices_type> indices(vectors.num_cols());
    std::vector<score_type> distances(nClusters);
    for (size_t i = 0; i < vectors.num_cols(); ++i) {
      for (size_t j = 0; j < nClusters; ++j) {
        distances[j] = l2_distance(vectors[i], centroids[j]);
      }
      indices[i] =
          std::min_element(begin(distances), end(distances)) - begin(distances);
    }
    return indices;
  }

  void dump_group(const std::string& msg) {
    group_->dump(msg);
  }

  void dump_metadata(const std::string& msg) {
    group_->metadata.dump(msg);
  }
};

#endif  // TILEDB_PQ_INDEX_H
