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
 *   - cluster_centroids_ is a feature_vector_array of feature_type, dimensions_
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
 *   feature_vector_array of feature_type, dimensions_ x num_partitions_
 *   - uses ivf_flat::train
 *     - uses kmeans (just as it is used today)
 * - add
 *   - partition training set using ivf_centroids_ to get
 * partitioned_pq_vectors_
 *     - uses ivf_flat::add
 *       - uses partitioned_matrix constructor, which just shuffles
 * - pq train with partitioned_pq_vectors_ to get cluster_centroids_ and
 *   distance_tables_
 * - compress partitioned_pq_vectors_ and ivf_centroids_
 * - query with symmetric distance
 *
 * III. OR, train , compress, add, the steps would be:
 * - train using training_set to get ivf_centroids_, which is a
 *   feature_vector_array of feature_type, dimensions_ x num_partitions_
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
#include "detail/linalg/tdb_matrix_multi_range.h"
#include "detail/linalg/tdb_matrix_with_ids.h"

#include <tiledb/tiledb>
#include <type_traits>

/**
 * Class representing an inverted file (IVF) index for flat (non-compressed)
 * feature vectors. The class simply holds the index data itself, it is
 * unaware of where the data comes from -- reading and writing data is done
 * via an ivf_pq_group. Thus, this class does not hold information
 * about the group (neither the group members, nor the group metadata).
 *
 * @tparam partitioned_pq_vectors_feature_type
 * @tparam partitioned_ids_type
 * @tparam partitioning_index_type
 */
template <
    class partitioned_pq_vectors_feature_type,
    class partitioned_ids_type = uint64_t,
    class partitioning_indices_type = uint64_t>
class ivf_pq_index {
 public:
  using feature_type = partitioned_pq_vectors_feature_type;
  using id_type = partitioned_ids_type;
  using indices_type = partitioning_indices_type;
  using score_type = float;  // @todo -- this should be a parameter?

  using group_type = ivf_pq_group<ivf_pq_index>;
  using metadata_type = ivf_pq_metadata;

  // @todo IMPORTANT: Use a uint64_t to store 8 bytes together -- should make
  // loads and other operations faster and SIMD friendly
  using pq_code_type = uint8_t;

  using pq_vector_feature_type = pq_code_type;

  // The pq_centroids store the (unencoded) centroids for each subspace
  using flat_vector_feature_type = score_type;

 private:
  using flat_storage_type = ColMajorMatrix<pq_code_type>;

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
   *   - The pq ivf centroids - compressed version of the ivf centroids. These
   *     are used in queries with symmetric distance. They are also used in
   *     the compress / train / add pattern.
   *   - The cluster_centroids_ that partition each subspace of the training set
   *     into another set of partitions. There are num_clusters of these, with
   *     each centroid being of size dimensions_. These are used to build the
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

  /****************************************************************************
   * Index group information
   ****************************************************************************/
  std::string group_uri_;
  size_t upper_bound_{0};
  TemporalPolicy temporal_policy_;
  IndexLoadStrategy index_load_strategy_{IndexLoadStrategy::PQ_INDEX};
  std::unique_ptr<ivf_pq_group<ivf_pq_index>> group_;

  /****************************************************************************
   * Index representation
   ****************************************************************************/

  // Cached information about the partitioned vectors in the index
  uint64_t dimensions_{0};
  uint64_t num_partitions_{0};

  // Cached information about the pq encoding
  uint32_t num_subspaces_{0};
  uint32_t sub_dimensions_{0};

  // num_clusters_ is the number of centroids we will train for each subspace.
  // Because it's fixed at 256, it means that we can only have 256 centroids per
  // subspace. So each subspace will have 256 possible centroids.
  constexpr static const uint32_t num_clusters_{256};
  // We don't really need to store bits_per_subspace_, as num_clusters_=
  // 2^bits_per_subspace_. But doing so makes the code more readable.
  constexpr static const uint32_t bits_per_subspace_{8};

  // The feature vectors. These contain the original input vectors, modified
  // with updates and deletions over time. Note that we only use this to
  // re-ingest data, so if we open this index by URI we will not read this data
  // from the URI. Instead, we'll fill it when we call `add()` and then write it
  // during `write_index()`.
  ColMajorMatrixWithIds<feature_type, id_type> feature_vectors_;

  // This holds the centroids we have determined in train_ivf(). We will have
  // one column for each partition, and each of those columns is a centroid with
  // dimensions_ elements. Note that these trained from uncompressed vectors and
  // are the same as IVF_FLAT centroids.
  flat_ivf_centroid_storage_type flat_ivf_centroids_;

  // For each subspace we will run kmeans on that subspace. We will generate
  // num_clusters_ (256) centroids for each of those subspaces. We store this as
  // a matrix with num_clusters_ (256) columns, each representing a centroid.
  // Then we have dimensions_ rows - each row holds several centroids in it, one
  // for each subspace. So if we have a 16 dimensional set of vectors, and 2
  // subspaces, we will have 256 columns and 16 rows. The first 8 rows will hold
  // a centroid for the first subspace, and the second 8 rows will hold a
  // centroid for the second subspace.
  cluster_centroid_storage_type cluster_centroids_;

  std::unique_ptr<pq_storage_type> partitioned_pq_vectors_;

  // These are the original training vectors encoded using the
  // cluster_centroids_. So each vector has been chunked up into num_subspaces_
  // sections, and for each section we find the closest centroid from
  // cluster_centroids_ and appen that index as the next number in the
  // pq_vector.
  std::unique_ptr<ColMajorMatrix<pq_code_type>> unpartitioned_pq_vectors_;
  // Or should these just be
  // pq_storage_type partitioned_pq_vectors_;
  // flat_storage_type unpartitioned_pq_vectors_;

  // Parameters for performing kmeans clustering for the ivf index and pq
  // compression.
  uint32_t max_iterations_{0};
  float convergence_tolerance_{0.f};
  float reassign_ratio_{0.f};

  DistanceMetric distance_metric_{DistanceMetric::SUM_OF_SQUARES};

  // Some parameters for execution
  uint64_t num_threads_{std::thread::hardware_concurrency()};
  uint64_t seed_{std::random_device{}()};

 public:
  using value_type = feature_type;
  using index_type = partitioning_indices_type;  // @todo This isn't quite right

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
   * parameters to be used subsequently in training. To fully create an index
   * we will need to call `train()` and `add()`.
   *
   * @param partitions Number of centroids / partitions to compute.
   * @param num_subspaces Number of subspaces to use for pq compression. This is
   * the number of sections to divide the vector into.
   * @param max_iterations Maximum number of iterations for kmeans algorithm.
   * @param convergence_tolerance Convergence convergence_toleranceerance for
   * kmeans algorithm.
   * @param temporal_policy Temporal policy for the index.
   *
   * @note PQ encoding generally is described as having parameter nbits, how
   * many bits to use for indexing into the codebook. In real implementations,
   * this seems to always be 8 -- it doesn't make sense to be anything other
   * than that, else indexing would be too slow. Accordingly, we set it as
   * a constexpr value to 8 -- and correspondingly, we set num_clusters to 256.
   *
   * @todo Use chained parameter technique for arguments
   * @todo -- Need something equivalent to "None" since user could pass 0
   * @todo -- Or something equivalent to "Use current time" and something
   * to indicate "no time traveling"
   * @todo -- May also want start/stop?  Use a variant?  TemporalPolicy?
   */
  static void create(
      const tiledb::Context& ctx,
      const std::string& group_uri,
      uint64_t dimensions,
      uint32_t num_subspaces = 16,
      uint32_t max_iterations = 2,
      float convergence_tolerance = 0.000025f,
      float reassign_ratio = 0.075f,
      std::optional<TemporalPolicy> optional_temporal_policy = std::nullopt,
      DistanceMetric distance_metric = DistanceMetric::SUM_OF_SQUARES,
      const std::string& storage_version = "",
      uint64_t seed = std::random_device{}()) {
    if (num_subspaces <= 0) {
      throw std::runtime_error(
          "num_subspaces (" + std::to_string(num_subspaces) +
          ") must be greater than zero");
    }
    if (dimensions % num_subspaces != 0) {
      throw std::runtime_error(
          "Dimension must be divisible by the number of subspaces - "
          "dimensions: " +
          std::to_string(dimensions) +
          ", num_subspaces: " + std::to_string(num_subspaces));
    }
    if (num_clusters_ != 1 << bits_per_subspace_) {
      throw std::runtime_error(
          "[ivf_pq_index@write_index] num_clusters_ != 1 << "
          "bits_per_subspace_");
    }
    auto temporal_policy =
        optional_temporal_policy.has_value() ?
            *optional_temporal_policy :
            TemporalPolicy{
                TimeTravel,
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count())};

    auto sub_dimensions = dimensions / num_subspaces;
    auto write_group = ivf_pq_group<ivf_pq_index>(
        ctx,
        group_uri,
        TILEDB_WRITE,
        temporal_policy,
        storage_version,
        dimensions,
        num_clusters_,
        num_subspaces);
    write_group.set_dimensions(dimensions);
    write_group.set_num_subspaces(num_subspaces);
    write_group.set_sub_dimensions(sub_dimensions);
    write_group.set_bits_per_subspace(bits_per_subspace_);
    write_group.set_num_clusters(num_clusters_);
    write_group.set_max_iterations(max_iterations);
    write_group.set_convergence_tolerance(convergence_tolerance);
    write_group.set_reassign_ratio(reassign_ratio);
    write_group.set_distance_metric(distance_metric);

    write_group.append_ingestion_timestamp(0);
    write_group.append_base_size(0);
    write_group.append_num_partitions(0);

    write_group.store_metadata();
  }

  /**
   * @brief Create the temp data group.
   *
   * @param partial_write_array_dir The directory to write the temp data group.
   */
  void create_temp_data_group(const std::string& partial_write_array_dir) {
    auto write_group = ivf_pq_group<ivf_pq_index>(
        group_->cached_ctx(),
        group_uri_,
        TILEDB_WRITE,
        temporal_policy_,
        group_->storage_version(),
        dimensions_,
        num_clusters_,
        num_subspaces_);
    write_group.create_temp_data_group(partial_write_array_dir);
  }

  /**
   * @brief Open a previously created index, stored as a TileDB group. This
   * class does not deal with the group itself, but rather calls the group
   * constructor. The group constructor will initialize itself with information
   * about the different constituent arrays needed for operation of this class,
   * but will not initialize any member data of the class.
   *
   * The group is opened with a timestamp, so the correct values of base_size
   * and num_partitions will be set.
   *
   * We go ahead and load the centroids here. We defer reading anything else
   * because that will depend on the type of query we are doing as well as the
   * contents of the query itself.
   *
   * @todo Is this the right place to load the centroids?
   *
   * @param ctx
   * @param uri
   * @param temporal_policy
   *
   */
  ivf_pq_index(
      const tiledb::Context& ctx,
      const std::string& uri,
      IndexLoadStrategy index_load_strategy = IndexLoadStrategy::PQ_INDEX,
      size_t upper_bound = 0,
      std::optional<TemporalPolicy> temporal_policy = std::nullopt)
      : group_uri_{uri}
      , upper_bound_{upper_bound}
      , temporal_policy_{temporal_policy.has_value() ? *temporal_policy : TemporalPolicy()}
      , index_load_strategy_{index_load_strategy}
      , group_{std::make_unique<ivf_pq_group<ivf_pq_index>>(
            ctx, uri, TILEDB_READ, temporal_policy_)} {
    if (upper_bound != 0 && index_load_strategy_ != IndexLoadStrategy::PQ_OOC) {
      throw std::runtime_error(
          "With upper_bound > 0 you must use IndexLoadStrategy::PQ_OOC.");
    }
    if (upper_bound == 0 && index_load_strategy_ == IndexLoadStrategy::PQ_OOC) {
      throw std::runtime_error(
          "With IndexLoadStrategy::PQ_OOC you must have an upper_bound > 0.");
    }
    /**
     * Read the centroids. How the partitioned_pq_vectors_ are read in will be
     * determined by the type of query we are doing. But they will be read
     * in at this same timestamp.
     */
    dimensions_ = group_->get_dimensions();
    num_partitions_ = group_->get_num_partitions();
    num_subspaces_ = group_->get_num_subspaces();
    sub_dimensions_ = group_->get_sub_dimensions();
    max_iterations_ = group_->get_max_iterations();
    convergence_tolerance_ = group_->get_convergence_tolerance();
    reassign_ratio_ = group_->get_reassign_ratio();
    distance_metric_ = group_->get_distance_metric();

    flat_ivf_centroids_ =
        tdbPreLoadMatrix<flat_vector_feature_type, stdx::layout_left>(
            group_->cached_ctx(),
            group_->flat_ivf_centroids_uri(),
            std::nullopt,
            num_partitions_,
            0,
            temporal_policy_);

    cluster_centroids_ =
        tdbPreLoadMatrix<flat_vector_feature_type, stdx::layout_left>(
            group_->cached_ctx(),
            group_->cluster_centroids_uri(),
            std::nullopt,
            std::nullopt,
            num_clusters_,
            temporal_policy_);

    if (upper_bound == 0) {
      read_index_infinite();
    }
    if (index_load_strategy_ ==
        IndexLoadStrategy::PQ_INDEX_AND_RERANKING_VECTORS) {
      auto num_vectors = group_->get_base_size();
      feature_vectors_ = tdbColMajorPreLoadMatrixWithIds<feature_type, id_type>(
          group_->cached_ctx(),
          group_->feature_vectors_uri(),
          group_->ids_uri(),
          dimensions_,
          num_vectors,
          0,
          temporal_policy_);
    }
  }

  void read_index_infinite() {
    // Read the the complete index arrays into ("infinite") memory. This will
    // read the centroids, indices, partitioned_ids, and and the complete set
    // of partitioned_pq_vectors, along with metadata from a group_uri. Load
    // all partitions for infinite query.
    auto infinite_parts =
        std::vector<indices_type>(::num_vectors(flat_ivf_centroids_));
    std::iota(begin(infinite_parts), end(infinite_parts), 0);

    partitioned_pq_vectors_ = std::make_unique<tdb_pq_storage_type>(
        group_->cached_ctx(),
        group_->pq_ivf_vectors_uri(),
        group_->feature_vectors_index_uri(),
        group_->get_num_partitions() + 1,
        group_->ids_uri(),
        infinite_parts,
        0,
        temporal_policy_);

    partitioned_pq_vectors_->load();

    if (::num_vectors(*partitioned_pq_vectors_) !=
        size(partitioned_pq_vectors_->ids())) {
      throw std::runtime_error(
          "[ivf_flat_index@read_index_infinite] "
          "::num_vectors(*partitioned_pq_vectors_) != "
          "size(partitioned_pq_vectors_->ids())");
    }
    if (size(partitioned_pq_vectors_->indices()) !=
        ::num_vectors(flat_ivf_centroids_) + 1) {
      throw std::runtime_error(
          "[ivf_flat_index@read_index_infinite] "
          "size(partitioned_pq_vectors_->indices()) != "
          "::num_vectors(flat_ivf_centroids_) + 1");
    }
  }

  /****************************************************************************
   * Methods for building, writing, and reading the complete index. Includes:
   *   - Method for encoding the training set using pq compression to create
   *     the cluster_centroids_.
   *   - Method to initialize the centroids that we will use for building the
   *IVF index.
   *   - Method to partition the pq_vectors_ into a partitioned_matrix of pq
   *encoded vectors.
   * @note With this approach, we are partitioning based on cluster_centroids_
   *the stored centroids are also encoded using pq. Thus we can do our search
   *using the symmetric distance function.
   *
   * @todo Create single function that trains and adds (ingests)
   * @todo Provide interface that takes URI rather than vectors
   * @todo Provide "kernel" interface for use in distributed computation
   * @todo Do we need an out-of-core version of this?
   ****************************************************************************/

  /**
   * @brief Create the `cluster_centroids_` (encoded from the training set).
   *
   * @tparam V type of the training vectors
   * @tparam SubDistance type of the distance function to use for encoding.
   * Must be a cached_sub_distance_function.
   * @param training_set The set of vectors to compress
   *
   * @note This is essentially the same as flat_pq_index::train
   * @note Recall that centroids_ are used for IVF indexing. cluster_centroids_
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
  auto train_pq(const V& training_set, kmeans_init init = kmeans_init::random) {
    scoped_timer _{"ivf_pq_index@train_pq"};

    // This basically the same thing we do in ivf_flat, but we perform it
    // num_subspaces_ times, once for each subspace.
    // @todo IMPORTANT This is highly suboptimal and will make multiple passes
    // through the training set. We need to move iteration over subspaces to
    // the inner loop -- and SIMDize it
    cluster_centroids_ =
        ColMajorMatrix<flat_vector_feature_type>(dimensions_, num_clusters_);
    for (uint32_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      auto sub_begin = subspace * dimensions_ / num_subspaces_;
      auto sub_end = (subspace + 1) * dimensions_ / num_subspaces_;

      // @todo Make choice of kmeans init configurable
      sub_kmeans_random_init(
          training_set, cluster_centroids_, sub_begin, sub_end);

      // sub_kmeans will invoke the sub_distance function with centroids
      // against new_centroids, and will call flat::qv_partition with centroids
      // and training_set which will invoke the sub_distance function with
      // centroids against training_set (though that can perhaps be reversed,
      // but will have to make sure asymmetric distance gets passed in).
      // operator()() is a function template, so it should do the "right thing"
      // @note we are doing this for one subspace at a time
      sub_kmeans<
          std::remove_cvref_t<decltype(training_set)>,
          std::remove_cvref_t<decltype(cluster_centroids_)>,
          SubDistance>(
          training_set,
          cluster_centroids_,
          sub_begin,
          sub_end,
          num_clusters_,
          convergence_tolerance_,
          max_iterations_,
          num_threads_);
    }
  }

  /***************************************************************************
   *
   * Distance functions for pq encoded vectors
   *
   ***************************************************************************/
  /**
   * @brief Computes the distance between a query and and a pq_encoded_vector
   * given the distance table of the query to the pq_centroids
   * query_to_pq_centroid_distance_table.
   *
   * @param query_to_pq_centroid_distance_table Distance table of the query
   * vector to the pq_centroids.
   * @param pq_encoded_vector PQ encoded database vector.
   *
   */
  template <feature_vector U, feature_vector V>
  float sub_distance_query_to_pq_centroid_distance_tables(
      const U& query_to_pq_centroid_distance_table,
      const V& pq_encoded_vector) const {
    float pq_distance = 0.0;
    size_t sub_id = 0;
    for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      auto j = pq_encoded_vector[subspace];
      pq_distance += query_to_pq_centroid_distance_table[sub_id + j];
      sub_id += num_clusters_;
    }
    return pq_distance;
  }

  template <
      typename query_to_pq_centroid_distance_tables_type,
      typename index_feature_type>
  auto make_pq_distance_query_to_pq_centroid_distance_tables() const {
    using A = query_to_pq_centroid_distance_tables_type;
    using B = index_feature_type;

    struct pq_distance {
      const ivf_pq_index* outer_;
      inline float operator()(const A& a, const B& b) {
        return outer_->sub_distance_query_to_pq_centroid_distance_tables(a, b);
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
   * @todo There is likely a copy constructor of the Distance functor. That
   * should be checked and possibly fixed so that there is just a reference to
   * an existing object.
   * @todo This also needs to be SIMDized.
   */
  template <
      feature_vector U,
      feature_vector V,
      class Distance = uncached_sub_sum_of_squares_distance>
  float sub_distance_asymmetric(const U& a, const V& b) const {
    float pq_distance = 0.0;
    auto local_distance = Distance{};

    for (uint32_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      auto sub_begin = subspace * sub_dimensions_;
      auto sub_end = (subspace + 1) * sub_dimensions_;
      auto i = b[subspace];

      pq_distance +=
          local_distance(a, cluster_centroids_[i], sub_begin, sub_end);
    }

    return pq_distance;
  }

  // @todo Parameterize by Distance
  template <typename queries_feature_type, typename index_feature_type>
  auto make_pq_distance_asymmetric() const {
    using A = queries_feature_type;
    using B = index_feature_type;

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
    ::kmeans_random_init(training_set, flat_ivf_centroids_, num_partitions_);
  }

  /**
   * @brief Initialize the centroids that we will use for building IVF index.
   */
  template <feature_vector_array V, class Distance = sum_of_squares_distance>
  void kmeans_pp(const V& training_set) {
    ::kmeans_pp<
        std::remove_cvref_t<V>,
        decltype(flat_ivf_centroids_),
        Distance>(
        training_set, flat_ivf_centroids_, num_partitions_, num_threads_);
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
    scoped_timer _{"ivf_pq_index@train_ivf"};
    dimensions_ = ::dimensions(training_set);
    if (num_partitions_ == 0) {
      num_partitions_ = std::sqrt(::num_vectors(training_set));
    }

    flat_ivf_centroids_ =
        flat_ivf_centroid_storage_type(dimensions_, num_partitions_);

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
        decltype(flat_ivf_centroids_),
        Distance>(
        training_set,
        flat_ivf_centroids_,
        dimensions_,
        num_partitions_,
        max_iterations_,
        convergence_tolerance_,
        num_threads_,
        reassign_ratio_);
  }

  /**
   * @brief Trains the index.
   *
   * @param training_set Array of vectors to partition.
   * @param training_set_ids IDs for each vector.
   */
  template <
      feature_vector_array Array,
      class Distance = sum_of_squares_distance>
  void train(
      const Array& training_set,
      size_t num_partitions = 0,
      std::optional<TemporalPolicy> temporal_policy = std::nullopt,
      Distance distance = Distance{}) {
    scoped_timer _{"ivf_pq_index@train"};
    if (num_subspaces_ <= 0) {
      throw std::runtime_error(
          "num_subspaces (" + std::to_string(num_subspaces_) +
          ") must be greater than zero");
    }

    if (temporal_policy.has_value()) {
      temporal_policy_ = *temporal_policy;
    }

    if (num_partitions != 0) {
      num_partitions_ = num_partitions;
    }
    if (num_partitions_ == 0) {
      num_partitions_ = std::sqrt(::num_vectors(training_set));
    }

    // 1. Fill in cluster_centroids_.
    // cluster_centroids_ holds the num_clusters_ (256) centroids for each
    // subspace.
    train_pq(training_set);

    // 2. Fill in flat_ivf_centroids_.
    // We compute num_partitions_ centroids and store in flat_ivf_centroids_.
    // This is the same as IVF_FLAT and has nothing to do with PQ.
    train_ivf(training_set);

    // Need so that we can keep using the index without re-opening the group.
    group_->set_num_partitions(num_partitions_);

    // 3. Update our metadata.
    auto write_group = ivf_pq_group<ivf_pq_index>(
        group_->cached_ctx(),
        group_uri_,
        TILEDB_WRITE,
        temporal_policy_,
        group_->storage_version(),
        dimensions_,
        num_clusters_,
        num_subspaces_);

    if (write_group.get_all_ingestion_timestamps().size() == 1 &&
        write_group.get_previous_ingestion_timestamp() == 0 &&
        write_group.get_all_base_sizes().size() == 1 &&
        write_group.get_previous_base_size() == 0) {
      write_group.set_ingestion_timestamp(temporal_policy_.timestamp_end());
      write_group.set_base_size(0);
      write_group.set_num_partitions(num_partitions_);
    } else {
      write_group.append_ingestion_timestamp(temporal_policy_.timestamp_end());
      write_group.append_base_size(0);
      write_group.append_num_partitions(num_partitions_);
    }

    write_group.store_metadata();

    // 4. Write the centroids.
    write_matrix(
        group_->cached_ctx(),
        cluster_centroids_,
        write_group.cluster_centroids_uri(),
        0,
        false,
        temporal_policy_);

    write_matrix(
        group_->cached_ctx(),
        flat_ivf_centroids_,
        write_group.flat_ivf_centroids_uri(),
        0,
        false,
        temporal_policy_);

    // 5. Clear partitioned_pq_vectors_ so that we re-load it when we next
    // query().
    partitioned_pq_vectors_ = nullptr;
  }

  /**
   * @brief Build the index from a training set, given the centroids. This
   * will partition the training set into a contiguous array, with one
   * partition per centroid. It will also create an array to record the
   * original ids locations of each vector (their locations in the original
   * training set) as well as a partitioning index array demarcating the
   * boundaries of each partition (including the very end of the array).
   * Note that we will write using the temporal_policy set in train(), and if
   * that was not set, in the construtor.
   *
   * @param training_set Array of vectors to partition.
   * @param training_set_ids IDs for each vector.
   * @param deleted_ids IDs of vectors to delete.
   * @param start The starting index of the training set to ingest.
   * @param end The ending index of the training set to ingest.
   * @param partition_start The starting index of the partitioned vectors to
   * write.
   * @param partial_write_array_dir The directory to write the temp arrays. If
   * not set, we will write to the main arrays instead of the temp arrays.
   * @param distance The distance function to use.
   * @param write_to_temp_arrays Whether to write to the temp arrays. True if
   * ingesting via Python.
   */
  template <
      feature_vector_array Array,
      class Distance = sum_of_squares_distance>
  void ingest_parts(
      const Array& training_set,
      const std::span<partitioned_ids_type>& training_set_ids,
      const std::span<partitioned_ids_type>& deleted_ids,
      size_t start,
      size_t end,
      size_t partition_start,
      const std::string& partial_write_array_dir = "",
      Distance distance = Distance{}) {
    // 1. pq-encode the vectors.
    // This results in a matrix where we have num_vectors(training_set) columns,
    // and num_subspaces_ rows. So if we had 10 vectors, each with 16
    // dimensions, and 2 num_subspaces_, we store 16 columns and 2 rows. Note
    // that we don't actually need this as a member variable, but do so for unit
    // tests.
    unpartitioned_pq_vectors_ =
        pq_encode<Array, ColMajorMatrix<pq_code_type>>(training_set);

    // 2. Write the partitioned original vectors and the partitioned pq-encoded
    // vectors.
    auto write_group = ivf_pq_group<ivf_pq_index>(
        group_->cached_ctx(),
        group_uri_,
        TILEDB_WRITE,
        temporal_policy_,
        group_->storage_version(),
        dimensions_,
        num_clusters_,
        num_subspaces_);
    detail::ivf::ivf_pq_index<
        Array,
        partitioned_pq_vectors_feature_type,
        pq_code_type,
        partitioned_ids_type,
        partitioning_indices_type,
        flat_vector_feature_type>(
        group_->cached_ctx(),
        training_set,
        *unpartitioned_pq_vectors_,
        training_set_ids,
        deleted_ids,
        flat_ivf_centroids_,
        partial_write_array_dir.empty() ?
            write_group.feature_vectors_uri() :
            write_group.feature_vectors_temp_uri(partial_write_array_dir),
        partial_write_array_dir.empty() ?
            write_group.feature_vectors_index_uri() :
            write_group.feature_vectors_index_temp_uri(partial_write_array_dir),
        partial_write_array_dir.empty() ?
            write_group.ids_uri() :
            write_group.ids_temp_uri(partial_write_array_dir),
        partial_write_array_dir.empty() ?
            write_group.pq_ivf_vectors_uri() :
            write_group.pq_ivf_vectors_temp_uri(partial_write_array_dir),
        start,
        end,
        num_threads_,
        temporal_policy_,
        partition_start);
  }

  /**
   * @brief Consolidate partitions. This will read the partitioned vectors and
   * ids from the temp arrays, and write them to the main arrays.
   *
   * @param partitions Number of partitions.
   * @param work_items Number of work items.
   * @param partition_id_start The starting partition id.
   * @param partition_id_end The ending partition id.
   * @param batch The batch size.
   * @param partial_write_array_dir The directory to write the temp arrays.
   */
  void consolidate_partitions(
      size_t partitions,
      size_t work_items,
      size_t partition_id_start,
      size_t partition_id_end,
      size_t batch,
      const std::string& partial_write_array_dir) {
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> partition_slices(
        partitions);

    auto total_partitions = work_items * (partitions + 1);
    std::vector<partitioning_indices_type> partial_indexes =
        read_vector<partitioning_indices_type>(
            group_->cached_ctx(),
            group_->feature_vectors_index_temp_uri(partial_write_array_dir),
            0,
            total_partitions,
            temporal_policy_);

    std::vector<partitioning_indices_type> index_array =
        read_vector<partitioning_indices_type>(
            group_->cached_ctx(),
            group_->feature_vectors_index_uri(),
            0,
            total_partitions,
            temporal_policy_);

    size_t i = 0;
    uint64_t prev_index = 0;
    for (size_t work_item_id = 0; work_item_id < work_items; ++work_item_id) {
      prev_index = partial_indexes[i];
      i++;
      for (size_t partition_id = 0; partition_id < partitions; ++partition_id) {
        auto slice = std::make_pair(
            static_cast<int>(prev_index),
            static_cast<int>(partial_indexes[i]) - 1);
        if (slice.first <= slice.second &&
            slice.first != std::numeric_limits<uint64_t>::max()) {
          partition_slices[partition_id].push_back(slice);
        }
        prev_index = partial_indexes[i];
        i++;
      }
    }

    for (size_t part = partition_id_start; part < partition_id_end;
         part += batch) {
      size_t part_end = std::min(part + batch, partition_id_end);

      std::vector<std::pair<uint64_t, uint64_t>> read_slices;
      size_t total_slices_size = 0;
      for (size_t p = part; p < part_end; ++p) {
        for (const auto& partition_slice : partition_slices[p]) {
          read_slices.push_back(partition_slice);
          total_slices_size +=
              partition_slice.second - partition_slice.first + 1;
        }
      }

      // Read from index array.
      uint64_t start_pos = index_array[part];
      uint64_t end_pos = index_array[part_end];

      if (read_slices.empty()) {
        if (start_pos != end_pos) {
          throw std::runtime_error("Incorrect partition size.");
        }
        continue;
      }
      // Read data.
      std::vector<partitioned_ids_type> ids = read_vector<partitioned_ids_type>(
          group_->cached_ctx(),
          group_->ids_temp_uri(partial_write_array_dir),
          read_slices,
          total_slices_size,
          temporal_policy_);

      auto vectors = tdbColMajorMatrixMultiRange<feature_type, uint64_t>(
          group_->cached_ctx(),
          group_->feature_vectors_temp_uri(partial_write_array_dir),
          dimensions_,
          read_slices,
          total_slices_size,
          0,
          temporal_policy_);
      vectors.load();

      auto pq_vectors = tdbColMajorMatrixMultiRange<pq_code_type, uint64_t>(
          group_->cached_ctx(),
          group_->pq_ivf_vectors_temp_uri(partial_write_array_dir),
          num_subspaces_,
          read_slices,
          total_slices_size,
          0,
          temporal_policy_);
      pq_vectors.load();

      // Write data to the arrays.
      write_vector(
          group_->cached_ctx(),
          ids,
          group_->ids_uri(),
          start_pos,
          false,
          temporal_policy_);

      write_vector(
          group_->cached_ctx(),
          ids,
          group_->ids_uri(),
          start_pos,
          false,
          temporal_policy_);

      write_matrix(
          group_->cached_ctx(),
          vectors,
          group_->feature_vectors_uri(),
          start_pos,
          false,
          temporal_policy_);

      write_matrix(
          group_->cached_ctx(),
          pq_vectors,
          group_->pq_ivf_vectors_uri(),
          start_pos,
          false,
          temporal_policy_);
    }

    // Now update the metadata.
    auto write_group = ivf_pq_group<ivf_pq_index>(
        group_->cached_ctx(),
        group_uri_,
        TILEDB_WRITE,
        temporal_policy_,
        group_->storage_version(),
        dimensions_,
        num_clusters_,
        num_subspaces_);

    write_group.set_base_size(write_group.get_temp_size());
    write_group.store_metadata();
  }

  // This will call ingest_parts and then consolidate_partitions() on the
  // training set. Can be used if you do not want to ingest part by part and
  // want to do it all at once. Note that we will write using the
  // temporal_policy set in train(), and if that was not set, in the construtor.
  template <
      feature_vector_array Array,
      class Distance = sum_of_squares_distance>
  void ingest(
      const Array& vectors,
      const std::span<partitioned_ids_type>& external_ids,
      const std::span<partitioned_ids_type>& deleted_ids = {},
      Distance distance = Distance{}) {
    ingest_parts(
        vectors,
        external_ids,
        deleted_ids,
        0,
        ::num_vectors(vectors),
        0,
        "",
        distance);

    auto write_group = ivf_pq_group<ivf_pq_index>(
        group_->cached_ctx(),
        group_uri_,
        TILEDB_WRITE,
        temporal_policy_,
        group_->storage_version(),
        dimensions_,
        num_clusters_,
        num_subspaces_);
    write_group.set_base_size(::num_vectors(vectors));
    write_group.store_metadata();
  }

  template <
      feature_vector Vector,
      feature_vector PQVector,
      class SubDistance = uncached_sub_sum_of_squares_distance>
  auto pq_encode_one(
      const Vector& v,
      PQVector&& pq,
      SubDistance sub_distance = SubDistance{}) const {
    // We have broken the vector into num_subspaces_ subspaces, and we will look
    // in cluster_centroids_ and find the closest cluster_centroids_ to that
    // chunk of the vector.
    for (uint32_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      auto sub_begin = sub_dimensions_ * subspace;
      auto sub_end = sub_begin + sub_dimensions_;

      auto min_score = std::numeric_limits<score_type>::max();
      pq_code_type idx{0};
      for (size_t i = 0; i < num_clusters_; ++i) {
        auto score = sub_distance(v, cluster_centroids_[i], sub_begin, sub_end);
        if (score < min_score) {
          min_score = score;
          idx = i;
        }
      }
      pq[subspace] = idx;
    }
  }

  template <
      feature_vector_array Array,
      feature_vector_array PQArray,
      class Distance = uncached_sub_sum_of_squares_distance>
  auto pq_encode(
      const Array& training_set, Distance distance = Distance{}) const {
    scoped_timer _{"ivf_pq_index@pq_encode"};
    auto pq_vectors =
        std::make_unique<PQArray>(num_subspaces_, ::num_vectors(training_set));
    auto& pqv = *pq_vectors;
    for (size_t i = 0; i < ::num_vectors(training_set); ++i) {
      pq_encode_one(training_set[i], pqv[i], distance);
    }
    return pq_vectors;
  }

  /**
   * @brief Builds distance tables between each query and pq centroid.
   *
   * For each query, we iterate through each of it's subspaces, and for
   * each subspace we will compute the distance from the vector's subspace to
   * each of the (num_clusters_) pq_centroids. This results in a distance table
   * per query that is used during the actual distance computation between the
   * query and the encoded database vectors. This should be combined with
   * sub_distance_query_to_pq_centroid_distance_tables.
   *
   *
   * @param query_vectors Array of query vectors to compute centroid distance
   * tables for.
   * @param distance Distance function.
   *
   */
  template <
      feature_vector_array U,
      class Matrix,
      class Distance = uncached_sub_sum_of_squares_distance>
  auto generate_query_to_pq_centroid_distance_tables(
      const U& query_vectors, Distance distance = Distance{}) const {
    // pq_vectors[i][0:num_clusters_] holds, for the ith query vector, the
    // distance from the first subspace to each of the pq centroids.
    // pq_vectors[i][num_clusters_:num_clusters_ * 2] holds, for the ith query
    // vector, the distance from the second subspace to each of the pq
    // centroids.
    auto pq_vectors = std::make_unique<Matrix>(
        num_subspaces_ * num_clusters_, ::num_vectors(query_vectors));
    auto& pqv = *pq_vectors;
    auto local_distance = Distance{};
    for (size_t i = 0; i < ::num_vectors(query_vectors); ++i) {
      auto sub_begin = 0;
      auto sub_id = 0;
      for (size_t subspace = 0; subspace < num_subspaces_; ++subspace) {
        auto sub_end = sub_begin + sub_dimensions_;
        for (size_t centroid_id = 0; centroid_id < num_clusters_;
             ++centroid_id) {
          float pq_distance = centroid_id >= cluster_centroids_.num_cols() ?
                                  std::numeric_limits<float>::max() :
                                  local_distance(
                                      query_vectors[i],
                                      cluster_centroids_[centroid_id],
                                      sub_begin,
                                      sub_end);
          pqv[i][sub_id + centroid_id] = pq_distance;
        }
        sub_begin = sub_end;
        sub_id += num_clusters_;
      }
    }
    return pq_vectors;
  }

  static void clear_history(
      const tiledb::Context& ctx,
      const std::string& group_uri,
      uint64_t timestamp) {
    auto write_group =
        ivf_pq_group<ivf_pq_index>(ctx, group_uri, TILEDB_WRITE, {});
    write_group.clear_history(timestamp);
  }

  /*****************************************************************************
   *
   * Queries, infinite and finite.
   *
   * An "infinite" query assumes there is enough RAM to load the entire array
   * of partitioned vectors into memory. The query function then searches in
   * the appropriate partitions of the array for the query vectors.
   *
   * A "finite" query, on the other hand, examines the query and only loads
   * the partitions that are necessary for that particular search. A finite
   * query also supports out of core operation, meaning that only a subset of
   * the necessary partitions are loaded into memory at any one time. The
   * query is applied to each subset until all of the necessary partitions to
   * satisfy the query have been read in. The number of partitions to be held
   * in memory is controlled by an upper bound parameter that the user can set.
   * The upper bound limits the total number of vectors that will be held in
   * memory as the partitions are loaded. Only complete partitions are loaded,
   * so the actual number of vectors in memory at any one time will generally
   * be less than the upper bound.
   *
   * @note Currently we have implemented two queries, `query_infinite_ram` and
   * `query_finite_ram`. These both use the asymmetric distance function.
   * Meaning they pass the uncompressed query vectors to the query function,
   * along with the asymmetric distance functor. With the asymmetric distance
   * function, the query is kept uncompressed and the target is uncompressed
   * on the fly as the distance is computed. This is more expensive than
   * computing a symmetric distance, which would only require looking up the
   * distance in a lookup table. However, with a query using symmetric distance,
   * the query vectors would need to be compressed before the query is made,
   * which can be potentially quite expensive.
   *
   * @todo Implement variants of the query functions that use symmetric distance
   * (take care to evaluate performace / accuracy tradeoffs for `float` and
   *`uint8_t` feature vectors -- and do so after adding SIMD support).
   *
   * @todo Add vq and dist queries (should dist be its own index?)
   * @todo Order queries so that partitions are queried in order
   *
   ****************************************************************************/

  /**
   * @brief Perform a query on the index, returning the nearest neighbors
   * and distances. The function returns a matrix containing k_nn nearest
   * neighbors for each given query and a matrix containing the distances
   * corresponding to each returned neighbor.
   *
   * @tparam Q Type of query vectors.
   * @param query_vectors Array of (uncompressed) vectors to query.
   * @param k_nn Number of nearest neighbors to return.
   * @param nprobe Number of centroids to search.
   * @param k_factor Specifies the multiplier on k_nn for the initial IVF_PQ
   * query, after which re-ranking is run.
   *
   * @return A tuple containing a matrix of nearest neighbors and a matrix
   * of the corresponding distances.
   *
   */
  template <feature_vector_array Q>
  auto query(
      const Q& query_vectors,
      size_t k_nn,
      size_t nprobe,
      float k_factor = 1.f) {
    scoped_timer _{"ivf_pq_index@query"};
    if (k_factor < 1.f) {
      throw std::runtime_error("k_factor must be >= 1");
    }
    if (::num_vectors(flat_ivf_centroids_) < nprobe) {
      nprobe = ::num_vectors(flat_ivf_centroids_);
    }

    if (upper_bound_ > 0) {
      // Searches for the nearest neighbors using "finite RAM",
      // that is, it only loads that portion of the IVF index into memory that
      // is necessary for the given query. In addition, it supports out of core
      // operation, meaning that only a subset of the necessary partitions are
      // loaded into memory at any one time.
      if (!group_) {
        throw std::runtime_error(
            "[ivf_pq_index@read_index_finite] group_ is not initialized. This "
            "happens if you do not load an index by URI. Please close the "
            "index "
            "and re-open it by URI.");
      }

      // Open the index from the arrays contained in the group_uri.
      // The "finite" queries only load as much data (ids and vectors) as are
      // necessary for a given query -- so we can't load any data until we
      // know what the query is. So, here we would have read the centroids and
      // indices into memory, when creating the index but would not have read
      // the partitioned_ids or partitioned_pq_vectors.
      auto&& [active_partitions, active_queries] =
          detail::ivf::partition_ivf_flat_index<indices_type>(
              flat_ivf_centroids_, query_vectors, nprobe, num_threads_);
      auto partitioned_pq_vectors = std::make_unique<tdb_pq_storage_type>(
          group_->cached_ctx(),
          group_->pq_ivf_vectors_uri(),
          group_->feature_vectors_index_uri(),
          group_->get_num_partitions() + 1,
          group_->ids_uri(),
          active_partitions,
          upper_bound_,
          temporal_policy_);

      auto query_to_pq_centroid_distance_tables =
          std::move(*generate_query_to_pq_centroid_distance_tables<
                    Q,
                    ColMajorMatrix<float>>(query_vectors));
      size_t k_initial = static_cast<size_t>(k_nn * k_factor);
      auto&& [initial_distances, initial_ids, initial_indices] =
          detail::ivf::query_finite_ram(
              *partitioned_pq_vectors,
              query_to_pq_centroid_distance_tables,
              active_queries,
              k_initial,
              upper_bound_,
              num_threads_,
              make_pq_distance_query_to_pq_centroid_distance_tables<
                  std::span<float>,
                  decltype(pq_storage_type{}[0])>());
      return rerank(
          std::move(initial_distances),
          std::move(initial_ids),
          std::move(initial_indices),
          query_vectors,
          k_initial,
          k_nn);
    }

    if (!partitioned_pq_vectors_ ||
        ::num_vectors(*partitioned_pq_vectors_) == 0) {
      read_index_infinite();
    }

    // This function searches for the nearest neighbors using "infinite RAM". We
    // have already loaded the partitioned_pq_vectors_ into memory in the
    // constructor, so we can just run the query.
    auto&& [active_partitions, active_queries] =
        detail::ivf::partition_ivf_flat_index<indices_type>(
            flat_ivf_centroids_, query_vectors, nprobe, num_threads_);
    auto query_to_pq_centroid_distance_tables =
        std::move(*generate_query_to_pq_centroid_distance_tables<
                  Q,
                  ColMajorMatrix<float>>(query_vectors));

    // Perform the initial search with k_nn * k_factor.
    size_t k_initial = static_cast<size_t>(k_nn * k_factor);
    auto&& [initial_distances, initial_ids, initial_indices] =
        detail::ivf::query_infinite_ram(
            *partitioned_pq_vectors_,
            active_partitions,
            query_to_pq_centroid_distance_tables,
            active_queries,
            k_initial,
            num_threads_,
            make_pq_distance_query_to_pq_centroid_distance_tables<
                std::span<float>,
                decltype(pq_storage_type{}[0])>());

    return rerank(
        std::move(initial_distances),
        std::move(initial_ids),
        std::move(initial_indices),
        query_vectors,
        k_initial,
        k_nn);
  }

  auto rerank(
      ColMajorMatrix<float>&& initial_distances,
      ColMajorMatrix<id_type>&& initial_ids,
      ColMajorMatrix<size_t>&& initial_indices,
      const auto& query_vectors,
      size_t k_initial,
      size_t k_nn) {
    if (k_initial == k_nn) {
      return std::make_tuple(
          std::move(initial_distances), std::move(initial_ids));
    }

    auto get_vector_id = [&](size_t query_index,
                             size_t nn_index) -> std::tuple<bool, size_t> {
      auto valid = initial_ids[query_index][nn_index] !=
                   std::numeric_limits<id_type>::max();
      return {valid, valid ? initial_ids[query_index][nn_index] : 0};
    };

    if (::num_vectors(feature_vectors_) == 0 && group_) {
      std::unordered_map<id_type, size_t> id_to_vector_index;
      std::vector<uint64_t> vector_indices;
      for (size_t i = 0; i < ::num_vectors(initial_ids); ++i) {
        for (size_t j = 0; j < ::dimensions(initial_ids[i]); ++j) {
          if (initial_ids[i][j] != std::numeric_limits<id_type>::max() &&
              id_to_vector_index.find(initial_ids[i][j]) ==
                  id_to_vector_index.end()) {
            id_to_vector_index[initial_ids[i][j]] = vector_indices.size();
            vector_indices.push_back(initial_indices[i][j]);
          }
        }
      }

      auto feature_vectors =
          tdbColMajorMatrixMultiRange<feature_type, uint64_t>(
              group_->cached_ctx(),
              group_->feature_vectors_uri(),
              dimensions_,
              vector_indices,
              0,
              temporal_policy_);
      feature_vectors.load();

      auto get_vector_index = [&](size_t query_index,
                                  size_t nn_index) -> size_t {
        return id_to_vector_index[initial_ids[query_index][nn_index]];
      };
      return rerank_query(
          feature_vectors,
          query_vectors,
          get_vector_index,
          get_vector_id,
          k_initial,
          k_nn);
    }

    auto get_vector_index = [&](size_t query_index, size_t nn_index) -> size_t {
      return initial_indices[query_index][nn_index];
    };
    return rerank_query<decltype(feature_vectors_), decltype(query_vectors)>(
        feature_vectors_,
        query_vectors,
        get_vector_index,
        get_vector_id,
        k_initial,
        k_nn);
  }

  template <class FeatureVectors, class QueryVectors>
  auto rerank_query(
      const FeatureVectors& feature_vectors,
      const QueryVectors& query_vectors,
      std::function<size_t(size_t, size_t)> get_vector_index,
      std::function<std::tuple<bool, size_t>(size_t, size_t)> get_vector_id,
      size_t k_initial,
      size_t k_nn) const {
    auto min_scores = std::vector<fixed_min_pair_heap<score_type, id_type>>(
        ::num_vectors(query_vectors),
        fixed_min_pair_heap<score_type, id_type>(k_nn));
    for (size_t i = 0; i < ::num_vectors(query_vectors); ++i) {
      for (size_t j = 0; j < k_initial; ++j) {
        auto vector_index = get_vector_index(i, j);
        auto [valid, id] = get_vector_id(i, j);
        if (!valid) {
          continue;
        }
        float distance;
        if (distance_metric_ == DistanceMetric::SUM_OF_SQUARES) {
          distance = sum_of_squares_distance{}(
              query_vectors[i], feature_vectors[vector_index]);
        } else if (distance_metric_ == DistanceMetric::L2) {
          distance = sqrt_sum_of_squares_distance{}(
              query_vectors[i], feature_vectors[vector_index]);
        } else if (distance_metric_ == DistanceMetric::INNER_PRODUCT) {
          distance = inner_product_distance{}(
              query_vectors[i], feature_vectors[vector_index]);
        } else if (distance_metric_ == DistanceMetric::COSINE) {
          distance = cosine_distance_normalized{}(
              query_vectors[i], feature_vectors[vector_index]);
        } else {
          throw std::runtime_error(
              "[ivf_pq_index@rerank_query] Invalid distance metric: " +
              to_string(distance_metric_));
        }
        min_scores[i].insert(distance, id);
      }
    }

    return get_top_k_with_scores(min_scores, k_nn);
  }

  /***************************************************************************
   * Getters. Note that we don't have a `num_vectors` because it isn't clear
   * what that means for a partitioned (possibly out-of-core) index.
   ***************************************************************************/
  const ivf_pq_group<ivf_pq_index>& group() const {
    if (!group_) {
      throw std::runtime_error("No group available");
    }
    return *group_;
  }

  uint64_t dimensions() const {
    return dimensions_;
  }

  size_t upper_bound() const {
    return upper_bound_;
  }

  auto num_partitions() const {
    if (num_partitions_ != ::num_vectors(flat_ivf_centroids_)) {
      throw std::runtime_error(
          "[ivf_pq_index@num_partitions] num_partitions_ != "
          "::num_vectors(flat_ivf_centroids_)");
    }
    return num_partitions_;
  }

  uint32_t num_subspaces() const {
    return num_subspaces_;
  }

  uint32_t sub_dimensions() const {
    return sub_dimensions_;
  }

  uint32_t bits_per_subspace() const {
    return bits_per_subspace_;
  }

  uint32_t num_clusters() const {
    return num_clusters_;
  }

  TemporalPolicy temporal_policy() const {
    return temporal_policy_;
  }

  uint32_t max_iterations() const {
    return max_iterations_;
  }

  float convergence_tolerance() const {
    return convergence_tolerance_;
  }

  uint64_t num_threads() const {
    return num_threads_;
  }

  float reassign_ratio() const {
    return reassign_ratio_;
  }

  uint64_t partitions() const {
    return num_partitions_;
  }

  constexpr auto distance_metric() const {
    return distance_metric_;
  }

  /***************************************************************************
   * Methods to aid Testing and Debugging
   *
   * @todo -- As elsewhere in this class, there is huge code duplication here
   *
   **************************************************************************/

  /**
   * @brief Verify that the pq encoding is correct by comparing every feature
   * vector against the corresponding pq encoded value
   * @param feature_vectors
   * @return the average error
   */
  double verify_pq_encoding(
      const ColMajorMatrix<feature_type>& feature_vectors,
      bool debug = false) const {
    double total_distance = 0.0;
    double total_normalizer = 0.0;

    for (size_t i = 0; i < ::num_vectors(feature_vectors); ++i) {
      if (debug) {
        std::cout << "-------------------" << std::endl;
      }
      auto reconstructed_vector = std::vector<feature_type>(dimensions_);
      for (uint32_t subspace = 0; subspace < num_subspaces_; ++subspace) {
        auto sub_begin = sub_dimensions_ * subspace;
        auto sub_end = sub_dimensions_ * (subspace + 1);
        auto centroid =
            cluster_centroids_[(*unpartitioned_pq_vectors_)(subspace, i)];

        // Reconstruct the encoded vector
        for (size_t j = sub_begin; j < sub_end; ++j) {
          reconstructed_vector[j] = centroid[j];
        }
      }

      // Measure the distance between the original vector and the reconstructed
      // vector and accumulate into the total distance as well as the total
      // weight of the feature vector
      auto distance = l2_distance(feature_vectors[i], reconstructed_vector);
      total_distance += distance;
      total_normalizer += l2_distance(feature_vectors[i]);

      if (debug) {
        debug_vector(feature_vectors[i], "original vector     ", 100);
        debug_vector(reconstructed_vector, "reconstructed vector", 100);
        std::cout << "distance: " << distance << std::endl;
      }
    }

    // Return the total accumulated distance between the encoded and original
    // vectors, divided by the total weight of the original feature vectors
    auto error =
        total_normalizer == 0. ? 0.f : total_distance / total_normalizer;
    if (debug) {
      std::cout << "total_distance: " << total_distance
                << ", total_normalizer: " << total_normalizer
                << ", error: " << error << std::endl;
    }
    return error;
  }

  /**
   * @brief Verify that recorded distances between centroids are correct by
   * comparing the distance between every pair of pq vectors against the
   * distance between every pair of the original feature vectors.
   *
   * Currently only supports sum of squares distance.
   *
   * @param feature_vectors
   * @return
   */
  auto verify_symmetric_pq_distances(
      const ColMajorMatrix<feature_type>& feature_vectors) const {
    double total_diff_symmetric = 0.0;
    double total_normalizer = 0.0;

    // Lookup table for the distance between centroids of each subspace.
    // distance_tables_[i](j, k) is the distance between the jth and kth
    // centroids in the ith subspace. Create tables of distances storing
    // distance between encoding keys, one table for each subspace. That is,
    // distance_tables_[i](j, k) is the distance between the jth and kth
    // centroids in the ith subspace. The distance between two encoded vectors
    // is looked up using the keys of the vectors in each subspace (summing up
    // the results obtained from each subspace).
    // @todo SIMDize with subspace iteration in inner loop
    auto distance_tables =
        std::vector<ColMajorMatrix<score_type>>(num_subspaces_);
    for (size_t i = 0; i < num_subspaces_; ++i) {
      distance_tables[i] =
          ColMajorMatrix<score_type>(num_clusters_, num_clusters_);
    }
    for (uint32_t subspace = 0; subspace < num_subspaces_; ++subspace) {
      auto sub_begin = subspace * sub_dimensions_;
      auto sub_end = (subspace + 1) * sub_dimensions_;
      auto local_sub_distance =
          cached_sub_sum_of_squares_distance{sub_begin, sub_end};

      for (size_t i = 0; i < num_clusters_; ++i) {
        for (size_t j = 0; j < num_clusters_; ++j) {
          auto sub_distance =
              local_sub_distance(cluster_centroids_[i], cluster_centroids_[j]);
          distance_tables[subspace](i, j) = sub_distance;
        }
      }
    }

    for (size_t i = 0; i < ::num_vectors(feature_vectors); ++i) {
      for (size_t j = i + 1; j < ::num_vectors(feature_vectors); ++j) {
        auto real_distance =
            l2_distance(feature_vectors[i], feature_vectors[j]);
        total_normalizer += real_distance;

        auto pq_distance_symmetric = 0.0;
        for (uint32_t subspace = 0; subspace < num_subspaces_; ++subspace) {
          auto sub_distance = distance_tables[subspace](
              (*unpartitioned_pq_vectors_)(subspace, i),
              (*unpartitioned_pq_vectors_)(subspace, j));
          pq_distance_symmetric += sub_distance;
        }

        total_diff_symmetric += std::abs(real_distance - pq_distance_symmetric);
      }
    }

    return total_diff_symmetric / total_normalizer;
  }

  auto verify_asymmetric_pq_distances(
      const ColMajorMatrix<feature_type>& feature_vectors) {
    double total_diff = 0.0;
    double total_normalizer = 0.0;

    score_type diff_max = 0.0;
    score_type vec_max = 0.0;
    for (size_t i = 0; i < ::num_vectors(feature_vectors); ++i) {
      for (size_t j = i + 1; j < ::num_vectors(feature_vectors); ++j) {
        auto real_distance =
            l2_distance(feature_vectors[i], feature_vectors[j]);
        total_normalizer += real_distance;

        auto pq_distance = this->sub_distance_asymmetric(
            feature_vectors[i], (*unpartitioned_pq_vectors_)[j]);

        auto diff = std::abs(real_distance - pq_distance);
        diff_max = std::max(diff_max, diff);
        total_diff += diff;
      }
      vec_max = std::max(vec_max, l2_distance(feature_vectors[i]));
    }

    return std::make_tuple(diff_max / vec_max, total_diff / total_normalizer);
  }

  /**
   * @brief Compare groups associated with two ivf_pq_index objects for
   * equality. Note that both indexes will have had to perform a read or
   * a write. An index created from partitioning will not yet have a group
   * associated with it.
   *
   * Comparing groups will also compare metadata associated with each group.
   *
   * @param rhs the index against which to compare
   * @return bool indicating equality of the groups
   */
  bool compare_group(const ivf_pq_index& rhs) const {
    if (!group_ && !rhs.group_) {
      return true;
    }
    if (!group_ || !rhs.group_) {
      return false;
    }
    return group_->compare_group(*(rhs.group_));
  }

  /**
   * @brief Compare metadata associated with two ivf_pq_index objects for
   * equality. This is not the same as the metadata associated with the index
   * group. Rather, it is the metadata associated with the index itself and is
   * only a small number of cached quantities.
   *
   * @param rhs the index against which to compare
   * @return bool indicating equality of the index metadata
   */
  bool compare_cached_metadata(const ivf_pq_index& rhs) const {
    if (dimensions_ != rhs.dimensions_) {
      return false;
    }
    if (num_partitions_ != rhs.num_partitions_) {
      return false;
    }
    if (num_subspaces_ != rhs.num_subspaces_) {
      return false;
    }
    if (sub_dimensions_ != rhs.sub_dimensions_) {
      return false;
    }
    if (bits_per_subspace_ != rhs.bits_per_subspace_) {
      return false;
    }
    if (num_clusters_ != rhs.num_clusters_) {
      return false;
    }
    if (max_iterations_ != rhs.max_iterations_) {
      return false;
    }
    if (convergence_tolerance_ != rhs.convergence_tolerance_) {
      return false;
    }
    if (reassign_ratio_ != rhs.reassign_ratio_) {
      return false;
    }
    if (distance_metric_ != rhs.distance_metric_) {
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
        ::dimensions(lhs) != ::dimensions(rhs)) {
      std::cout << "num_vectors(lhs) != num_vectors(rhs) || dimensions(lhs) != "
                   "dimensions(rhs)"
                << std::endl;
      std::cout << "num_vectors(lhs): " << ::num_vectors(lhs)
                << " num_vectors(rhs): " << ::num_vectors(rhs) << std::endl;
      std::cout << "dimensions(lhs): " << ::dimensions(lhs)
                << " dimensions(rhs): " << ::dimensions(rhs) << std::endl;
      debug_matrix(lhs, "[ivf_pq_index@compare_feature_vector_arrays] lhs");
      debug_matrix(rhs, "[ivf_pq_index@compare_feature_vector_arrays] rhs");
      return false;
    }
    for (size_t i = 0; i < ::num_vectors(lhs); ++i) {
      if (!std::equal(begin(lhs[i]), end(lhs[i]), begin(rhs[i]))) {
        std::cout << "lhs[" << i << "] != rhs[" << i << "]" << std::endl;
        std::cout << "lhs[" << i << "]: ";
        for (size_t j = 0; j < ::dimensions(lhs); ++j) {
          std::cout << lhs[i][j] << " ";
        }
        std::cout << std::endl;
        std::cout << "rhs[" << i << "]: ";
        for (size_t j = 0; j < ::dimensions(rhs); ++j) {
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
  auto compare_feature_vectors(
      const L& lhs, const R& rhs, const std::string& msg = "") const {
    if (::dimensions(lhs) != ::dimensions(rhs)) {
      std::cout << "[ivf_pq_index@compare_feature_vectors] " << msg
                << " dimensions(lhs) != dimensions(rhs) (" << ::dimensions(lhs)
                << " != " << ::dimensions(rhs) << ")" << std::endl;
      return false;
    }
    auto equal = std::equal(begin(lhs), end(lhs), begin(rhs));
    if (!equal) {
      std::cout << "[ivf_pq_index@compare_feature_vectors] " << msg
                << " failed the equality check." << std::endl;
      auto printed = 0;
      for (size_t i = 0; i < ::dimensions(lhs); ++i) {
        if (lhs[i] != rhs[i]) {
          std::cout << "  lhs[" << i << "]: " << lhs[i] << " - rhs[" << i
                    << "]: " << rhs[i] << std::endl;
          printed++;
        }
        if (printed > 50) {
          std::cout << "  ..." << std::endl;
          break;
        }
      }
    }
    return equal;
  }

  auto compare_cluster_centroids(const ivf_pq_index& rhs) const {
    return compare_feature_vector_arrays(
        cluster_centroids_, rhs.cluster_centroids_);
  }

  auto compare_flat_ivf_centroids(const ivf_pq_index& rhs) const {
    return compare_feature_vector_arrays(
        flat_ivf_centroids_, rhs.flat_ivf_centroids_);
  }

  auto compare_ivf_index(const ivf_pq_index& rhs) const {
    if (!partitioned_pq_vectors_ && !rhs.partitioned_pq_vectors_) {
      return true;
    }
    if (!partitioned_pq_vectors_ || !rhs.partitioned_pq_vectors_) {
      std::cout << "[ivf_pq_index@compare_ivf_index] partitioned_pq_vectors_ "
                   "|| rhs.partitioned_pq_vectors_ is nullptr"
                << std::endl;
      return false;
    }
    return compare_feature_vectors(
        partitioned_pq_vectors_->indices(),
        rhs.partitioned_pq_vectors_->indices(),
        "partitioned_pq_vectors_->indices()");
  }

  auto compare_ivf_ids(const ivf_pq_index& rhs) const {
    if (!partitioned_pq_vectors_ && !rhs.partitioned_pq_vectors_) {
      return true;
    }
    if (!partitioned_pq_vectors_ || !rhs.partitioned_pq_vectors_) {
      std::cout << "[ivf_pq_index@compare_ivf_ids] partitioned_pq_vectors_ || "
                   "rhs.partitioned_pq_vectors_ is nullptr"
                << std::endl;
      return false;
    }
    return compare_feature_vectors(
        partitioned_pq_vectors_->ids(),
        rhs.partitioned_pq_vectors_->ids(),
        "partitioned_pq_vectors_->ids()");
  }

  auto compare_pq_ivf_vectors(const ivf_pq_index& rhs) const {
    if (!partitioned_pq_vectors_ && !rhs.partitioned_pq_vectors_) {
      return true;
    }
    if (!partitioned_pq_vectors_ || !rhs.partitioned_pq_vectors_) {
      return false;
    }
    return compare_feature_vector_arrays(
        *partitioned_pq_vectors_, *(rhs.partitioned_pq_vectors_));
  }

  template <class Other>
  bool operator==(const Other& rhs) const {
    if (this == &rhs) {
      return true;
    }
    if (!std::is_same_v<feature_type, typename Other::feature_type>) {
      return false;
    }
    if (!std::is_same_v<indices_type, typename Other::indices_type>) {
      return false;
    }
    if (!std::is_same_v<id_type, typename Other::id_type>) {
      return false;
    }

    if (compare_group(rhs) == false) {
      return false;
    }
    if (compare_cached_metadata(rhs) == false) {
      return false;
    }
    if (compare_cluster_centroids(rhs) == false) {
      return false;
    }
    if (compare_flat_ivf_centroids(rhs) == false) {
      return false;
    }
    if (compare_ivf_index(rhs) == false) {
      return false;
    }
    if (compare_ivf_ids(rhs) == false) {
      return false;
    }
    if (compare_pq_ivf_vectors(rhs) == false) {
      return false;
    }
    return true;
  }

  auto set_flat_ivf_centroids(const ColMajorMatrix<feature_type>& centroids) {
    num_partitions_ = centroids.num_cols();
    flat_ivf_centroids_ = flat_ivf_centroid_storage_type(
        ::dimensions(centroids), ::num_vectors(centroids));
    std::copy(
        centroids.data(),
        centroids.data() + centroids.num_rows() * centroids.num_cols(),
        flat_ivf_centroids_.data());
  }

  const auto& get_flat_ivf_centroids() const {
    return flat_ivf_centroids_;
  }

  void dump(const std::string& msg) const {
    if (!group_) {
      throw std::runtime_error(
          "[ivf_flat_index@dump] Cannot dump group because there is no "
          "group");
    }
    group_->dump(msg);
  }
};

#endif  // TILEDB_PQ_INDEX_H
