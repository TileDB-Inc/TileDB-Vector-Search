/**
 * @file   vamana_index.h
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2023 TileDB, Inc.
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
 *
 */

#ifndef TDB_VAMANA_INDEX_H
#define TDB_VAMANA_INDEX_H

#include <cstddef>

#include <functional>
#include <queue>
#include <unordered_set>

#include <tiledb/tiledb>
#include "detail/graph/adj_list.h"
#include "detail/graph/best_first.h"
#include "detail/graph/bfs.h"
#include "detail/graph/diskann.h"
#include "detail/graph/graph_utils.h"
#include "detail/graph/greedy_search.h"
#include "detail/linalg/tdb_matrix_with_ids.h"
#include "detail/linalg/vector.h"
#include "detail/time/temporal_policy.h"
#include "index/vamana_group.h"
#include "scoring.h"
#include "stats.h"
#include "utils/fixed_min_heap.h"
#include "utils/print_types.h"

#include <tiledb/tiledb>

/**
 * Find the vector that is closest to the centroid of the set of vectors P.
 * @tparam Distance The distance functor used to compare vectors
 * @param P The set of vectors to be computed over
 * @param distance The distance functor used to compare vectors
 * @return The index of the vector in P that is closest to the centroid of P
 *
 * @todo Instead of <float>, centroid type should be return type of distance
 */
template <class Distance = sum_of_squares_distance>
auto medoid(auto&& P, Distance distance = Distance{}) {
  auto n = num_vectors(P);
  auto centroid = Vector<float>(P[0].size());
  std::fill(begin(centroid), end(centroid), 0.0);

  for (size_t j = 0; j < n; ++j) {
    auto p = P[j];
    for (size_t i = 0; i < p.size(); ++i) {
      centroid[i] += p[i];
    }
  }
  for (size_t i = 0; i < centroid.size(); ++i) {
    centroid[i] /= (float)num_vectors(P);
  }

  std::vector<float> tmp{begin(centroid), end(centroid)};
  auto min_score = std::numeric_limits<float>::max();
  auto med = 0UL;
  for (size_t i = 0; i < n; ++i) {
    auto score = distance(P[i], centroid);
    if (score < min_score) {
      min_score = score;
      med = i;
    }
  }

  return med;
}

/**
 * @brief The Vamana index.
 *
 * @tparam FeatureType Type of the elements in the feature vectors.
 * @tparam IdType Type of the ids of the feature vectors.
 * @tparam AdjacencyRowIndexType Types of the indexes used in the graph.
 */
template <
    class FeatureType,
    class IdType,
    class AdjacencyRowIndexType = uint64_t>
class vamana_index {
 public:
  using feature_type = FeatureType;
  using id_type = IdType;
  using adjacency_row_index_type = AdjacencyRowIndexType;
  using adjacency_scores_type = float;

  using group_type = vamana_index_group<vamana_index>;
  using metadata_type = vamana_index_metadata;

 private:
  /****************************************************************************
   * Index group information
   ****************************************************************************/
  TemporalPolicy temporal_policy_;
  std::unique_ptr<vamana_index_group<vamana_index>> group_;

  /*
   * The feature vectors.  These contain the original input vectors, modified
   * with updates and deletions over time.
   */
  ColMajorMatrixWithIds<feature_type, id_type> feature_vectors_;

  /****************************************************************************
   * Index representation
   ****************************************************************************/

  // Cached information about the index
  uint64_t dimensions_{0};
  uint64_t num_vectors_{0};
  uint64_t num_edges_{0};

  /** The graph representing the index over `feature_vectors_` */
  ::detail::graph::adj_list<adjacency_scores_type, id_type> graph_;

  /*
   * The medoid of the feature vectors -- the vector in the set that is closest
   * to the centroid of the entire set. This is used as the starting point for
   * queries.
   * @todo -- In the partitioned case, we will want to use a vector of medoids,
   * one for each partition.
   */
  id_type medoid_{0};

  /*
   * Training parameters
   */
  uint64_t l_build_{0};       // diskANN paper says default = 100
  uint64_t r_max_degree_{0};  // diskANN paper says default = 64
  float alpha_min_{1.0};      // per diskANN paper
  float alpha_max_{1.2};      // per diskANN paper

 public:
  /****************************************************************************
   * Constructors (et al)
   ****************************************************************************/

  vamana_index() = delete;
  vamana_index(const vamana_index& index) = delete;
  vamana_index& operator=(const vamana_index& index) = delete;
  vamana_index(vamana_index&& index) = default;
  vamana_index& operator=(vamana_index&& index) = default;

  ~vamana_index() = default;

  /**
   * Construct empty index in preparation for construction and training
   */
  vamana_index(
      size_t num_nodes,
      size_t l_build,
      size_t r_max_degree,
      std::optional<TemporalPolicy> temporal_policy = std::nullopt)
      : temporal_policy_{
        temporal_policy.has_value() ? *temporal_policy :
        TemporalPolicy{TimeTravel, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count())}}
      , num_vectors_{num_nodes}
      , graph_{num_vectors_}
      , l_build_{l_build}
      , r_max_degree_{r_max_degree} {
  }

  /**
   * @brief Load a vamana graph index from a TileDB group
   * @param ctx TileDB context
   * @param group_uri URI of the group containing the index
   */
  vamana_index(
      tiledb::Context ctx,
      const std::string& uri,
      std::optional<TemporalPolicy> temporal_policy = std::nullopt)
      : temporal_policy_{temporal_policy.has_value() ? *temporal_policy : TemporalPolicy()}
      , group_{std::make_unique<vamana_index_group<vamana_index>>(
            ctx, uri, TILEDB_READ, temporal_policy_)} {
    // @todo Make this table-driven
    dimensions_ = group_->get_dimensions();
    num_vectors_ = group_->get_base_size();
    num_edges_ = group_->get_num_edges();
    l_build_ = group_->get_l_build();
    r_max_degree_ = group_->get_r_max_degree();
    alpha_min_ = group_->get_alpha_min();
    alpha_max_ = group_->get_alpha_max();
    medoid_ = group_->get_medoid();

    if (group_->should_skip_query()) {
      num_vectors_ = 0;
    }

    feature_vectors_ =
        std::move(tdbColMajorPreLoadMatrixWithIds<feature_type, id_type>(
            group_->cached_ctx(),
            group_->feature_vectors_uri(),
            group_->ids_uri(),
            dimensions_,
            num_vectors_,
            0,
            temporal_policy_));
    // If we have time travelled to before any vectors were written to the
    // arrays then we may have metadata which says we have N vectors, but in
    // reality we have 0 vectors. So here we check how many vectors were
    // actually read and update to that number.
    num_vectors_ = _cpo::num_vectors(feature_vectors_);

    /*
     * Read the feature vectors
     * Read the graph
     *   Read the adjacency scores
     *   Read the adjacency ids
     *   Read the adjacency row index
     * @todo Encapsulate reading the graph?
     */

    /****************************************************************************
     * Read the graph
     * Here, we assume a dynamic graph, which is one that we can later add more
     * edges and vertices to (to index new vectors).
     * @todo Add case for static graph -- i.e., CSR -- that we can read the
     * vectors into directly
     * @todo Encapsulate reading the graph?
     * @todo Instead of saving scores, recompute them on ingestion
     ****************************************************************************/
    graph_ = ::detail::graph::adj_list<feature_type, id_type>(num_vectors_);

    auto adj_scores = read_vector<adjacency_scores_type>(
        group_->cached_ctx(),
        group_->adjacency_scores_uri(),
        0,
        num_edges_,
        temporal_policy_);
    auto adj_ids = read_vector<id_type>(
        group_->cached_ctx(),
        group_->adjacency_ids_uri(),
        0,
        num_edges_,
        temporal_policy_);
    auto adj_index = read_vector<adjacency_row_index_type>(
        group_->cached_ctx(),
        group_->adjacency_row_index_uri(),
        0,
        num_vectors_ + 1,
        temporal_policy_);

    // Here we build a graph using the graph data we read in.  We do it this
    // way for a dynamic graph, which is one that we can later add more edges
    // and vertices to (to index new vectors).
    for (size_t i = 0; i < num_vectors_; ++i) {
      auto start = adj_index[i];
      auto end = adj_index[i + 1];
      for (size_t j = start; j < end; ++j) {
        graph_.add_edge(i, adj_ids[j], adj_scores[j]);
      }
    }
  }

  vamana_index(const std::string& diskann_index) {
    const std::string diskann_data = diskann_index + ".data";
    feature_vectors_ = read_diskann_data(diskann_data);
    size_t num_nodes = num_vectors(feature_vectors_);
    graph_ = read_diskann_mem_index_with_scores(
        diskann_index, diskann_data, num_nodes);
  }

  /**
   * @brief Build a vamana graph index.  This is algorithm is from the Filtered
   * Fresh DiskAnn paper -- which is a different training process than the
   * original DiskAnn paper. The Filtered Fresh DiskAnn paper
   * (https://arxiv.org/pdf/2103.01937.pdf):
   *
   * Initialize G to an empty graph
   * Let s denote the medoid of P
   * Let st (f) denote the start node for filter label f for every f∈F
   * Let σ be a random permutation of [n]
   * Let F_x be the label-set for every x∈P
   * foreach i∈[n] do
   * Let S_(F_(x_(σ(i)) ) )={st⁡(f):f∈F_(x_(σ(i)) ) }
   * Let [∅;V_(F_(x_(σ(i)) ) ) ]← FilteredGreedySearch (S_(F_(x_(σ(i)) ) ) ┤,
   *     (├ x_(σ(i)),0,L,F_(x_(σ(i)) ) )@V←V∪V_(F_(x_(σ(i)) ) ) )
   * Run FilteredRobustPrune (σ(i),V_(F_(x_(σ(i)) ) ),α,R) to update
   * out-neighbors of σ(i). foreach " j∈N_"out "  (σ(i))" do " Update N_"out "
   * (j)←N_"out " (j)∪{σ(i)} if |N_"out "  (j)|>R then Run FilteredRobustPrune
   * (j,N_"out " (j),α,R) to update out-neighbors of j.
   */
  template <
      feature_vector_array Array,
      feature_vector Vector,
      class Distance = sum_of_squares_distance>
  void train(
      const Array& training_set,
      const Vector& training_set_ids,
      Distance distance = Distance{}) {
    feature_vectors_ = std::move(ColMajorMatrixWithIds<feature_type, id_type>(
        ::dimensions(training_set), ::num_vectors(training_set)));
    std::copy(
        training_set.data(),
        training_set.data() +
            ::dimensions(training_set) * ::num_vectors(training_set),
        feature_vectors_.data());
    std::copy(
        training_set_ids.begin(),
        training_set_ids.end(),
        feature_vectors_.ids());

    dimensions_ = ::dimensions(feature_vectors_);
    num_vectors_ = ::num_vectors(feature_vectors_);
    // graph_ = ::detail::graph::init_random_adj_list<feature_type, id_type>(
    //     feature_vectors_, r_max_degree_);

    graph_ = ::detail::graph::adj_list<feature_type, id_type>(num_vectors_);
    // dump_edgelist("edges_" + std::to_string(0) + ".txt", graph_);

    medoid_ = medoid(feature_vectors_);

    // debug_index();

    size_t counter{0};
    //    for (float alpha : {alpha_min_, alpha_max_}) {
    // Just use one value of alpha
    for (float alpha : {alpha_max_}) {
      scoped_timer _("train " + std::to_string(counter));
      size_t total_visited{0};
      for (size_t p = 0; p < num_vectors_; ++p) {
        ++counter;

        // Do not need top_k or top_k scores here -- use path_only enum
        auto&& [top_k_scores, top_k, visited] =
            ::best_first_O4 /*greedy_search*/ (
                graph_,
                feature_vectors_,
                medoid_,
                feature_vectors_[p],
                1,
                l_build_,
                distance);
        total_visited += visited.size();

        robust_prune(
            graph_,
            feature_vectors_,
            p,
            visited,
            alpha,
            r_max_degree_,
            distance);
        {
          scoped_timer _{"post search prune"};
          for (auto&& [i, j] : graph_.out_edges(p)) {
            // @todo Do this without copying -- prune should take vector of
            //  tuples and p (it copies anyway) maybe scan for p and then only
            //  build tmp after if?
            auto tmp = std::vector<id_type>(graph_.out_degree(j) + 1);
            tmp.push_back(p);
            for (auto&& [_, k] : graph_.out_edges(j)) {
              tmp.push_back(k);
            }

            if (size(tmp) > r_max_degree_) {
              robust_prune(
                  graph_,
                  feature_vectors_,
                  j,
                  tmp,
                  alpha,
                  r_max_degree_,
                  distance);
            } else {
              graph_.add_edge(
                  j, p, distance(feature_vectors_[p], feature_vectors_[j]));
            }
          }
        }
        if ((counter) % 10 == 0) {
          // dump_edgelist("edges_" + std::to_string(counter) + ".txt", graph_);
        }
      }
      // debug_index();
    }
  }

  /**
   * @brief Add a set of vectors to the index (the vectors that will be
   * searched over in subsequent queries).  This is a no-op for vamana.
   *
   * @tparam A Type of the array of vectors to be added
   * @param database The vectors to be added
   */
  template <feature_vector_array A>
  void add(const A& database) {
  }

  /*
   * Some diagnostic variables and accessors
   */
  size_t num_visited_vertices_{0};
  size_t num_visited_edges_{0};
  static size_t num_comps_;

  size_t num_visited_vertices() const {
    return num_visited_vertices_;
  }

  size_t num_comps() const {
    return num_comps_;
  }

  TemporalPolicy temporal_policy() const {
    return temporal_policy_;
  }

  auto bfs_O0() {
    return ::bfs_O0(graph_, medoid_);
  }

  template <query_vector_array Q>
  auto bfs_O1(const Q& queries, size_t k_nn, size_t Lbuild) {
    for (size_t i = 0; i < num_vectors(queries); ++i) {
      ::bfs_O1(graph_, feature_vectors_, medoid_, queries[i], Lbuild);
    }
  }

  template <query_vector_array Q>
  auto best_first_O0(const Q& queries) {
    for (size_t i = 0; i < num_vectors(queries); ++i) {
      ::best_first_O0(graph_, feature_vectors_, medoid_, queries[i]);
    }
  }

  template <query_vector_array Q>
  auto best_first_O1(const Q& queries, size_t k_nn, size_t Lbuild) {
    for (size_t i = 0; i < num_vectors(queries); ++i) {
      ::best_first_O1(graph_, feature_vectors_, medoid_, queries[i], Lbuild);
    }
  }

  template <query_vector_array Q>
  auto best_first_O2(
      const Q& queries, size_t k_nn, std::optional<size_t> l_search) {
    size_t Lbuild = l_search ? *l_search : l_build_;

    auto top_k = ColMajorMatrix<id_type>(k_nn, ::num_vectors(queries));
    auto top_k_scores =
        ColMajorMatrix<adjacency_scores_type>(k_nn, ::num_vectors(queries));

    for (size_t i = 0; i < num_vectors(queries); ++i) {
      auto&& [tk_scores, tk, V] = ::best_first_O2(
          graph_,
          feature_vectors_,
          medoid_,
          queries[i],
          k_nn,
          Lbuild,
          sum_of_squares_distance{});
      std::copy(
          tk_scores.data(), tk_scores.data() + k_nn, top_k_scores[i].data());
      std::copy(tk.data(), tk.data() + k_nn, top_k[i].data());
      num_visited_vertices_ += V.size();
    }

    return std::make_tuple(std::move(top_k_scores), std::move(top_k));
  }

  template <query_vector_array Q>
  auto best_first_O3(
      const Q& queries, size_t k_nn, std::optional<size_t> l_search) {
    size_t Lbuild = l_search ? *l_search : l_build_;

    auto top_k = ColMajorMatrix<id_type>(k_nn, ::num_vectors(queries));
    auto top_k_scores =
        ColMajorMatrix<adjacency_scores_type>(k_nn, ::num_vectors(queries));

    for (size_t i = 0; i < num_vectors(queries); ++i) {
      auto&& [tk_scores, tk, V] = ::best_first_O3(
          graph_,
          feature_vectors_,
          medoid_,
          queries[i],
          k_nn,
          Lbuild,
          sum_of_squares_distance{});
      std::copy(
          tk_scores.data(), tk_scores.data() + k_nn, top_k_scores[i].data());
      std::copy(tk.data(), tk.data() + k_nn, top_k[i].data());
      num_visited_vertices_ += V.size();
    }

    return std::make_tuple(std::move(top_k_scores), std::move(top_k));
  }

  template <query_vector_array Q>
  auto best_first_O4(
      const Q& queries, size_t k_nn, std::optional<size_t> l_search) {
    size_t Lbuild = l_search ? *l_search : l_build_;

    auto top_k = ColMajorMatrix<id_type>(k_nn, ::num_vectors(queries));
    auto top_k_scores =
        ColMajorMatrix<adjacency_scores_type>(k_nn, ::num_vectors(queries));

    for (size_t i = 0; i < num_vectors(queries); ++i) {
      auto&& [tk_scores, tk, V] = ::best_first_O4(
          graph_,
          feature_vectors_,
          medoid_,
          queries[i],
          k_nn,
          Lbuild,
          sum_of_squares_distance{});
      std::copy(
          tk_scores.data(), tk_scores.data() + k_nn, top_k_scores[i].data());
      std::copy(tk.data(), tk.data() + k_nn, top_k[i].data());
      num_visited_vertices_ += V.size();
    }

    return std::make_tuple(std::move(top_k_scores), std::move(top_k));
  }

  template <query_vector_array Q>
  auto best_first_O5(
      const Q& queries, size_t k_nn, std::optional<size_t> l_search) {
    size_t Lbuild = l_search ? *l_search : l_build_;

    auto top_k = ColMajorMatrix<id_type>(k_nn, ::num_vectors(queries));
    auto top_k_scores =
        ColMajorMatrix<adjacency_scores_type>(k_nn, ::num_vectors(queries));

    for (size_t i = 0; i < num_vectors(queries); ++i) {
      auto&& [tk_scores, tk, V] = ::best_first_O5(
          graph_,
          feature_vectors_,
          medoid_,
          queries[i],
          k_nn,
          Lbuild,
          sum_of_squares_distance{});
      std::copy(
          tk_scores.data(), tk_scores.data() + k_nn, top_k_scores[i].data());
      std::copy(tk.data(), tk.data() + k_nn, top_k[i].data());
      num_visited_vertices_ += V.size();
    }

    return std::make_tuple(std::move(top_k_scores), std::move(top_k));
  }

  /**
   * @brief Query the index for the top k nearest neighbors of the query set
   * @tparam Q Type of query set
   * @param query_set Container of query vectors
   * @param k How many nearest neighbors to return
   * @param l_search How deep to search
   * @return Tuple of top k scores and top k ids
   */
  template <query_vector_array Q, class Distance = sum_of_squares_distance>
  auto query(
      const Q& query_set,
      size_t k,
      std::optional<size_t> l_search = std::nullopt,
      Distance distance = Distance{}) {
    scoped_timer __{tdb_func__ + std::string{" (outer)"}};

    size_t L = l_search ? *l_search : l_build_;
    // L = std::min<size_t>(L, l_build_);

    auto top_k = ColMajorMatrix<id_type>(k, ::num_vectors(query_set));
    auto top_k_scores =
        ColMajorMatrix<adjacency_scores_type>(k, ::num_vectors(query_set));

#if 0
    // Parallelized implementation -- we stay single-threaded for now
    // for purposes of comparison
    size_t nthreads = std::thread::hardware_concurrency();
    auto par = stdx::execution::indexed_parallel_policy{nthreads};

    stdx::range_for_each(std::move(par), query_set, [&](auto&& query_vec, auto n, auto i) {
      auto&& [tk_scores, tk, V] = greedy_search(
          graph_, feature_vectors_, medoid_, query_vec, k, L, distance, true);
      std::copy(tk_scores.data(), tk_scores.data() + k, top_k_scores[i].data());
      std::copy(tk.data(), tk.data() + k, top_k[i].data());
    });
#else
    for (size_t i = 0; i < num_vectors(query_set); ++i) {
      auto&& [tk_scores, tk, V] = greedy_search(
          graph_,
          feature_vectors_,
          medoid_,
          query_set[i],
          k,
          L,
          distance,
          true);
      std::copy(tk_scores.data(), tk_scores.data() + k, top_k_scores[i].data());
      std::copy(tk.data(), tk.data() + k, top_k[i].data());
      num_visited_vertices_ += V.size();
    }
#endif

#if 0
    for (size_t i = 0; i < ::num_vectors(query_set); ++i) {
      auto&& [_top_k_scores, _top_k, V] = greedy_search(
          graph_, feature_vectors_, medoid_, query_set[i], k, l_build_, distance, true);
      std::copy(
          _top_k_scores.data(),
          _top_k_scores.data() + k,
          top_k_scores[i].data());
      std::copy(_top_k.data(), _top_k.data() + k, top_k[i].data());
    }
#endif

    return std::make_tuple(std::move(top_k_scores), std::move(top_k));
  }

  /**
   * @brief Query the index for the top k nearest neighbors of a single
   * query vector
   * @tparam Q Type of query vector
   * @param query_vec The vector to query
   * @param k How many nearest neighbors to return
   * @param l_search How deep to search
   * @return Top k scores and top k ids
   */
  template <query_vector Q, class Distance = sum_of_squares_distance>
  auto query(
      const Q& query_vec,
      size_t k,
      std::optional<size_t> l_search = std::nullopt,
      Distance distance = Distance{}) {
    size_t L = l_search ? *l_search : l_build_;
    auto&& [top_k_scores, top_k, V] = greedy_search(
        graph_, feature_vectors_, medoid_, query_vec, k, L, distance, true);

    return std::make_tuple(std::move(top_k_scores), std::move(top_k));
  }

  auto remove() {
  }

  auto update() {
  }

  constexpr auto dimensions() const {
    return dimensions_;
  }

  constexpr auto ntotal() const {
    return num_vectors_;
  }

  constexpr auto l_build() const {
    return l_build_;
  }

  constexpr auto r_max_degree() const {
    return r_max_degree_;
  }

  /**
   * @brief Write the index to a TileDB group. The group consists of the
   * original feature vectors, and the graph index, which comprises the
   * adjacency scores and adjacency ids, written contiguously, along with an
   * offset (adj_index) to the start of each adjacency list.
   *
   * @param ctx TileDB context
   * @param group_uri The URI of the TileDB group where the index will be saved
   * @param temporal_policy If set, we'll use the end timestamp of the policy as
   * the write timestamp.
   * @param storage_version The storage version to use. If empty, use the most
   * defult version.
   * @return Whether the write was successful
   *
   * @todo Do we need to copy and/or write out the original vectors since
   * those will presumably be in a known array that can be made part of
   * the group?
   */
  auto write_index(
      const tiledb::Context& ctx,
      const std::string& group_uri,
      std::optional<TemporalPolicy> temporal_policy = std::nullopt,
      const std::string& storage_version = "") {
    if (temporal_policy.has_value()) {
      temporal_policy_ = *temporal_policy;
    }
    // metadata: dimension, ntotal, L, R, B, alpha_min, alpha_max, medoid
    // Save as a group: metadata, feature_vectors, graph edges, offsets

    auto write_group = vamana_index_group<vamana_index>(
        ctx,
        group_uri,
        TILEDB_WRITE,
        temporal_policy_,
        storage_version,
        dimensions_);

    // @todo Make this table-driven
    write_group.set_dimensions(dimensions_);
    write_group.set_l_build(l_build_);
    write_group.set_r_max_degree(r_max_degree_);
    write_group.set_alpha_min(alpha_min_);
    write_group.set_alpha_max(alpha_max_);
    write_group.set_medoid(medoid_);

    // When we create an index with Python, we will call write_index() twice,
    // once with empty data and once with the actual data. Here we add custom
    // logic so that during that second call to write_index(), we will overwrite
    // the metadata lists. If we don't do this we will end up with
    // ingestion_timestamps = [0, timestamp] and base_sizes = [0, initial size],
    // whereas indexes created just in Python will end up with
    // ingestion_timestamps = [timestamp] and base_sizes = [initial size]. If we
    // have 2 item lists it causes crashes and subtle issues when we try to
    // modify the index later (i.e. through index.update() / Index.clear()). So
    // here we make sure we end up with the same metadata that Python indexes
    // do.
    if (write_group.get_all_ingestion_timestamps().size() == 1 &&
        write_group.get_previous_ingestion_timestamp() == 0 &&
        write_group.get_all_base_sizes().size() == 1 &&
        write_group.get_previous_base_size() == 0) {
      write_group.set_ingestion_timestamp(temporal_policy_.timestamp_end());
      write_group.set_base_size(::num_vectors(feature_vectors_));
      write_group.set_num_edges(graph_.num_edges());
    } else {
      write_group.append_ingestion_timestamp(temporal_policy_.timestamp_end());
      write_group.append_base_size(::num_vectors(feature_vectors_));
      write_group.append_num_edges(graph_.num_edges());
    }

    // When creating from Python we initially call write_index() at timestamp 0.
    // The goal here is just to create the arrays and save metadata. Return here
    // so that we don't write the arrays, as if we write with timestamp=0 then
    // TileDB Core will interpret this as the current timestamp instead, leading
    // to array fragments created at the current time.
    if (temporal_policy_.timestamp_end() == 0) {
      return true;
    }

    write_matrix(
        ctx,
        feature_vectors_,
        write_group.feature_vectors_uri(),
        0,
        false,
        temporal_policy_);

    write_vector(
        ctx,
        feature_vectors_.raveled_ids(),
        write_group.ids_uri(),
        0,
        false,
        temporal_policy_);

    auto adj_scores = Vector<adjacency_scores_type>(graph_.num_edges());
    auto adj_ids = Vector<id_type>(graph_.num_edges());
    auto adj_index =
        Vector<adjacency_row_index_type>(graph_.num_vertices() + 1);

    size_t edge_offset{0};
    for (size_t i = 0; i < num_vertices(graph_); ++i) {
      adj_index[i] = edge_offset;
      for (auto&& [score, id] : graph_.out_edges(i)) {
        adj_scores[edge_offset] = score;
        adj_ids[edge_offset] = id;
        ++edge_offset;
      }
    }
    adj_index.back() = edge_offset;

    write_vector(
        ctx,
        adj_scores,
        write_group.adjacency_scores_uri(),
        0,
        false,
        temporal_policy_);
    write_vector(
        ctx,
        adj_ids,
        write_group.adjacency_ids_uri(),
        0,
        false,
        temporal_policy_);
    write_vector(
        ctx,
        adj_index,
        write_group.adjacency_row_index_uri(),
        0,
        false,
        temporal_policy_);

    return true;
  }

  static void clear_history(
      const tiledb::Context& ctx,
      const std::string& group_uri,
      uint64_t timestamp) {
    auto write_group =
        vamana_index_group<vamana_index>(ctx, group_uri, TILEDB_WRITE, {});
    write_group.clear_history(timestamp);
  }

  const vamana_index_group<vamana_index>& group() const {
    if (!group_) {
      throw std::runtime_error("[vamana_index@group] No group available");
    }
    return *group_;
  }

  /**
   * @brief Log statistics about the index
   */
  void log_index() {
    _count_data.insert_entry("dimensions", dimensions_);
    _count_data.insert_entry("num_vectors", num_vectors_);
    _count_data.insert_entry("l_build", l_build_);
    _count_data.insert_entry("r_max_degree", r_max_degree_);
    _count_data.insert_entry("num_edges", graph_.num_edges());
    _count_data.insert_entry("num_comps", num_comps());
    _count_data.insert_entry("num_visited_vertices", num_visited_vertices());

    auto&& [min_degree, max_degree] =
        minmax_element(begin(graph_), end(graph_), [](auto&& a, auto&& b) {
          return a.size() < b.size();
        });
    _count_data.insert_entry("min_degree", min_degree->size());
    _count_data.insert_entry("max_degree", max_degree->size());
    _count_data.insert_entry(
        "avg_degree",
        (double)graph_.num_edges() / (double)num_vertices(graph_));
  }

  /**
   * Print debugging information about the index
   */
  void debug_index() {
    auto&& [min_degree, max_degree] =
        minmax_element(begin(graph_), end(graph_), [](auto&& a, auto&& b) {
          return a.size() < b.size();
        });

    size_t counted_edges{0};
    for (size_t i = 0; i < num_vertices(graph_); ++i) {
      counted_edges += graph_.out_edges(i).size();
    }
    std::cout << "# counted edges " << counted_edges << std::endl;
    std::cout << "# num_edges " << graph_.num_edges() << std::endl;
    std::cout << "# min degree " << min_degree->size() << std::endl;
    std::cout << "# max degree " << max_degree->size() << std::endl;
    std::cout << "# avg degree "
              << (double)counted_edges / (double)num_vertices(graph_)
              << std::endl;
  }

  bool compare_cached_metadata(const vamana_index& rhs) const {
    if (dimensions_ != rhs.dimensions_) {
      std::cout << "dimensions_ != rhs.dimensions_" << dimensions_
                << " ! = " << rhs.dimensions_ << std::endl;
      return false;
    }
    if (num_vectors_ != rhs.num_vectors_) {
      std::cout << "num_vectors_ != rhs.num_vectors_" << num_vectors_
                << " ! = " << rhs.num_vectors_ << std::endl;
      return false;
    }
    if (l_build_ != rhs.l_build_) {
      std::cout << "l_build_ != rhs.l_build_" << l_build_
                << " ! = " << rhs.l_build_ << std::endl;
      return false;
    }
    if (r_max_degree_ != rhs.r_max_degree_) {
      std::cout << "r_max_degree_ != rhs.r_max_degree_" << r_max_degree_
                << " ! = " << rhs.r_max_degree_ << std::endl;
      return false;
    }
    if (alpha_min_ != rhs.alpha_min_) {
      std::cout << "alpha_min_ != rhs.alpha_min_" << alpha_min_
                << " ! = " << rhs.alpha_min_ << std::endl;
      return false;
    }
    if (alpha_max_ != rhs.alpha_max_) {
      std::cout << "alpha_max_ != rhs.alpha_max_" << alpha_max_
                << " ! = " << rhs.alpha_max_ << std::endl;
      return false;
    }
    if (medoid_ != rhs.medoid_) {
      std::cout << "medoid_ != rhs.medoid_" << medoid_ << " ! = " << rhs.medoid_
                << std::endl;
      return false;
    }
    if (temporal_policy_.timestamp_start() !=
        rhs.temporal_policy_.timestamp_start()) {
      std::cout << "temporal_policy_.timestamp_start() != "
                   "rhs.temporal_policy_.timestamp_start()"
                << temporal_policy_.timestamp_start()
                << " ! = " << rhs.temporal_policy_.timestamp_start()
                << std::endl;
      return false;
    }
    // Do not compare temporal_policy_.timestamp_end() because if we create an
    // index and then load it with the URI, these timestamps will differ. The
    // first one will have the current timestamp, and the second uint64_t::max.

    return true;
  }

  /**
   * @brief Compare the scores of adjacency lists of two vamana_index
   * objects -- useful for tes`ting
   * @param rhs The other vamana_index to compare with
   * @return True if the adjacency lists are the same, false otherwise
   */
  bool compare_adj_scores(const vamana_index& rhs) {
    for (size_t i = 0; i < num_vertices(graph_); ++i) {
      auto start = graph_.out_edges(i).begin();
      auto end = graph_.out_edges(i).end();
      auto rhs_start = rhs.graph_.out_edges(i).begin();
      auto rhs_end = rhs.graph_.out_edges(i).end();
      if (std::distance(start, end) != std::distance(rhs_start, rhs_end)) {
        std::cout
            << "std::distance(start, end) != std::distance(rhs_start, rhs_end)"
            << std::endl;
        return false;
      }
      for (; start != end; ++start, ++rhs_start) {
        if (std::get<0>(*start) != std::get<0>(*rhs_start)) {
          std::cout << "std::get<0>(*start) != std::get<0>(*rhs_start)"
                    << std::endl;
          return false;
        }
      }
    }
    return true;
  }

  /**
   * @brief Compare the ids of adjacency lists of two vamana_index
   * @param rhs The other vamana_index to compare with
   * @return True if the adjacency lists are the same, false otherwise
   */
  bool compare_adj_ids(const vamana_index& rhs) {
    for (size_t i = 0; i < num_vertices(graph_); ++i) {
      auto start = graph_.out_edges(i).begin();
      auto end = graph_.out_edges(i).end();
      auto rhs_start = rhs.graph_.out_edges(i).begin();
      auto rhs_end = rhs.graph_.out_edges(i).end();
      if (std::distance(start, end) != std::distance(rhs_start, rhs_end)) {
        std::cout
            << "std::distance(start, end) != std::distance(rhs_start, rhs_end)"
            << std::endl;
        return false;
      }
      for (; start != end; ++start, ++rhs_start) {
        if (std::get<1>(*start) != std::get<1>(*rhs_start)) {
          std::cout << "std::get<1>(*start) != std::get<1>(*rhs_start)"
                    << std::endl;
          return false;
        }
      }
    }
    return true;
  }

  /**
   * @brief Compare the feature vectors of two vamana_index objects -- useful
   * for testing.
   * @param rhs The other vamana_index to compare with
   * @return True if the feature vectors are the same, false otherwise
   */
  bool compare_feature_vectors(const vamana_index& rhs) {
    for (size_t i = 0; i < ::num_vectors(feature_vectors_); ++i) {
      for (size_t j = 0; j < ::dimensions(feature_vectors_); ++j) {
        auto lhs_val = feature_vectors_(j, i);
        auto rhs_val = rhs.feature_vectors_(j, i);
        if (lhs_val != rhs_val) {
          std::cout << "lhs_val != rhs_val" << std::endl;
          // return false;
        }
      }
    }

    return std::equal(
        feature_vectors_.data(),
        feature_vectors_.data() +
            ::dimensions(feature_vectors_) * ::num_vectors(feature_vectors_),
        rhs.feature_vectors_.data());
  }

  bool compare_group(const vamana_index& rhs) const {
    return group_->compare_group(*(rhs.group_));
  }

 public:
  void dump_edgelist__(const std::string& str) {
    ::dump_edgelist(str, graph_);
  }
};

/**
 * @brief Variable to count the number of comparisons made during training
 * and querying
 * @tparam feature_type Type of element of feature vectors
 * @tparam id_type Type of id of feature vectors
 */
template <class feature_type, class id_type, class index_type>
size_t vamana_index<feature_type, id_type, index_type>::num_comps_ = 0;

#endif  // TDB_VAMANA_INDEX_H
