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

#include "detail/graph/adj_list.h"
#include "detail/graph/graph_utils.h"
#include "detail/linalg/tdb_matrix_with_ids.h"
#include "detail/linalg/vector.h"
#include "index/vamana_group.h"
#include "scoring.h"
#include "stats.h"
#include "utils/fixed_min_heap.h"
#include "utils/print_types.h"

#include <tiledb/tiledb>

#include <tiledb/group_experimental.h>

static bool noisy = false;
[[maybe_unused]] static void set_noisy(bool b) {
  noisy = b;
}

/**
 * @brief Truncated best-first search
 * @tparam Distance The distance function used to compare vectors
 * @param graph Graph to be searched
 * @param source start node index
 * @param query query node index
 * @param k result size
 * @param L search list size, L >= k
 * @return result set ell containing k-approximate nearest neighbors to query
 * and set vee containing all the visited nodes
 *
 * Per the DiskANN paper, the algorithm is as follows:
 * 1. Initialize the result list with the source node and visited list with
 *    empty
 * 2. While the result list \ visited list is not empty
 *    a. Find p* in the result list \ visited list with the smallest distance to
 *       the query
 *    b. update the result list with the out neighbors of p*
 *    c. Add p* to the visited list d. If size of the result list > L, trim
 *      the result list to keep L closest points to query
 * 3. Copy the result list to the output
 *
 * This is essentially a best-first search with a fixed size priority queue
 *
 * @todo -- add a `SearchPath `template parameter to determine whether to
 * return the top k results of the search or just the path taken.
 * @todo -- remove printf debugging code
 * @todo -- would it be more efficient somehow to process multiple queries?
 */
template </* SearchPath SP, */ class Distance = sum_of_squares_distance>
auto greedy_search(
    auto&& graph,
    auto&& db,
    typename std::decay_t<decltype(graph)>::id_type source,
    auto&& query,
    size_t k_nn,
    size_t L,
    Distance distance = Distance{}) {
  // using feature_type = typename std::decay_t<decltype(graph)>::feature_type;
  using id_type = typename std::decay_t<decltype(graph)>::id_type;
  using score_type = typename std::decay_t<decltype(graph)>::score_type;

  static_assert(std::integral<id_type>);

  assert(L >= k_nn);

  std::unordered_set<id_type> visited_vertices;
  auto visited = [&visited_vertices](auto&& v) {
    return visited_vertices.contains(v);
  };

  auto result = k_min_heap<score_type, id_type>{L};  // 𝓛: |𝓛| <= L
  // auto result = std::set<id_type>{};
  auto q1 = k_min_heap<score_type, id_type>{L};  // 𝓛 \ 𝓥
  auto q2 = k_min_heap<score_type, id_type>{L};  // 𝓛 \ 𝓥

  // 𝓛 <- {s} and 𝓥 <- ∅
  result.insert(distance(db[source], query), source);

  // q1 = 𝓛 \ 𝓥 = {s}
  q1.insert(distance(db[source], query), source);

  size_t counter{0};

  // while 𝓛 \ 𝓥 ≠ ∅
  while (!q1.empty()) {
    if (noisy) {
      std::cout << "\n:::: " << counter++ << " ::::" << std::endl;
      debug_min_heap(q1, "q1: ", 1);
    }

    // p* <- argmin_{p ∈ 𝓛 \ 𝓥} distance(p, q)

    // Although we use the name `k_min_heap` -- it actually stores a finite
    // number of elements in a max heap (we remove the max element
    // every time we have a smaller element to insert).  Since we are using
    // a `k_min_heap` for q1, to get and pop the min element, we have to
    // change it to a min heap, get the min element, and then change it back
    // to a max heap.
    // @todo -- There must be a better way of doing this

    // Change q1 into a min_heap
    std::make_heap(begin(q1), end(q1), [](auto&& a, auto&& b) {
      return std::get<0>(a) > std::get<0>(b);
    });

    // Get and pop the min element
    std::pop_heap(begin(q1), end(q1), [](auto&& a, auto&& b) {
      return std::get<0>(a) > std::get<0>(b);
    });

    auto [s_star, p_star] = q1.back();
    q1.pop_back();

    if (noisy) {
      std::cout << "p*: " << p_star
                << " --  distance = " << distance(db[p_star], query)
                << std::endl;
    }

    // Change back to max heap
    std::make_heap(begin(q1), end(q1), [](auto&& a, auto&& b) {
      return std::get<0>(a) < std::get<0>(b);
    });

    if (visited(p_star)) {
      continue;
    }

    // V <- V \cup {p*} ; L\V <- L\V \ p*
    visited_vertices.insert(p_star);

    if (noisy) {
      debug_vector(visited_vertices, "visited_vertices: ");
      debug_min_heap(graph.out_edges(p_star), "Nout(p*): ", 1);
    }

    // q2 <- L \ V
    // @todo Is there a better way to do this?  By somehow removing
    // elements from q1?  We can probably make the insertion into q2
    // more efficient -- have a batch interface to insert into q2
    for (auto&& [s, p] : result) {
      if (!visited(p)) {
        q2.insert(s, p);
      }
    }

    // L <- L \cup Nout(p*)  ; L \ V <- L \ V \cup Nout(p*)

    // In looking at profiling, a majority of time is spent in this loop,
    // in visited() and in result.insert()
    for (auto&& [_, p] : graph.out_edges(p_star)) {
      // assert(p != p_star);
      if (!visited(p)) {
        auto score = distance(db[p], query);

        // unique id or not does not seem to make a difference
        // @todo (actually it does, but shouldn't need it -- need to
        // investigate) if (result.template insert /*<unique_id>*/ (score, p)) {
        if (result.template insert<unique_id>(score, p)) {
          q2.insert(score, p);
        }
      }
    }

    if (noisy) {
      debug_min_heap(result, "result, aka Ell: ", 1);
      debug_min_heap(result, "result, aka Ell: ", 0);
    }

    q1.swap(q2);
    q2.clear();
  }

  // auto top_k = Vector<id_type>(k_nn);
  // auto top_k_scores = Vector<score_type>(k_nn);
  auto top_k = std::vector<id_type>(k_nn);
  auto top_k_scores = std::vector<score_type>(k_nn);

  get_top_k_with_scores_from_heap(result, top_k, top_k_scores);
  return std::make_tuple(
      std::move(top_k_scores), std::move(top_k), std::move(visited_vertices));
}

/**
 * @brief RobustPrune(p, vee, alpha, R)
 * @tparam I index type
 * @tparam Distance distance functor
 * @param graph Graph
 * @param p point \in P
 * @param V candidate set
 * @param alpha distance threshold >= 1
 * @param R Degree bound
 *
 * From the DiskANN paper:
 * V <- (V \cup Nout(p) \ p
 * Nout(p) < 0
 * while (!V.empty()) {
 *  Find p* \in V with smallest distance to p
 *  Nout(p) <- Nout(p) \cup p*
 *  if size(Nout(p)) == R {
 *    break
 *  }
 *  for pp \in V {
 *  if alpha * distance(p*, pp) <= distance(p, pp) {
 *    remove pp from V
 *  }
 * }
 */
template <class I = size_t, class Distance = sum_of_squares_distance>
auto robust_prune(
    auto&& graph,
    auto&& db,
    I p,
    auto&& V_in,
    float alpha,
    size_t R,
    Distance distance = Distance{}) {
  constexpr bool noisy = false;

  // using feature_type = typename std::decay_t<decltype(graph)>::feature_type;
  using id_type = typename std::decay_t<decltype(graph)>::id_type;
  using score_type = typename std::decay_t<decltype(graph)>::score_type;

  std::unordered_map<id_type, score_type> V_map;

  for (auto&& v : V_in) {
    if (v != p) {
      auto score = distance(db[v], db[p]);
      V_map.try_emplace(v, score);
    }
  }

  // V <- (V \cup Nout(p) \ p
  for (auto&& [ss, pp] : graph.out_edges(p)) {
    // assert(pp != p);
    if (pp != p) {
      // assert(ss == distance(db[p], db[pp]));
      V_map.try_emplace(pp, ss);
    }
  }

  std::vector<std::tuple<score_type, id_type>> V;
  V.reserve(V_map.size() + R);
  std::vector<std::tuple<score_type, id_type>> new_V;
  new_V.reserve(V_map.size() + R);

  for (auto&& v : V_map) {
    V.emplace_back(v.second, v.first);
  }

  if (noisy) {
    debug_min_heap(V, "V: ", 1);
  }

  // Nout(p) <- 0
  graph.out_edges(p).clear();

  size_t counter{0};
  // while V != 0
  while (!V.empty()) {
    if (noisy) {
      std::cout << "\n:::: " << counter++ << " ::::" << std::endl;
    }

    // p* <- argmin_{pp \in V} distance(p, pp)
    auto&& [s_star, p_star] =
        *(std::min_element(begin(V), end(V), [](auto&& a, auto&& b) {
          return std::get<0>(a) < std::get<0>(b);
        }));

    assert(p_star != p);
    if (noisy) {
      std::cout << "::::" << p_star << std::endl;
      debug_min_heap(V, "V: ", 1);
    }

    // Nout(p) <- Nout(p) \cup p*
    graph.add_edge(p, p_star, s_star);

    if (noisy) {
      debug_min_heap(graph.out_edges(p), "Nout(p): ", 1);
    }

    if (graph.out_edges(p).size() == R) {
      break;
    }

    // For p' in V
    for (auto&& [ss, pp] : V) {
      // if alpha * d(p*, p') <= d(p, p')
      // assert(ss == distance(db[p], db[pp]));
      if (alpha * distance(db[p_star], db[pp]) <= ss) {
        // V.erase({ss, pp});
        ;
      } else {
        if (pp != p) {
          new_V.emplace_back(ss, pp);
        }
      }
    }
    if (noisy) {
      debug_min_heap(V, "after prune V: ", 1);
    }

    std::swap(V, new_V);
    new_V.clear();
  }
}

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
 * @brief Index class for vamana search
 * @tparam feature_type Type of the elements in the feature vectors
 * @tparam id_type Type of the ids of the feature vectors
 */
template <class FeatureType, class IdType, class IndexType = uint64_t>
class vamana_index {
 public:
  using feature_type = FeatureType;
  using id_type = IdType;
  using adjacency_row_index_type = IndexType;
  using score_type = float;

 private:
  /****************************************************************************
   * Index group information
   ****************************************************************************/

  /** The timestamp at which the index was created */
  uint64_t timestamp_{0};

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
  uint64_t dimension_{0};
  uint64_t num_vectors_{0};
  uint64_t num_edges_{0};

  /** The graph representing the index over `feature_vectors_` */
  ::detail::graph::adj_list<score_type, id_type> graph_;

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
  uint64_t L_build_{0};       // diskANN paper says default = 100
  uint64_t R_max_degree_{0};  // diskANN paper says default = 64
  uint64_t B_backtrack_{0};   //
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
      size_t L,
      size_t R,
      size_t B = 0,
      size_t timestamp = 0):
      timestamp_{
          (timestamp == 0) ?
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count() :
              timestamp}
      , num_vectors_{num_nodes}
      , graph_{num_vectors_}
      , L_build_{L}
      , R_max_degree_{R}
      , B_backtrack_{B == 0 ? L_build_ : B} {
  }

  /**
   * @brief Load a vamana graph index from a TileDB group
   * @param ctx TileDB context
   * @param group_uri URI of the group containing the index
   */
  vamana_index(tiledb::Context ctx, const std::string& uri)
      : group_{std::make_unique<vamana_index_group<vamana_index>>(
            *this, ctx, uri, TILEDB_READ, timestamp_)} {
    if (timestamp_ == 0) {
      timestamp_ = group_->get_previous_ingestion_timestamp();
    }

    // @todo Make this table-driven
    dimension_ = group_->get_dimension();
    num_vectors_ = group_->get_base_size();
    num_edges_ = group_->get_num_edges();
    L_build_ = group_->get_L_build();
    R_max_degree_ = group_->get_R_max_degree();
    B_backtrack_ = group_->get_B_backtrack();
    alpha_min_ = group_->get_alpha_min();
    alpha_max_ = group_->get_alpha_max();
    medoid_ = group_->get_medoid();

    feature_vectors_ =
        std::move(tdbColMajorPreLoadMatrixWithIds<feature_type, id_type>(
            group_->cached_ctx(),
            group_->feature_vectors_uri(),
            group_->feature_vector_ids_uri(),
            dimension_,
            num_vectors_,
            0,
            timestamp_));

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

    auto adj_scores = read_vector<score_type>(
        group_->cached_ctx(),
        group_->adjacency_scores_uri(),
        0,
        num_edges_,
        timestamp_);
    auto adj_ids = read_vector<id_type>(
        group_->cached_ctx(),
        group_->adjacency_ids_uri(),
        0,
        num_edges_,
        timestamp_);
    auto adj_index = read_vector<adjacency_row_index_type>(
        group_->cached_ctx(),
        group_->adjacency_row_index_uri(),
        0,
        num_vectors_ + 1,
        timestamp_);

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
      class Distance = sum_of_squares_distance>
  void train(const Array& training_set, Distance distance = Distance{}) {
    feature_vectors_ = std::move(ColMajorMatrixWithIds<feature_type, id_type>(
        ::dimension(training_set), ::num_vectors(training_set)));
    std::copy(
        training_set.data(),
        training_set.data() +
            ::dimension(training_set) * ::num_vectors(training_set),
        feature_vectors_.data());
    // TODO(paris): Read IDs from training_set if they exist.
    auto ids = std::vector<id_type>(::num_vectors(training_set));
    std::iota(ids.begin(), ids.end(), 0);
    std::copy(ids.begin(), ids.end(), feature_vectors_.ids().begin());

    dimension_ = ::dimension(feature_vectors_);
    num_vectors_ = ::num_vectors(feature_vectors_);
    // graph_ = ::detail::graph::init_random_adj_list<feature_type, id_type>(
    //     feature_vectors_, R_max_degree_);

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
        auto&& [top_k_scores, top_k, visited] = greedy_search(
            graph_,
            feature_vectors_,
            medoid_,
            feature_vectors_[p],
            1,
            L_build_,
            distance);
        total_visited += visited.size();

        robust_prune(
            graph_,
            feature_vectors_,
            p,
            visited,
            alpha,
            R_max_degree_,
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

            if (size(tmp) > R_max_degree_) {
              robust_prune(
                  graph_,
                  feature_vectors_,
                  j,
                  tmp,
                  alpha,
                  R_max_degree_,
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

  /**
   * @brief Query the index for the top k nearest neighbors of the query set
   * @tparam Q Type of query set
   * @param query_set Container of query vectors
   * @param k How many nearest neighbors to return
   * @param opt_L How deep to search
   * @return Tuple of top k scores and top k ids
   */
  template <query_vector_array Q, class Distance = sum_of_squares_distance>
  auto query(
      const Q& query_set,
      size_t k,
      std::optional<size_t> opt_L = std::nullopt,
      Distance distance = Distance{}) {
    scoped_timer __{tdb_func__ + std::string{" (outer)"}};

    size_t L = opt_L ? *opt_L : L_build_;
    // L = std::min<size_t>(L, L_build_);

    auto top_k = ColMajorMatrix<id_type>(k, ::num_vectors(query_set));
    auto top_k_scores = ColMajorMatrix<score_type>(k, ::num_vectors(query_set));

#if 0
    // Parallelized implementation -- we stay single-threaded for now
    // for purposes of comparison
    size_t nthreads = std::thread::hardware_concurrency();
    auto par = stdx::execution::indexed_parallel_policy{nthreads};

    stdx::range_for_each(std::move(par), query_set, [&](auto&& query_vec, auto n, auto i) {
      auto&& [tk_scores, tk, V] = greedy_search(
          graph_, feature_vectors_, medoid_, query_vec, k, L);
      std::copy(tk_scores.data(), tk_scores.data() + k, top_k_scores[i].data());
      std::copy(tk.data(), tk.data() + k, top_k[i].data());
    });
#else
    for (size_t i = 0; i < num_vectors(query_set); ++i) {
      auto&& [tk_scores, tk, V] = greedy_search(
          graph_, feature_vectors_, medoid_, query_set[i], k, L, distance);
      std::copy(tk_scores.data(), tk_scores.data() + k, top_k_scores[i].data());
      std::copy(tk.data(), tk.data() + k, top_k[i].data());
      num_visited_vertices_ += V.size();
    }
#endif

#if 0
    for (size_t i = 0; i < ::num_vectors(query_set); ++i) {
      auto&& [_top_k_scores, _top_k, V] = greedy_search(
          graph_, feature_vectors_, medoid_, query_set[i], k, L_build_);
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
   * @param opt_L How deep to search
   * @return Top k scores and top k ids
   */
  template <query_vector Q, class Distance = sum_of_squares_distance>
  auto query(
      const Q& query_vec,
      size_t k,
      std::optional<size_t> opt_L = std::nullopt,
      Distance distance = Distance{}) {
    size_t L = opt_L ? *opt_L : L_build_;

    auto&& [top_k_scores, top_k, V] = greedy_search(
        graph_, feature_vectors_, medoid_, query_vec, k, L, distance);

    return std::make_tuple(std::move(top_k_scores), std::move(top_k));
  }

  auto remove() {
  }

  auto update() {
  }

  constexpr auto dimension() const {
    return dimension_;
  }

  constexpr auto ntotal() const {
    return num_vectors_;
  }

  /**
   * @brief Write the index to a TileDB group
   * @param group_uri The URI of the TileDB group where the index will be saved
   * @param overwrite Whether to overwrite an existing group
   * @return Whether the write was successful
   *
   * The group consists of the original feature vectors, and the graph index,
   * which comprises the adjacency scores and adjacency ids, written
   * contiguously, along with an offset (adj_index) to the start of each
   * adjacency list.
   *
   * @todo Do we need to copy and/or write out the original vectors since
   * those will presumably be in a known array that can be made part of
   * the group?
   */
  auto write_index(
      const tiledb::Context& ctx,
      const std::string& group_uri,
      bool overwrite = false) const {
    // metadata: dimension, ntotal, L, R, B, alpha_min, alpha_max, medoid
    // Save as a group: metadata, feature_vectors, graph edges, offsets

    tiledb::VFS vfs(ctx);
    if (vfs.is_dir(group_uri)) {
      if (overwrite == false) {
        return false;
      }
      vfs.remove_dir(group_uri);
    }

    auto write_group = vamana_index_group(*this, ctx, group_uri, TILEDB_WRITE);

    // @todo Make this table-driven
    write_group.set_dimension(dimension_);
    write_group.set_L_build(L_build_);
    write_group.set_R_max_degree(R_max_degree_);
    write_group.set_B_backtrack(B_backtrack_);
    write_group.set_alpha_min(alpha_min_);
    write_group.set_alpha_max(alpha_max_);
    write_group.set_medoid(medoid_);

    write_group.append_ingestion_timestamp(timestamp_);
    write_group.append_base_size(::num_vectors(feature_vectors_));
    write_group.append_num_edges(graph_.num_edges());

    write_matrix(
        ctx,
        feature_vectors_,
        write_group.feature_vectors_uri(),
        0,
        false,
        timestamp_);

    write_vector(
        ctx,
        feature_vectors_.ids(),
        write_group.feature_vector_ids_uri(),
        0,
        false,
        timestamp_);

    auto adj_scores = Vector<score_type>(graph_.num_edges());
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
        timestamp_);
    write_vector(
        ctx, adj_ids, write_group.adjacency_ids_uri(), 0, false, timestamp_);
    write_vector(
        ctx,
        adj_index,
        write_group.adjacency_row_index_uri(),
        0,
        false,
        timestamp_);

    return true;
  }

  /**
   * @brief Log statistics about the index
   */
  void log_index() {
    _count_data.insert_entry("dimension", dimension_);
    _count_data.insert_entry("num_vectors", num_vectors_);
    _count_data.insert_entry("L_build", L_build_);
    _count_data.insert_entry("R_max_degree", R_max_degree_);
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
    if (dimension_ != rhs.dimension_) {
      std::cout << "dimension_ != rhs.dimension_" << dimension_
                << " ! = " << rhs.dimension_ << std::endl;
      return false;
    }
    if (num_vectors_ != rhs.num_vectors_) {
      std::cout << "num_vectors_ != rhs.num_vectors_" << num_vectors_
                << " ! = " << rhs.num_vectors_ << std::endl;
      return false;
    }
    if (L_build_ != rhs.L_build_) {
      std::cout << "L_build_ != rhs.L_build_" << L_build_
                << " ! = " << rhs.L_build_ << std::endl;
      return false;
    }
    if (R_max_degree_ != rhs.R_max_degree_) {
      std::cout << "R_max_degree_ != rhs.R_max_degree_" << R_max_degree_
                << " ! = " << rhs.R_max_degree_ << std::endl;
      return false;
    }
    if (B_backtrack_ != rhs.B_backtrack_) {
      std::cout << "B_backtrack_ != rhs.B_backtrack_" << B_backtrack_
                << " ! = " << rhs.B_backtrack_ << std::endl;
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
      for (size_t j = 0; j < ::dimension(feature_vectors_); ++j) {
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
            ::dimension(feature_vectors_) * ::num_vectors(feature_vectors_),
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
