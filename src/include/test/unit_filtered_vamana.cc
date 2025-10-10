/**
 * @file   unit_filtered_vamana.cc
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
 * Unit tests for Filtered-Vamana pre-filtering implementation based on
 * "Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters"
 * (Gollapudi et al., WWW 2023)
 */

#include <catch2/catch_all.hpp>
#include <filesystem>
#include <tiledb/tiledb>
#include <unordered_set>
#include "cpos.h"
#include "detail/graph/adj_list.h"
#include "detail/graph/greedy_search.h"
#include "detail/linalg/matrix.h"
#include "index/vamana_index.h"
#include "test/utils/array_defs.h"
#include "test/utils/test_utils.h"

namespace fs = std::filesystem;

/**
 * Test find_medoid() with multiple labels
 *
 * This tests Algorithm 2 from the paper: load-balanced start node selection
 */
TEST_CASE("find_medoid with multiple labels", "[filtered_vamana]") {
  const bool debug = false;

  // Create a simple 2D dataset with 10 points
  size_t num_vectors = 10;
  size_t dimensions = 2;
  auto training_set = ColMajorMatrix<float>(dimensions, num_vectors);

  // Create 10 vectors in 2D space
  // Points 0-2: cluster around (0, 0) with label 0
  // Points 3-5: cluster around (10, 10) with label 1
  // Points 6-9: cluster around (5, 5) with labels 0 and 1 (shared)
  training_set(0, 0) = 0.0f;  training_set(1, 0) = 0.0f;   // label 0
  training_set(0, 1) = 0.5f;  training_set(1, 1) = 0.5f;   // label 0
  training_set(0, 2) = 0.3f;  training_set(1, 2) = 0.2f;   // label 0
  training_set(0, 3) = 10.0f; training_set(1, 3) = 10.0f;  // label 1
  training_set(0, 4) = 10.5f; training_set(1, 4) = 10.5f;  // label 1
  training_set(0, 5) = 10.3f; training_set(1, 5) = 10.2f;  // label 1
  training_set(0, 6) = 5.0f;  training_set(1, 6) = 5.0f;   // labels 0, 1
  training_set(0, 7) = 5.5f;  training_set(1, 7) = 5.5f;   // labels 0, 1
  training_set(0, 8) = 5.3f;  training_set(1, 8) = 5.2f;   // labels 0, 1
  training_set(0, 9) = 5.1f;  training_set(1, 9) = 5.3f;   // labels 0, 1

  // Define filter labels: each vector has a set of label IDs
  std::vector<std::unordered_set<uint32_t>> filter_labels(num_vectors);
  filter_labels[0] = {0};
  filter_labels[1] = {0};
  filter_labels[2] = {0};
  filter_labels[3] = {1};
  filter_labels[4] = {1};
  filter_labels[5] = {1};
  filter_labels[6] = {0, 1};  // shared label
  filter_labels[7] = {0, 1};  // shared label
  filter_labels[8] = {0, 1};  // shared label
  filter_labels[9] = {0, 1};  // shared label

  // Call find_medoid
  auto start_nodes = find_medoid(training_set, filter_labels);

  // Verify we have exactly 2 start nodes (one per unique label)
  CHECK(start_nodes.size() == 2);
  CHECK(start_nodes.count(0) == 1);
  CHECK(start_nodes.count(1) == 1);

  // The start nodes should be from vectors that have these labels
  auto start_for_label_0 = start_nodes[0];
  auto start_for_label_1 = start_nodes[1];

  // Verify start nodes have the correct labels
  CHECK(filter_labels[start_for_label_0].count(0) > 0);
  CHECK(filter_labels[start_for_label_1].count(1) > 0);

  if (debug) {
    std::cout << "Start node for label 0: " << start_for_label_0 << std::endl;
    std::cout << "Start node for label 1: " << start_for_label_1 << std::endl;
  }
}

/**
 * Test filtered_greedy_search_multi_start with multiple start nodes
 *
 * This tests Algorithm 1 from the paper: filter-aware greedy search
 */
TEST_CASE("filtered_greedy_search_multi_start", "[filtered_vamana]") {
  const bool debug = false;

  // Create a simple dataset
  size_t num_vectors = 8;
  size_t dimensions = 2;
  auto db = ColMajorMatrix<float>(dimensions, num_vectors);

  // Create 8 vectors: 4 with label 0, 4 with label 1
  db(0, 0) = 0.0f;  db(1, 0) = 0.0f;   // label 0
  db(0, 1) = 1.0f;  db(1, 1) = 0.0f;   // label 0
  db(0, 2) = 0.0f;  db(1, 2) = 1.0f;   // label 0
  db(0, 3) = 1.0f;  db(1, 3) = 1.0f;   // label 0
  db(0, 4) = 10.0f; db(1, 4) = 10.0f;  // label 1
  db(0, 5) = 11.0f; db(1, 5) = 10.0f;  // label 1
  db(0, 6) = 10.0f; db(1, 6) = 11.0f;  // label 1
  db(0, 7) = 11.0f; db(1, 7) = 11.0f;  // label 1

  // Create filter labels
  std::vector<std::unordered_set<uint32_t>> filter_labels(num_vectors);
  for (size_t i = 0; i < 4; ++i) {
    filter_labels[i] = {0};
  }
  for (size_t i = 4; i < 8; ++i) {
    filter_labels[i] = {1};
  }

  // Create a simple graph connecting nearby points
  using id_type = uint32_t;
  using score_type = float;
  auto graph = detail::graph::adj_list<score_type, id_type>(num_vectors);

  // Connect label 0 vectors
  graph.add_edge(0, 1, sum_of_squares_distance{}(db[0], db[1]));
  graph.add_edge(0, 2, sum_of_squares_distance{}(db[0], db[2]));
  graph.add_edge(1, 3, sum_of_squares_distance{}(db[1], db[3]));
  graph.add_edge(2, 3, sum_of_squares_distance{}(db[2], db[3]));

  // Connect label 1 vectors
  graph.add_edge(4, 5, sum_of_squares_distance{}(db[4], db[5]));
  graph.add_edge(4, 6, sum_of_squares_distance{}(db[4], db[6]));
  graph.add_edge(5, 7, sum_of_squares_distance{}(db[5], db[7]));
  graph.add_edge(6, 7, sum_of_squares_distance{}(db[6], db[7]));

  SECTION("Query with single label filter") {
    // Query for label 0 vectors
    auto query = std::vector<float>{0.5f, 0.5f};
    std::unordered_set<uint32_t> query_filter = {0};
    std::vector<id_type> start_nodes = {0};  // Start from vector 0

    size_t k_nn = 2;
    uint32_t L = 4;

    auto&& [top_k_scores, top_k, visited] = filtered_greedy_search_multi_start(
        graph, db, filter_labels, start_nodes, query, query_filter, k_nn, L);

    // All returned results should have label 0
    for (size_t i = 0; i < k_nn; ++i) {
      if (top_k[i] != std::numeric_limits<id_type>::max()) {
        CHECK(filter_labels[top_k[i]].count(0) > 0);
      }
    }

    // Should NOT return any vectors with label 1
    for (size_t i = 0; i < k_nn; ++i) {
      if (top_k[i] != std::numeric_limits<id_type>::max()) {
        CHECK(top_k[i] < 4);  // Vectors 0-3 have label 0
      }
    }

    if (debug) {
      std::cout << "Top-k results for label 0: ";
      for (size_t i = 0; i < k_nn; ++i) {
        std::cout << top_k[i] << " ";
      }
      std::cout << std::endl;
    }
  }

  SECTION("Multi-start with multiple start nodes") {
    // Use two start nodes
    std::vector<id_type> start_nodes = {0, 2};
    std::unordered_set<uint32_t> query_filter = {0};
    auto query = std::vector<float>{0.5f, 0.5f};

    size_t k_nn = 3;
    uint32_t L = 5;

    auto&& [top_k_scores, top_k, visited] = filtered_greedy_search_multi_start(
        graph, db, filter_labels, start_nodes, query, query_filter, k_nn, L);

    // Verify all results match the filter
    for (size_t i = 0; i < k_nn; ++i) {
      if (top_k[i] != std::numeric_limits<id_type>::max()) {
        CHECK(filter_labels[top_k[i]].count(0) > 0);
      }
    }

    if (debug) {
      std::cout << "Visited " << visited.size() << " nodes" << std::endl;
    }
  }
}

/**
 * Test filtered_robust_prune preserves label connectivity
 *
 * This tests Algorithm 3 from the paper: filter-aware pruning
 */
TEST_CASE("filtered_robust_prune preserves label connectivity", "[filtered_vamana]") {
  const bool debug = false;

  // Create a simple dataset
  size_t num_vectors = 6;
  size_t dimensions = 2;
  auto db = ColMajorMatrix<float>(dimensions, num_vectors);

  // Create vectors with different labels
  db(0, 0) = 0.0f;  db(1, 0) = 0.0f;   // label 0
  db(0, 1) = 1.0f;  db(1, 1) = 0.0f;   // label 1
  db(0, 2) = 2.0f;  db(1, 2) = 0.0f;   // labels 0, 1 (shared)
  db(0, 3) = 3.0f;  db(1, 3) = 0.0f;   // label 0
  db(0, 4) = 4.0f;  db(1, 4) = 0.0f;   // label 1
  db(0, 5) = 5.0f;  db(1, 5) = 0.0f;   // label 0

  // Create filter labels
  std::vector<std::unordered_set<uint32_t>> filter_labels(num_vectors);
  filter_labels[0] = {0};
  filter_labels[1] = {1};
  filter_labels[2] = {0, 1};  // shared - important for connectivity
  filter_labels[3] = {0};
  filter_labels[4] = {1};
  filter_labels[5] = {0};

  using id_type = uint32_t;
  using score_type = float;
  auto graph = detail::graph::adj_list<score_type, id_type>(num_vectors);

  // Test pruning from node 2 (which has labels 0 and 1)
  size_t p = 2;
  std::vector<id_type> candidates = {0, 1, 3, 4, 5};  // All neighbors except p itself
  float alpha = 1.2f;
  size_t R = 3;  // Max degree

  filtered_robust_prune(
      graph, db, filter_labels, p, candidates, alpha, R, sum_of_squares_distance{});

  // After pruning, node 2 should have at most R edges
  CHECK(graph.out_degree(p) <= R);

  // The pruned edges should maintain connectivity to both label 0 and label 1
  bool has_label_0_neighbor = false;
  bool has_label_1_neighbor = false;

  for (auto&& [score, neighbor] : graph.out_edges(p)) {
    if (filter_labels[neighbor].count(0) > 0) {
      has_label_0_neighbor = true;
    }
    if (filter_labels[neighbor].count(1) > 0) {
      has_label_1_neighbor = true;
    }
  }

  // Since p has both labels, it should maintain edges to both label types
  // (This is the key property of filtered_robust_prune)
  CHECK(has_label_0_neighbor);
  CHECK(has_label_1_neighbor);

  if (debug) {
    std::cout << "Node " << p << " has " << graph.out_degree(p) << " edges after pruning:" << std::endl;
    for (auto&& [score, neighbor] : graph.out_edges(p)) {
      std::cout << "  -> " << neighbor << " (labels: ";
      for (auto label : filter_labels[neighbor]) {
        std::cout << label << " ";
      }
      std::cout << ")" << std::endl;
    }
  }
}

/**
 * End-to-end test: Train and query filtered Vamana index
 */
TEST_CASE("filtered vamana index end-to-end", "[filtered_vamana]") {
  const bool debug = false;

  // Create a dataset with two clusters, each with different labels
  size_t num_vectors = 20;
  size_t dimensions = 2;
  auto training_set = ColMajorMatrix<float>(dimensions, num_vectors);
  std::vector<uint64_t> ids(num_vectors);
  std::iota(begin(ids), end(ids), 0);

  // Cluster 1 (label "dataset_A"): 10 points around (0, 0)
  for (size_t i = 0; i < 10; ++i) {
    training_set(0, i) = static_cast<float>(i % 3);
    training_set(1, i) = static_cast<float>(i / 3);
  }

  // Cluster 2 (label "dataset_B"): 10 points around (10, 10)
  for (size_t i = 10; i < 20; ++i) {
    training_set(0, i) = 10.0f + static_cast<float>((i - 10) % 3);
    training_set(1, i) = 10.0f + static_cast<float>((i - 10) / 3);
  }

  // Create filter labels using enumeration IDs
  // Label 0 = "dataset_A", Label 1 = "dataset_B"
  std::vector<std::unordered_set<uint32_t>> filter_labels(num_vectors);
  for (size_t i = 0; i < 10; ++i) {
    filter_labels[i] = {0};  // "dataset_A"
  }
  for (size_t i = 10; i < 20; ++i) {
    filter_labels[i] = {1};  // "dataset_B"
  }

  // Build filtered index
  uint32_t l_build = 10;
  uint32_t r_max_degree = 5;
  auto idx = vamana_index<float, uint64_t>(num_vectors, l_build, r_max_degree);

  // Train with filter labels
  idx.train(training_set, ids, filter_labels);

  SECTION("Query with filter for dataset_A") {
    // Query near cluster 1
    auto query = std::vector<float>{0.5f, 0.5f};
    std::unordered_set<uint32_t> query_filter = {0};  // Label for "dataset_A"

    size_t k = 5;
    auto&& [top_k_scores, top_k] = idx.query(query, k, std::nullopt, query_filter);

    // All results should be from cluster 1 (indices 0-9)
    for (size_t i = 0; i < k; ++i) {
      if (top_k[i] != std::numeric_limits<uint64_t>::max()) {
        CHECK(top_k[i] < 10);
      }
    }

    if (debug) {
      std::cout << "Query results for dataset_A: ";
      for (size_t i = 0; i < k; ++i) {
        std::cout << top_k[i] << " ";
      }
      std::cout << std::endl;
    }
  }

  SECTION("Query with filter for dataset_B") {
    // Query near cluster 2
    auto query = std::vector<float>{10.5f, 10.5f};
    std::unordered_set<uint32_t> query_filter = {1};  // Label for "dataset_B"

    size_t k = 5;
    auto&& [top_k_scores, top_k] = idx.query(query, k, std::nullopt, query_filter);

    // All results should be from cluster 2 (indices 10-19)
    for (size_t i = 0; i < k; ++i) {
      if (top_k[i] != std::numeric_limits<uint64_t>::max()) {
        CHECK(top_k[i] >= 10);
        CHECK(top_k[i] < 20);
      }
    }

    if (debug) {
      std::cout << "Query results for dataset_B: ";
      for (size_t i = 0; i < k; ++i) {
        std::cout << top_k[i] << " ";
      }
      std::cout << std::endl;
    }
  }

  SECTION("Query without filter returns mixed results") {
    // Query in the middle
    auto query = std::vector<float>{5.0f, 5.0f};
    size_t k = 10;

    // Query WITHOUT filter - should return from both clusters
    auto&& [top_k_scores, top_k] = idx.query(query, k);

    // Results can be from either cluster (we just check they're valid)
    for (size_t i = 0; i < k; ++i) {
      if (top_k[i] != std::numeric_limits<uint64_t>::max()) {
        CHECK(top_k[i] < 20);
      }
    }

    if (debug) {
      std::cout << "Query results without filter: ";
      for (size_t i = 0; i < k; ++i) {
        std::cout << top_k[i] << " ";
      }
      std::cout << std::endl;
    }
  }
}

/**
 * Test that filtered index maintains backward compatibility
 */
TEST_CASE("filtered vamana backward compatibility", "[filtered_vamana]") {
  // Create a simple dataset
  size_t num_vectors = 10;
  size_t dimensions = 2;
  auto training_set = ColMajorMatrix<float>(dimensions, num_vectors);
  std::vector<uint64_t> ids(num_vectors);
  std::iota(begin(ids), end(ids), 0);

  for (size_t i = 0; i < num_vectors; ++i) {
    training_set(0, i) = static_cast<float>(i);
    training_set(1, i) = static_cast<float>(i);
  }

  uint32_t l_build = 5;
  uint32_t r_max_degree = 3;

  SECTION("Train without filters (backward compatible)") {
    auto idx = vamana_index<float, uint64_t>(num_vectors, l_build, r_max_degree);

    // Train WITHOUT filter labels (empty vector)
    idx.train(training_set, ids);  // No filter_labels parameter

    // Query should work normally
    auto query = std::vector<float>{2.0f, 2.0f};
    size_t k = 3;
    auto&& [top_k_scores, top_k] = idx.query(query, k);

    // Should get valid results
    CHECK(top_k[0] != std::numeric_limits<uint64_t>::max());
  }
}
