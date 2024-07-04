/**
 * @file   unit_vamana.cc
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

#include <catch2/catch_all.hpp>

#include "cpos.h"
#include "detail/flat/qv.h"
#include "detail/graph/adj_list.h"
#include "detail/graph/diskann.h"
#include "detail/graph/nn-descent.h"
#include "detail/graph/nn-graph.h"
#include "detail/linalg/matrix.h"
#include "detail/linalg/tdb_io.h"
#include "index/vamana_index.h"
#include "test/utils/array_defs.h"
#include "test/utils/gen_graphs.h"
#include "test/utils/query_common.h"
#include "test/utils/test_utils.h"
#include "test/utils/tiny_graphs.h"
#include "utils/logging.h"
#include "utils/utils.h"

#include <filesystem>
namespace fs = std::filesystem;

#include <tiledb/tiledb>

TEST_CASE("diskann", "[vamana]") {
  const bool debug = false;
  const bool noisy = false;

  set_noisy(noisy);

  for (auto&& s :
       {diskann_test_data_file,
        diskann_disk_index,
        diskann_mem_index,
        diskann_truth_disk_layout,
        diskann_truth_index_data}) {
    if (debug) {
      std::cout << s << std::endl;
    }
    CHECK(local_file_exists(s));
  }

  std::ifstream binary_file(diskann_mem_index, std::ios::binary);
  REQUIRE(binary_file.is_open());

  uint64_t index_file_size;
  uint32_t max_degree;
  uint32_t medioid_;
  uint64_t vamana_frozen_num;

  binary_file.read((char*)&index_file_size, 8);
  binary_file.read((char*)&max_degree, 4);
  binary_file.read((char*)&medioid_, 4);
  binary_file.read((char*)&vamana_frozen_num, 8);

  fs::path p = diskann_mem_index;
  auto on_disk_file_size = fs::file_size(p);
  CHECK(on_disk_file_size == index_file_size);

  CHECK(max_degree == 4);
  CHECK(medioid_ == 72);
  CHECK(vamana_frozen_num == 0);

  if (debug) {
    std::cout << "index_file_size " << index_file_size << std::endl;
    std::cout << "max_degree " << max_degree << std::endl;
    std::cout << "medoid " << medioid_ << std::endl;
    std::cout << "vamana_frozen_num " << vamana_frozen_num << std::endl;
  }

  binary_file.close();

  auto g = read_diskann_mem_index(diskann_mem_index);
  CHECK(g.size() == 256);

  for (size_t i = 0; i < g.size(); ++i) {
    CHECK(g.out_degree(i) == 4);
  }

  auto h = read_diskann_mem_index_with_scores(
      diskann_mem_index, diskann_test_data_file);
  CHECK(h.size() == 256);

  for (size_t i = 0; i < h.size(); ++i) {
    CHECK(h.out_degree(i) == 4);
  }
  for (size_t i = 0; i < h.size(); ++i) {
    CHECK(std::equal(
        begin(g.out_edges(i)),
        end(g.out_edges(i)),
        begin(h.out_edges(i)),
        [](auto&& a, auto&& b) { return a == std::get<1>(b); }));
  }

  auto f = read_diskann_data(diskann_test_data_file);
  CHECK(num_vectors(f) == 256);
  CHECK(dimensions(f) == 128);
  CHECK(f.data() != nullptr);
  CHECK(!std::equal(
      f.data(), f.data() + 256 * 128, std::vector<float>(128 * 256, 0).data()));

  CHECK(f.num_rows() == 128);
  CHECK(f.num_cols() == 256);

  CHECK(l2_distance(f[0], f[72]) == 125678);
  {
    auto n = num_vectors(f);
    CHECK(n != 0);
    CHECK(n == 256);
    CHECK(f[0].size() == 128);
    CHECK(dimensions(f) == 128);

    auto centroid = Vector<float>(f[0].size());
    std::fill(begin(centroid), end(centroid), 0.0);
    for (size_t j = 0; j < n; ++j) {
      auto p = f[j];
      for (size_t i = 0; i < p.size(); ++i) {
        centroid[i] += p[i];
      }
    }
    float sum = 0.0;
    for (size_t i = 0; i < centroid.size(); ++i) {
      sum += std::abs(centroid[i]);
      centroid[i] /= (float)num_vectors(f);
    }
    CHECK(sum > 0);

    std::vector<float> tmp{begin(centroid), end(centroid)};
    auto min_score = std::numeric_limits<float>::max();
    auto med = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < n; ++i) {
      auto score = l2_distance(f[i], centroid);
      if (score < min_score) {
        min_score = score;
        med = i;
      }
    }
    CHECK(med != std::numeric_limits<size_t>::max());
  }

  auto med = ::medoid(f);

  if (debug) {
    std::cout << "med " << med << std::endl;
    std::cout << "f[0] - f[72] = " << l2_distance(f[0], f[72]) << std::endl;
  }

  CHECK(med == 72);

  //  if (debug) {
  //    tiledb::Context ctx;
  //    write_matrix(ctx, f, "/tmp/diskann_test_data_file.tdb");
  //  }
}

TEST_CASE("small256 build index", "[vamana]") {
  const bool debug = false;
  const bool noisy = false;

  set_noisy(noisy);

  std::vector<size_t> vectors_of_interest{
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 72};

  // DiskANN rust code has this test:
  // DiskANN/rust/diskann/src/algorithm/search/search.rs,
  //   search_for_point_initial_call():
  //   query = 0, medoid = 72?
  // TEST_DATA_FILE = tests/data/siftsmall_learn_256pts.fbin
  // assert_eq!(visited_nodes.len(), 1);
  // assert_eq!(scratch.best_candidates.size(), 1);
  // assert_eq!(scratch.best_candidates[0].id, 72);
  // assert_eq!(scratch.best_candidates[0].distance, 125678.0_f32);
  // assert!(scratch.best_candidates[0].visited);
  // Load 256 points
  // search_list_size == 50, max degree == 4, alpha == 1.2
  // num_nodes, Lbuild, Rmax_degree

  // The function search_for_point_initial_call() just seems to compute
  // the distance from the medoid to the query point.  We don't really
  // have that functionality for anything.  So here we just test that
  // the distance between 0 and 72 is 125678.0

  auto x =
      read_diskann_data(diskann_test_data_file);  // siftsmall_learn_256pts.fbin
  int med = 72;
  int query = 0;
  CHECK(l2_distance(x[med], x[query]) == 125678);

  // We might want to also do a search and verify that the path to 0 from 72
  // is less than 125678
}

/*
 * The data in this test were cribbed from DiskANN's tests.
 * See DiskANN/rust//diskann/src/algorithm/search/search.rs
 * function search_for_point_works_with_edges()
 */
TEST_CASE("small greedy search", "[vamana]") {
  const bool debug = false;
  const bool noisy = false;

  set_noisy(noisy);

  uint32_t npoints{0};
  uint32_t ndim{0};
  std::vector<size_t> vectors_of_interest{
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 72};

  // See copy_aligned_data_from_file() in DiskANN/rust/diskann/src/utils.rs
  // The name of the file is "tests/data/siftsmall_learn_256pts.fbin";
  std::ifstream binary_file(diskann_test_256bin, std::ios::binary);
  if (!binary_file.is_open()) {
    throw std::runtime_error("Could not open file " + diskann_test_256bin);
  }

  binary_file.read((char*)&npoints, 4);
  binary_file.read((char*)&ndim, 4);
  REQUIRE(npoints == 256);
  REQUIRE(ndim == 128);

  auto x = ColMajorMatrixWithIds<float>(ndim, npoints);
  std::iota(x.ids(), x.ids() + x.num_ids(), 0);

  binary_file.read((char*)x.data(), npoints * ndim * sizeof(float));
  binary_file.close();

  auto init_nbrs = std::vector<std::list<int>>{
      {12, 72, 5, 9},
      {2, 12, 10, 4},
      {1, 72, 9},
      {13, 6, 5, 11},
      {1, 3, 7, 9},
      {3, 0, 8, 11, 13},
      {3, 72, 7, 10, 13},
      {72, 4, 6},
      {72, 5, 9, 12},
      {8, 4, 0, 2},
      {72, 1, 9, 6},
      {3, 0, 5},
      {1, 0, 8, 9},
      {3, 72, 5, 6},
      {7, 2, 10, 8, 13},
  };
  auto init_nodes =
      std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 72};
  auto expected =
      std::vector<int>{2, 8, 72, 4, 7, 10, 1, 12, 9, 0, 6, 5, 3, 13, 11};
  auto expected_scores = std::vector<float>{
      120899,
      145538,
      146046,
      148462,
      148912,
      154570,
      159448,
      170698,
      177205,
      259996,
      371819,
      385240,
      413899,
      416386,
      449266};

  auto graph = detail::graph::adj_list<float, int>(num_vectors(x));
  for (size_t i = 0; i < size(init_nodes); ++i) {
    auto j = init_nodes[i];
    graph.out_edges(j).clear();
    for (auto&& dst : init_nbrs[i]) {
      auto score = l2_distance(x[j], x[dst]);
      graph.add_edge(j, dst, score);
      if (debug) {
        std::cout << "Adding edge " << j << " " << dst << " " << score
                  << std::endl;
      }
    }
  }
  for (size_t i = 0; i < size(init_nodes); ++i) {
    auto j = init_nodes[i];
    CHECK(size(graph.out_edges(j)) == size(init_nbrs[i]));
  }

  if (debug) {
    for (size_t i : vectors_of_interest) {
      std::cout << i << ": ";
      for (auto&& j : graph.out_edges(i)) {
        std::cout << std::get<1>(j) << " ";
      }
      std::cout << std::endl;
    }
  }

  auto yack = sum_of_squares_distance{}(x[72], x[14]);
  if (debug) {
    std::cout << "distance(x[72], x[14] = " << yack << std::endl;
  }
  // L = 50, R = 4
  size_t L = 45;
  auto query_id = 14;
  size_t k = 15;

  // A few different options that could be used for testing this
  // auto med = medoid(x);
  // int med = GENERATE(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 72);
  int med = 72;
  if (debug) {
    std::cout << "medoid is " << med << std::endl;
  }

  set_noisy(noisy);
  auto&& [top_k_scores, top_k, visited] =
      greedy_search(graph, x, med, x[query_id], k, L);

  CHECK(size(top_k) == k);
  CHECK(size(top_k_scores) == k);
  CHECK(size(visited) == size(expected));

  {
    auto&& [top_k_scores, top_k, visited] =
        greedy_search_O0(graph, x, med, x[query_id], k, L);
    CHECK(size(top_k) == k);
    CHECK(size(top_k_scores) == k);
    CHECK(size(visited) == size(expected));
    for (size_t i = 0; i < k; ++i) {
      CHECK(top_k[i] == expected[i]);
      CHECK(top_k_scores[i] == expected_scores[i]);
    }
  }
  set_noisy(noisy);
  {
    auto&& [top_k_scores, top_k, visited] =
        greedy_search_O1(graph, x, med, x[query_id], k, L);
    CHECK(size(top_k) == k);
    CHECK(size(top_k_scores) == k);
    CHECK(size(visited) == size(expected));
    for (size_t i = 0; i < k; ++i) {
      CHECK(top_k[i] == expected[i]);
      CHECK(top_k_scores[i] == expected_scores[i]);
    }
  }
  if (debug) {
    std::cout << "top_k_scores: " << std::endl;
    for (size_t i = 0; i < size(top_k); ++i) {
      std::cout << "( " << top_k[i] << ", " << top_k_scores[i] << " ), ";
    }
    std::cout << std::endl;

    std::cout << "visited: " << std::endl;
    for (auto&& v : visited) {
      std::cout << v << ", ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < size(expected); ++i) {
      std::cout << expected[i] << ", ";
    }
    std::cout << std::endl;
  }
  set_noisy(false);
  /*
   * The original Rust code has this test:
    set_neighbors(&index, 0, vec![12, 72, 5, 9]);
    set_neighbors(&index, 1, vec![2, 12, 10, 4]);
    set_neighbors(&index, 2, vec![1, 72, 9]);
    set_neighbors(&index, 3, vec![13, 6, 5, 11]);
    set_neighbors(&index, 4, vec![1, 3, 7, 9]);
    set_neighbors(&index, 5, vec![3, 0, 8, 11, 13]);
    set_neighbors(&index, 6, vec![3, 72, 7, 10, 13]);
    set_neighbors(&index, 7, vec![72, 4, 6]);
    set_neighbors(&index, 8, vec![72, 5, 9, 12]);
    set_neighbors(&index, 9, vec![8, 4, 0, 2]);
    set_neighbors(&index, 10, vec![72, 1, 9, 6]);
    set_neighbors(&index, 11, vec![3, 0, 5]);
    set_neighbors(&index, 12, vec![1, 0, 8, 9]);
    set_neighbors(&index, 13, vec![3, 72, 5, 6]);
    set_neighbors(&index, 72, vec![7, 2, 10, 8, 13]);

    let mut scratch = InMemQueryScratch::new(
                          index.configuration.index_write_parameter.search_list_size,
                          &index.configuration.index_write_parameter,
                          false,
                          )
                          .unwrap();
    let visited_nodes = index.search_for_point(&query, &mut scratch).unwrap();
    assert_eq!(visited_nodes.len(), 15);
    assert_eq!(scratch.best_candidates.size(), 15);
    assert_eq!(scratch.best_candidates[0].id, 2);
    assert_eq!(scratch.best_candidates[0].distance, 120899.0_f32);
    assert_eq!(scratch.best_candidates[1].id, 8);
    assert_eq!(scratch.best_candidates[1].distance, 145538.0_f32);
    assert_eq!(scratch.best_candidates[2].id, 72);
    assert_eq!(scratch.best_candidates[2].distance, 146046.0_f32);
    assert_eq!(scratch.best_candidates[3].id, 4);
    assert_eq!(scratch.best_candidates[3].distance, 148462.0_f32);
    assert_eq!(scratch.best_candidates[4].id, 7);
    assert_eq!(scratch.best_candidates[4].distance, 148912.0_f32);
    assert_eq!(scratch.best_candidates[5].id, 10);
    assert_eq!(scratch.best_candidates[5].distance, 154570.0_f32);
    assert_eq!(scratch.best_candidates[6].id, 1);
    assert_eq!(scratch.best_candidates[6].distance, 159448.0_f32);
    assert_eq!(scratch.best_candidates[7].id, 12);
    assert_eq!(scratch.best_candidates[7].distance, 170698.0_f32);
    assert_eq!(scratch.best_candidates[8].id, 9);
    assert_eq!(scratch.best_candidates[8].distance, 171405.0_f32);
    assert_eq!(scratch.best_candidates[9].id, 0);
    assert_eq!(scratch.best_candidates[9].distance, 259996.0_f32);
    assert_eq!(scratch.best_candidates[10].id, 6);
    assert_eq!(scratch.best_candidates[10].distance, 371819.0_f32);
    assert_eq!(scratch.best_candidates[11].id, 5);
    assert_eq!(scratch.best_candidates[11].distance, 385240.0_f32);
    assert_eq!(scratch.best_candidates[12].id, 3);
    assert_eq!(scratch.best_candidates[12].distance, 413899.0_f32);
    assert_eq!(scratch.best_candidates[13].id, 13);
    assert_eq!(scratch.best_candidates[13].distance, 416386.0_f32);
    assert_eq!(scratch.best_candidates[14].id, 11);
    assert_eq!(scratch.best_candidates[14].distance, 449266.0_f32);
  */
}

TEST_CASE("greedy grid search", "[vamana]") {
  const bool debug = false;
  const bool noisy = false;

  set_noisy(noisy);

  // using feature_type = uint8_t;
  using id_type = uint32_t;
  using score_type = float;

  size_t M = 5;
  size_t N = 7;

  // auto one_two = GENERATE(1);
  auto one_two = GENERATE(2);
  // auto one_two = GENERATE(1, 2);
  auto&& [vecs, edges] = ([one_two, M, N]() {
    if (one_two == 1) {
      return gen_uni_grid(M, N);
    } else {
      return gen_bi_grid(M, N);
    }
  })();

  auto expected_size = ((M - 1) * N + M * (N - 1)) * one_two;
  CHECK(num_vectors(vecs) == M * N);
  CHECK(edges.size() == expected_size);

  detail::graph::adj_list<score_type, id_type> A(35);
  for (auto&& [src, dst] : edges) {
    CHECK(src < A.num_vertices());
    CHECK(dst < A.num_vertices());
    A.add_edge(src, dst, l2_distance(vecs[src], vecs[dst]));
  }

  // (2, 3): 17 -> {10, 16, 17, 18, 24}
  // (3, 4): 25 -> {18, 24, 25, 26, 32}
  // (3, 6): 27 -> {20, 26, 27, 34}
  // (4, 5): 33 -> {26, 32, 33, 34}
  // (4, 6): 33 -> {27, 33, 34}

  using expt_type = std::tuple<
      std::vector<score_type>,
      id_type,
      std::vector<score_type>,
      id_type,
      id_type,
      std::vector<id_type>>;
  std::vector<expt_type> expts{
      {{0, 0}, 0, {2, 3}, 17, 7, {10, 16, 17, 18, 24}},
      {{0, 0}, 0, {3, 4}, 25, 9, {18, 24, 25, 26, 32}},
      {{0, 0}, 0, {3, 6}, 27, 11, {20, 26, 27, 34}},  // 33
      {{0, 0}, 0, {4, 5}, 33, 11, {26, 32, 33, 34}},  // 25
      {{0, 0}, 0, {4, 6}, 34, 11, {27, 33, 34}},      // 26 32
  };

  size_t count = 0;
  for (
      auto&& [source_vec, source_coord, query_vec, query_coord, path_length, expected] :
      expts) {
    size_t k = 5;
    size_t L = 5;  // --> L must be >= k
    size_t expected_length =
        query_vec[0] - source_vec[0] + query_vec[1] - source_vec[1] + 2;

    auto&& [top_k_scores, top_k, V] =
        greedy_search(A, vecs, source_coord, query_vec, size(expected), L);
    CHECK(size(top_k) == size(expected));

    std::sort(begin(top_k), begin(top_k) + size(expected));
    CHECK(std::equal(begin(expected), end(expected), begin(top_k)));

    if (debug) {
      std::cout << ":::: " << count << " :::: \n";
      for (auto&& n : top_k) {
        std::cout << n << " ";
      }
      std::cout << std::endl;
      for (auto&& e : expected) {
        std::cout << e << " ";
      }
      std::cout << std::endl;
      for (auto&& v : V) {
        std::cout << "(" << vecs[v][0] << ", " << vecs[v][1] << ") ";
      }
      std::cout << std::endl;
    }
  }
}

TEST_CASE("greedy search hypercube", "[vamana]") {
  const bool debug = false;
  const bool noisy = false;

  set_noisy(noisy);

  size_t k_near = 5;
  size_t k_far = 5;
  size_t L = 5;

  auto nn_hypercube = build_hypercube(k_near, k_far);

  auto g = detail::graph::init_random_nn_graph<float, uint32_t>(
      nn_hypercube, k_near);

  for (float kk : {-1, 1}) {
    auto query = Vector<float>{kk * 1.05f, kk * 0.95f, 1.09};
    auto [top_k_scores, top_k, V] =
        greedy_search(g, nn_hypercube, 2, query, k_near, L);

    if (debug) {
      std::cout << "Nearest neighbors:" << std::endl;
      for (auto&& n : top_k) {
        std::cout << n << " (" << nn_hypercube(0, n) << ", "
                  << nn_hypercube(1, n) << ", " << nn_hypercube(2, n) << "), "
                  << l2_distance(nn_hypercube[n], query) << std::endl;
      }
      std::cout << "-----\ntop_k\n";
    }

    auto query_mat = ColMajorMatrix<float>(3, 1);
    for (size_t i = 0; i < 3; ++i) {
      query_mat(i, 0) = query[i];
    }

    {
      auto&& [top_k_scores, top_k] =
          detail::flat::qv_query_heap(nn_hypercube, query_mat, k_near, 1);

      if (debug) {
        for (size_t i = 0; i < k_near; ++i) {
          std::cout << top_k(i, 0) << " (" << nn_hypercube(0, top_k(i, 0))
                    << ", " << nn_hypercube(1, top_k(i, 0)) << ", "
                    << nn_hypercube(2, top_k(i, 0)) << "), "
                    << top_k_scores(i, 0) << std::endl;
        }
        for (auto&& v : V) {
          std::cout << v << ", ";
        }
        std::cout << std::endl;
      }
    }
  }
}

TEST_CASE("greedy search with nn descent", "[vamana]") {
  const bool debug = false;
  const bool noisy = false;

  set_noisy(noisy);

  size_t k_near = 5;
  size_t k_far = 5;
  size_t L = 7;

  auto nn_hypercube = build_hypercube(k_near, k_far);

  std::vector<std::tuple<float, size_t>> top_k(k_near);
  auto g = detail::graph::init_random_nn_graph<float, uint32_t>(
      nn_hypercube, k_near);

  auto valid = validate_graph(g, nn_hypercube);
  CHECK(valid.size() == 0);

  for (int kk : {-1, 1}) {
    auto query = Vector<float>{kk * 1.05f, kk * 0.95f, 1.09f};

    auto&& [top_k_scores, top_k, V] =
        greedy_search(g, nn_hypercube, 0UL, query, k_near, L);

    if (debug) {
      std::cout << "size(V): " << size(V) << std::endl;
    }

    for (size_t i = 0; i < 4; ++i) {
      auto num_updates = nn_descent_1_step_all(g, nn_hypercube);
      if (debug) {
        std::cout << "num_updates: " << num_updates << std::endl;
      }
      if (num_updates == 0) {
        break;
      }
      auto&& [top_k_scores, top_k, V] =
          greedy_search(g, nn_hypercube, 0UL, query, k_near, L);

      if (debug) {
        std::cout << "size(V): " << size(V) << std::endl;
      }
    }

    if (debug) {
      for (auto&& v : V) {
        std::cout << v << ", ";
      }
      std::cout << std::endl;

      std::cout << "Nearest neighbors:" << std::endl;
      for (auto&& n : top_k) {
        std::cout << n << " (" << nn_hypercube(0, n) << ", "
                  << nn_hypercube(1, n) << ", " << nn_hypercube(2, n) << "), "
                  << sum_of_squares_distance{}(nn_hypercube[n], query)
                  << std::endl;
      }
      std::cout << "-----\n";
    }
  }
}

TEST_CASE("diskann fbin", "[vamana]") {
  const bool debug = false;
  const bool noisy = false;

  set_noisy(noisy);

  size_t k_nn = 5;
  size_t L = 5;

  // should be dim = 128, num = 256
  // npoints, ndims
  uint32_t npoints{0};
  uint32_t ndim{0};

  std::ifstream binary_file(diskann_test_256bin, std::ios::binary);
  if (!binary_file.is_open()) {
    throw std::runtime_error("Could not open file " + diskann_test_256bin);
  }

  binary_file.read((char*)&npoints, 4);
  binary_file.read((char*)&ndim, 4);
  REQUIRE(npoints == 256);
  REQUIRE(ndim == 128);

  auto x = ColMajorMatrixWithIds<float>(ndim, npoints);
  std::iota(x.ids(), x.ids() + x.num_ids(), 0);

  binary_file.read((char*)x.data(), npoints * ndim);
  binary_file.close();

  SECTION("vary starts") {
    auto start = GENERATE(0, 17, 127, 128, 129, 254, 255);

    auto g = detail::graph::init_random_nn_graph<float, uint32_t>(x, k_nn);

    auto&& [top_k_scores, top_k, V] =
        greedy_search(g, x, start, x[start], k_nn, L);
    std::sort(begin(top_k), end(top_k));

    // There seems to be some ambiguity in the DiskANN code about whether
    // the start point should be included in the top_k list.  Check here
    // that it is not.
    CHECK(std::find(begin(top_k), end(top_k), start) != end(top_k));
    CHECK(std::find(begin(V), end(V), start) != end(V));
  }
}

TEST_CASE("fmnist", "[vamana]") {
  const bool debug = false;
  const bool noisy = false;

  set_noisy(noisy);

  using feature_type = float;
  using score_type = float;
  using id_type = uint32_t;

  size_t nthreads = 1;
  size_t L = 7;
  size_t k_nn = L;
  size_t num_queries = 10;
  size_t N = 5000;

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);

  auto ids_uri = write_ids_to_uri<id_type>(ctx, vfs, N);
  auto db = tdbColMajorMatrixWithIds<test_feature_type, id_type>(
      ctx, fmnist_inputs_uri, ids_uri, N);
  db.load();
  auto g = detail::graph::init_random_nn_graph<score_type, id_type>(db, L);

  auto valid = validate_graph(g, db);
  CHECK(valid.size() == 0);

  auto query = db[599];

  auto query_mat = ColMajorMatrix<feature_type>(size(query), 1);
  for (size_t i = 0; i < size(query); ++i) {
    query_mat(i, 0) = query[i];
  }

  auto&& [top_scores, qv_top_k] =
      detail::flat::qv_query_heap(db, query_mat, k_nn, 1);
  std::sort(begin(qv_top_k[0]), end(qv_top_k[0]));

  if (debug) {
    std::cout << "Neighbors: ";
    for (size_t i = 0; i < k_nn; ++i) {
      std::cout << qv_top_k(i, 0) << " ";
    }
    std::cout << "\nDistances: ";
    for (size_t i = 0; i < k_nn; ++i) {
      std::cout << sum_of_squares_distance{}(db[qv_top_k(i, 0)], query) << " ";
    }
    std::cout << "\n-----\n";
  }
  auto&& [top_k_scores, top_k, V] = greedy_search(g, db, 0UL, query, k_nn, L);

  if (debug) {
    std::cout << "size(V): " << size(V) << std::endl;
  }
  for (size_t i = 0; i < 7; ++i) {
    auto num_updates = nn_descent_1_step_all(g, db);

    if (debug) {
      std::cout << "num_updates: " << num_updates << std::endl;
    }

    auto&& [top_k_scores, top_k, V] = greedy_search(g, db, 0UL, query, k_nn, L);

    auto valid = validate_graph(g, db);
    CHECK(valid.size() == 0);

    if (debug) {
      std::cout << "size(V): " << size(V) << std::endl;
    }

    auto top_n = ColMajorMatrix<size_t>(k_nn, 1);
    for (size_t i = 0; i < k_nn; ++i) {
      top_n(i, 0) = top_k[i];
    }

    auto num_intersected = count_intersections(top_n, qv_top_k, k_nn);

    if (debug) {
      std::cout << "num_intersected: " << num_intersected << " / " << k_nn
                << " = "
                << ((double)num_intersected) /
                       ((double)query_mat.num_cols() * k_nn)
                << std::endl;

      std::cout << "Greedy nearest neighbors: ";
      for (auto&& n : top_k) {
        std::cout << n << " ";
      }
      std::cout << "\nGreedy distances: ";
      for (auto&& n : top_k) {
        std::cout << sum_of_squares_distance{}(db[n], query) << " ";
      }
      std::cout << "\n-----\n";
    }

    if (num_updates == 0) {
      break;
    }
  }

  auto&& [s, t] = nn_descent_1_query(g, db, query_mat, k_nn, k_nn + 5, 3);

  auto num_intersected = count_intersections(t, qv_top_k, k_nn);

  if (debug) {
    std::cout << "num_intersected: " << num_intersected << " / " << k_nn
              << " = "
              << ((double)num_intersected) /
                     ((double)query_mat.num_cols() * k_nn)
              << std::endl;

    std::cout << "NN-descent nearest neighbors: ";
    for (size_t i = 0; i < k_nn; ++i) {
      std::cout << t(i, 0) << " ";
    }

    std::cout << "\nNN-descent returned distances: ";
    for (size_t i = 0; i < k_nn; ++i) {
      std::cout << s(i, 0) << " ";
    }
    std::cout << "\nNN-descent computed distances: ";
    for (size_t i = 0; i < k_nn; ++i) {
      std::cout << sum_of_squares_distance{}(db[t(i, 0)], query) << " ";
    }
    std::cout << std::endl;
  }
}

TEST_CASE("fmnist compare greedy search", "[vamana]") {
  const bool debug = false;
  const bool noisy = false;

  set_noisy(noisy);

  using feature_type = float;
  using score_type = float;
  using id_type = uint32_t;

  size_t nthreads = 1;
  size_t L = GENERATE(7, 15, 50);
  size_t k_nn = L;
  size_t num_queries = 10;
  size_t N = 5000;

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);

  auto ids_uri = write_ids_to_uri<siftsmall_ids_type>(ctx, vfs, N);
  auto db = tdbColMajorMatrixWithIds<test_feature_type>(
      ctx, fmnist_inputs_uri, ids_uri, N);
  db.load();
  auto g = detail::graph::init_random_nn_graph<score_type, id_type>(db, L);

  auto valid = validate_graph(g, db);
  CHECK(valid.size() == 0);

  auto query = db[599];

  auto query_mat = ColMajorMatrix<feature_type>(size(query), 1);
  for (size_t i = 0; i < size(query); ++i) {
    query_mat(i, 0) = query[i];
  }

  auto&& [top_scores, qv_top_k] =
      detail::flat::qv_query_heap(db, query_mat, k_nn, 1);
  std::sort(begin(qv_top_k[0]), end(qv_top_k[0]));

  auto&& [top_k_scores_O0, top_k_O0, V_O0] =
      greedy_search_O0(g, db, 0UL, query, k_nn, L);
  auto&& [top_k_scores_O1, top_k_O1, V_O1] =
      greedy_search_O1(g, db, 0UL, query, k_nn, L);

  CHECK(top_k_scores_O0 == top_k_scores_O1);
  CHECK(top_k_O0 == top_k_O1);
  CHECK(V_O0 == V_O1);

  auto top_n_O0 = ColMajorMatrix<size_t>(k_nn, 1);
  auto top_n_O1 = ColMajorMatrix<size_t>(k_nn, 1);
  for (size_t i = 0; i < k_nn; ++i) {
    top_n_O0(i, 0) = top_k_O0[i];
    top_n_O1(i, 0) = top_k_O1[i];
  }
  auto num_intersected_O0 = count_intersections(top_n_O0, qv_top_k, k_nn);
  auto num_intersected_O1 = count_intersections(top_n_O1, qv_top_k, k_nn);
  CHECK(num_intersected_O0 == num_intersected_O1);
  if (debug) {
    std::cout << "num intersected_O0: " << num_intersected_O0 << " / " << L
              << std::endl;
    std::cout << "num intersected_O1: " << num_intersected_O1 << " / " << L
              << std::endl;
  }
}

TEST_CASE("robust prune hypercube", "[vamana]") {
  const bool debug = false;
  const bool noisy = false;

  set_noisy(noisy);
  size_t k_near = 5;
  size_t k_far = 5;
  size_t L = 7;
  size_t R = 7;

  float alpha = 1.0;

  auto nn_hypercube = build_hypercube(k_near, k_far);
  auto start = medoid(nn_hypercube);

  if (debug) {
    for (auto&& s : nn_hypercube[start]) {
      std::cout << s << ", ";
    }
    std::cout << std::endl;
  }

  std::vector<std::tuple<float, size_t>> top_k(k_near);
  auto g =
      detail::graph::init_random_nn_graph<float, uint32_t>(nn_hypercube, R);

  auto valid = validate_graph(g, nn_hypercube);
  CHECK(valid.size() == 0);

  auto query = Vector<float>{1.05, 0.95, 1.09};

  SECTION("One node") {
    auto p = 8UL;

    if (debug) {
      for (auto&& [s, t] : g.out_edges(8UL)) {
        std::cout << " ( " << t << ", " << s << " ) ";
      }
      std::cout << std::endl;
    }

    auto&& [top_k_scores, top_k, V] =
        greedy_search(g, nn_hypercube, start, nn_hypercube[p], k_near, L);
    robust_prune(g, nn_hypercube, p, V, alpha, R);

    if (debug) {
      for (auto&& [s, t] : g.out_edges(8UL)) {
        std::cout << " ( " << t << ", " << s << " ) ";
      }
      std::cout << std::endl;
    }
  }

  SECTION("One pass") {
    for (size_t p = 0; p < nn_hypercube.num_cols(); ++p) {
      if (debug) {
        for (auto&& [s, t] : g.out_edges(p)) {
          std::cout << " ( " << t << ", " << s << " ) ";
        }
        std::cout << std::endl;
      }

      auto&& [top_k_scores, top_k, V] =
          greedy_search(g, nn_hypercube, start, nn_hypercube[p], k_near, L);

      auto valid = validate_graph(g, nn_hypercube);
      CHECK(valid.size() == 0);

      robust_prune(g, nn_hypercube, p, V, alpha, R);

      auto valid2 = validate_graph(g, nn_hypercube);
      CHECK(valid2.size() == 0);

      if (debug) {
        for (auto&& [s, t] : g.out_edges(p)) {
          std::cout << " ( " << t << ", " << s << " ) ";
        }
        std::cout << std::endl;
      }
    }

    auto&& [top_k_scores, top_k, V] =
        greedy_search(g, nn_hypercube, start, query, k_near, L);

    auto valid = validate_graph(g, nn_hypercube);
    CHECK(valid.size() == 0);

    if (debug) {
      std::cout << "V.size: " << size(V) << std::endl;
      for (auto&& v : V) {
        std::cout << v << ", ";
      }
      std::cout << std::endl;
      for (auto&& n : top_k) {
        std::cout << n << " (" << nn_hypercube(0, n) << ", "
                  << nn_hypercube(1, n) << ", " << nn_hypercube(2, n) << "), "
                  << sum_of_squares_distance{}(nn_hypercube[n], query)
                  << std::endl;
      }
    }
  }
}

TEST_CASE("robust prune fmnist", "[vamana]") {
  const bool debug = false;
  const bool noisy = false;

  set_noisy(noisy);

  size_t nthreads = 1;
  size_t k_nn = 5;
  size_t L = 7;
  size_t R = 7;
  size_t num_queries = 10;
  size_t N = 500;
  float alpha = 1.0;

  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);

  auto ids_uri = write_ids_to_uri<siftsmall_ids_type>(ctx, vfs, N);
  auto db = tdbColMajorMatrixWithIds<test_feature_type, siftsmall_ids_type>(
      ctx, fmnist_inputs_uri, ids_uri, N);
  db.load();
  auto g = detail::graph::
      init_random_nn_graph<siftsmall_feature_type, siftsmall_ids_type>(db, L);

  auto valid = validate_graph(g, db);
  REQUIRE(valid.size() == 0);

  auto x = std::accumulate(begin(db[0]), end(db[0]), 0.0F);

  auto query = db[N / 2 + 3];
  std::vector<float> query_vec(begin(query), end(query));

  auto query_mat = ColMajorMatrix<float>(size(query), 1);
  for (size_t i = 0; i < size(query); ++i) {
    query_mat(i, 0) = query[i];
  }

  auto valid6 = validate_graph(g, db);
  REQUIRE(valid6.size() == 0);

  auto qv_timer = log_timer{"qv", debug};
  auto&& [top_scores, qv_top_k] =
      detail::flat::qv_query_heap(db, query_mat, k_nn, 1);
  std::sort(begin(qv_top_k[0]), end(qv_top_k[0]));
  qv_timer.stop();

  auto valid2 = validate_graph(g, db);
  REQUIRE(valid2.size() == 0);

  auto start = medoid(db);

  for (float alpha : {1.0, 1.25}) {
    if (debug) {
      std::cout << ":::: alpha: " << alpha << std::endl;
    }
    for (size_t p = 0; p < db.num_cols(); ++p) {
      auto valid3 = validate_graph(g, db);
      REQUIRE(valid3.size() == 0);

      auto&& [top_k_scores, top_k, V] =
          greedy_search(g, db, start, db[p], k_nn, L);

      auto valid4 = validate_graph(g, db);
      CHECK(valid4.size() == 0);

      robust_prune(g, db, p, V, alpha, R);

      auto valid5 = validate_graph(g, db);
      CHECK(valid5.size() == 0);
    }
  }

  auto greedy_timer = log_timer{"greedy", debug};
  auto&& [top_k_scores, top_k, V] = greedy_search(g, db, start, query, k_nn, L);
  greedy_timer.stop();

  if (debug) {
    std::cout << "V.size: " << size(V) << std::endl;

    // for (auto&& v : V) {
    //   std::cout << v << ", ";
    // }
  }

  auto top_n = ColMajorMatrix<size_t>(k_nn, 1);
  for (size_t i = 0; i < k_nn; ++i) {
    top_n(i, 0) = top_k[i];
  }

  auto num_intersected = count_intersections(top_n, qv_top_k, k_nn);

  if (debug) {
    std::cout << "num_intersected: " << num_intersected << " / " << k_nn
              << " = "
              << ((double)num_intersected) /
                     ((double)query_mat.num_cols() * k_nn)
              << std::endl;

    std::cout << "Greedy nearest neighbors: ";
    for (auto&& n : top_k) {
      std::cout << n << " ";
    }
    std::cout << "\nGreedy distances: ";
    for (auto&& n : top_k) {
      std::cout << sum_of_squares_distance{}(db[n], query) << " ";
    }
    std::cout << "\n-----\n";
  }

#if 0
  std::cout << std::endl;
  for (auto&& n : top_k[0]) {
    std::cout << n << " (" << db(0, n) << ", " << db(1, n) << ", " << db(2, n)
              << "), " << sum_of_squares_distance{}(db[n], query) << std::endl;
  }
  auto num_intersected = count_intersections(top_k, top_k, k_nn);
  std::cout << "num_intersected: " << num_intersected << " / " << k_nn << " = "
            << ((double)num_intersected) / ((double)query_mat.num_cols() * k_nn)
            << std::endl;
#endif
}

TEST_CASE("vamana_index vector diskann_test_256bin", "[vamana]") {
  const bool debug = false;
  const bool noisy = false;

  set_noisy(noisy);

  // should be dim = 128, num = 256
  // npoints, ndims
  uint32_t npoints{0};
  uint32_t ndim{0};

  std::ifstream binary_file(diskann_test_256bin, std::ios::binary);
  if (!binary_file.is_open()) {
    throw std::runtime_error("Could not open file " + diskann_test_256bin);
  }

  binary_file.read((char*)&npoints, 4);
  binary_file.read((char*)&ndim, 4);
  REQUIRE(npoints == 256);
  REQUIRE(ndim == 128);

  auto x = ColMajorMatrix<float>(ndim, npoints);

  binary_file.read((char*)x.data(), npoints * ndim);
  binary_file.close();

  size_t l_build = 50;
  size_t r_max_degree = 4;
  auto index = vamana_index<siftsmall_feature_type, siftsmall_ids_type>(
      num_vectors(x), l_build, r_max_degree);

  auto x0 = std::vector<float>(ndim);
  std::copy(x.data(), x.data() + ndim, begin(x0));

  std::vector<siftsmall_ids_type> ids(num_vectors(x));
  std::iota(begin(ids), end(ids), 0);

  index.train(x, ids);

  // x, 5, 7, 7, 1.0, 1, 1);
  // 0
  // 14
  auto&& [s0, v0] = index.query(x0, 5);
  CHECK(v0[0] == 0);
}

TEST_CASE("vamana by hand random index", "[vamana]") {
  const bool debug = false;
  const bool noisy = false;

  set_noisy(noisy);

  size_t num_nodes = 20;

  float alpha_0 = 1.0;
  float alpha_1 = 1.2;

  size_t l_build = 2;
  size_t r_max_degree = 2;

  auto training_set_ = random_geometric_2D(num_nodes);
  dump_coordinates("coords.txt", training_set_);

  auto g = ::detail::graph::init_random_adj_list<float, uint32_t>(
      training_set_, r_max_degree);

  auto dimensions_ = ::dimensions(training_set_);
  auto num_vectors_ = ::num_vectors(training_set_);
  auto graph_ = ::detail::graph::
      init_random_nn_graph<siftsmall_feature_type, siftsmall_ids_type>(
          training_set_, r_max_degree);

  auto medioid_ = medoid(training_set_);

  if (debug) {
    std::cout << "medoid: " << medioid_ << std::endl;
  }

  size_t counter{0};
  for (float alpha : {alpha_0, alpha_1}) {
    for (size_t p = 0; p < num_vectors_; ++p) {
      if (debug) {
        dump_edgelist("edges_" + std::to_string(counter++) + ".txt", graph_);
      }

      auto&& [top_k_scores, top_k, visited] = greedy_search(
          graph_, training_set_, medioid_, training_set_[p], 1, l_build);

      if (debug) {
        std::cout << ":::: Post search prune" << std::endl;
      }
      robust_prune(graph_, training_set_, p, visited, alpha, r_max_degree);

      for (auto&& [i, j] : graph_.out_edges(p)) {
        if (debug) {
          std::cout << ":::: Checking neighbor " << j << std::endl;
        }

        // @todo Do this without copying -- prune should take vector of tuples
        // and p (it copies anyway)
        auto tmp = std::vector<size_t>(graph_.out_degree(j) + 1);
        tmp.push_back(p);
        for (auto&& [_, k] : graph_.out_edges(j)) {
          if (k != p) {
            tmp.push_back(k);
          }
        }

        if (size(tmp) > r_max_degree) {
          if (debug) {
            std::cout << ":::: Pruning neighbor " << j << std::endl;
          }
          robust_prune(graph_, training_set_, j, tmp, alpha, r_max_degree);
        } else {
          graph_.add_edge(
              j,
              p,
              sum_of_squares_distance()(training_set_[p], training_set_[j]));
        }
      }
    }
  }
}

/**
 * This test recapitulates the 200 node 2D graph in the DiskANN paper
 */
TEST_CASE("vamana_index geometric 2D graph", "[vamana]") {
  const bool debug = false;
  const bool noisy = false;

  set_noisy(noisy);

  size_t num_nodes = 200;

  float alpha_0 = 1.0;
  float alpha_1 = 1.2;

  size_t l_build = 15;
  size_t r_max_degree = 15;

  size_t k_nn = 5;

  auto training_set = random_geometric_2D(num_nodes);

  auto idx = vamana_index<siftsmall_feature_type, siftsmall_ids_type>(
      num_vectors(training_set), l_build, r_max_degree);
  std::vector<siftsmall_ids_type> ids(num_nodes);
  auto ids_start = 10;
  std::iota(begin(ids), end(ids), ids_start);
  idx.train(training_set, ids);

  auto query = training_set[17];
  auto&& [scores, top_k] = idx.query(query, k_nn);
  CHECK(top_k[0] == 17 + ids_start);

  auto query_mat = ColMajorMatrix<float>(dimensions(training_set), 7);
  size_t counter{0};
  for (size_t i : {17, 19, 23, 37, 49, 50, 195}) {
    std::copy(
        training_set[i].data(),
        training_set[i].data() + dimensions(training_set),
        query_mat[counter++].data());
  }

  auto&& [qv_scores, qv_top_k] =
      detail::flat::qv_query_heap(training_set, query_mat, k_nn, 4);
  auto&& [mat_scores, mat_top_k] = idx.query(query_mat, k_nn);
  size_t total_intersected = count_intersections(mat_top_k, qv_top_k, k_nn);

  if (debug) {
    std::cout << total_intersected << " / " << k_nn * num_vectors(query_mat)
              << " = "
              << ((double)total_intersected) /
                     ((double)k_nn * num_vectors(query_mat))
              << std::endl;
  }
}

TEST_CASE("vamana_index siftsmall", "[vamana]") {
  const bool debug = false;
  const bool noisy = false;

  set_noisy(noisy);

  size_t num_nodes = 10000;
  size_t num_queries = 200;

  float alpha_0 = 1.0;
  float alpha_1 = 1.2;

  size_t l_build = 15;
  size_t r_max_degree = 12;

  size_t k_nn = 10;

  tiledb::Context ctx;
  auto training_set = tdbColMajorMatrix<siftsmall_feature_type>(
      ctx, siftsmall_inputs_uri, num_nodes);
  training_set.load();
  std::vector<siftsmall_ids_type> ids(num_nodes);
  std::iota(begin(ids), end(ids), 0);

  auto queries = tdbColMajorMatrix<siftsmall_feature_type>(
      ctx, siftsmall_query_uri, num_queries);
  queries.load();

  auto idx = vamana_index<siftsmall_feature_type, siftsmall_ids_type>(
      num_vectors(training_set), l_build, r_max_degree);
  idx.train(training_set, ids);

  auto&& [mat_scores, mat_top_k] = idx.query(queries, k_nn);

  auto gk = tdbColMajorMatrix<test_groundtruth_type>(ctx, sift_groundtruth_uri);
  gk.load();
  size_t total_intersected = count_intersections(mat_top_k, gk, k_nn);

  auto recall =
      ((double)total_intersected) / ((double)k_nn * num_vectors(queries));
  CHECK(recall > 0.80);  // @todo -- had been 0.95?

  if (debug) {
    std::cout << total_intersected << " / " << k_nn * num_vectors(queries)
              << " = "
              << ((double)total_intersected) /
                     ((double)k_nn * num_vectors(queries))
              << std::endl;
  }
}

TEST_CASE("vamana_index write and read", "[vamana]") {
  const bool debug = false;
  const bool noisy = false;

  set_noisy(noisy);

  size_t l_build{37};
  size_t r_max_degree{41};
  size_t k_nn{10};

  tiledb::Context ctx;
  std::string vamana_index_uri =
      (std::filesystem::temp_directory_path() / "tmp_vamana_index").string();
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(vamana_index_uri)) {
    vfs.remove_dir(vamana_index_uri);
  }
  auto training_set = tdbColMajorMatrix<float>(ctx, siftsmall_inputs_uri, 0);
  load(training_set);
  std::vector<siftsmall_ids_type> ids(num_vectors(training_set));
  std::iota(begin(ids), end(ids), 0);

  auto idx = vamana_index<siftsmall_feature_type, siftsmall_ids_type>(
      num_vectors(training_set), l_build, r_max_degree);
  idx.train(training_set, ids);
  uint64_t write_timestamp = 1000;
  idx.write_index(
      ctx, vamana_index_uri, TemporalPolicy(TimeTravel, write_timestamp));

  {
    // Load the index and check metadata.
    auto idx2 = vamana_index<siftsmall_feature_type, siftsmall_ids_type>(
        ctx, vamana_index_uri);

    CHECK(idx2.group().get_l_build() == l_build);
    CHECK(idx2.group().get_r_max_degree() == r_max_degree);
    CHECK(idx2.group().get_dimensions() == sift_dimensions);
    CHECK(idx2.group().get_temp_size() == 0);

    CHECK(idx2.group().get_all_num_edges().size() == 1);
    CHECK(idx2.group().get_all_base_sizes().size() == 1);
    CHECK(idx2.group().get_all_ingestion_timestamps().size() == 1);

    CHECK(idx2.group().get_all_num_edges()[0] > 0);
    CHECK(idx2.group().get_all_base_sizes()[0] == num_sift_vectors);
    CHECK(idx2.group().get_all_ingestion_timestamps()[0] == write_timestamp);

    // Can't compare groups because a write_index does not create a group
    // @todo Should it?
    // CHECK(idx.compare_group(idx2));

    CHECK(idx.compare_cached_metadata(idx2));
    CHECK(idx.compare_feature_vectors(idx2));
    CHECK(idx.compare_adj_scores(idx2));
    CHECK(idx.compare_adj_ids(idx2));
  }

  {
    // Clear history.
    vamana_index<siftsmall_feature_type, siftsmall_ids_type>::clear_history(
        ctx, vamana_index_uri, write_timestamp + 10);

    auto idx2 = vamana_index<siftsmall_feature_type, siftsmall_ids_type>(
        ctx, vamana_index_uri);

    CHECK(idx2.group().get_l_build() == l_build);
    CHECK(idx2.group().get_r_max_degree() == r_max_degree);
    CHECK(idx2.group().get_dimensions() == sift_dimensions);
    CHECK(idx2.group().get_temp_size() == 0);

    CHECK(idx2.group().get_all_num_edges().size() == 1);
    CHECK(idx2.group().get_all_base_sizes().size() == 1);
    CHECK(idx2.group().get_all_ingestion_timestamps().size() == 1);

    CHECK(idx2.group().get_all_num_edges()[0] == 0);
    CHECK(idx2.group().get_all_base_sizes()[0] == 0);
    CHECK(idx2.group().get_all_ingestion_timestamps()[0] == 0);
  }
}

TEST_CASE("query empty index", "[vamana]") {
  size_t l_build = 100;
  size_t r_max_degree = 100;
  size_t num_vectors = 0;
  size_t dimensions = 5;
  auto index = vamana_index<siftsmall_feature_type, siftsmall_ids_type>(
      num_vectors, l_build, r_max_degree);
  auto data =
      ColMajorMatrixWithIds<siftsmall_feature_type>(dimensions, num_vectors);
  index.train(data, data.raveled_ids());

  auto queries = std::vector<std::vector<siftsmall_feature_type>>{
      {1, 1, 1, 1, 1}, {2, 2, 2, 2, 2}};
  auto&& [scores, ids] = index.query(data, 1);
  CHECK(_cpo::num_vectors(scores) == 0);
  CHECK(_cpo::num_vectors(ids) == 0);
}
