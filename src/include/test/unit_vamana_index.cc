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

#include "array_defs.h"
#include "cpos.h"
#include "detail/flat/qv.h"
#include "detail/graph/adj_list.h"
#include "detail/graph/diskann.h"
#include "detail/graph/nn-descent.h"
#include "detail/graph/nn-graph.h"
#include "detail/linalg/matrix.h"
#include "detail/linalg/tdb_io.h"
#include "gen_graphs.h"
#include "graphs/tiny.h"
#include "index/vamana_index.h"
#include "query_common.h"
#include "utils/logging.h"
#include "utils/utils.h"

#include <filesystem>
namespace fs = std::filesystem;

#include <tiledb/tiledb>

bool global_debug = false;

TEST_CASE("vamana: test test", "[vamana]") {
  REQUIRE(true);
}

TEST_CASE("vamana: diskann", "[vamana]") {
  bool debug = false;
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
  CHECK(dimension(f) == 128);
  CHECK(f.data() != nullptr);
  CHECK(!std::equal(
      f.data(), f.data() + 256 * 128, std::vector<float>(128 * 256, 0).data()));

  CHECK(f.num_rows() == 128);
  CHECK(f.num_cols() == 256);

  CHECK(sum_of_squares(f[0], f[72]) == 125678);
  {
    auto n = num_vectors(f);
    CHECK(n != 0);
    CHECK(n == 256);
    CHECK(f[0].size() == 128);
    CHECK(dimension(f) == 128);

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
      sum += abs(centroid[i]);
      centroid[i] /= (float)num_vectors(f);
    }
    CHECK(sum > 0);

    std::vector<float> tmp{begin(centroid), end(centroid)};
    auto min_score = std::numeric_limits<float>::max();
    auto med = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < n; ++i) {
      auto score = sum_of_squares(f[i], centroid);
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
    std::cout << "f[0] - f[72] = " << sum_of_squares(f[0], f[72]) << std::endl;
  }

  CHECK(med == 72);

  //  if (debug) {
  //    tiledb::Context ctx;
  //    write_matrix(ctx, f, "/tmp/diskann_test_data_file.tdb");
  //  }
}

TEST_CASE("vamana: small256 build index", "[vamana]") {
  const bool debug = false;
  const bool noisy = false;

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
  CHECK(l2_distance{}(x[med], x[query]) == 125678);

  // We might want to also do a search and verify that the path to 0 from 72
  // is less than 125678
}

/*
 * The data in this test were cribbed from DiskANN's tests.
 * See DiskANN/rust//diskann/src/algorithm/search/search.rs
 * function search_for_point_works_with_edges()
 */
TEST_CASE("vamana: small greedy search", "[vamana]") {
  const bool debug = false;

  uint32_t npoints{0};
  uint32_t ndim{0};
  std::vector<size_t> vectors_of_interest{
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 72};

  // See copy_aligned_data_from_file() in DiskANN/rust/diskann/src/utils.rs
  // The name of the file is "tests/data/siftsmall_learn_256pts.fbin";
  std::ifstream binary_file(diskann_test_256bin, std::ios::binary);
  if (!binary_file.is_open()) {
    throw std::runtime_error(
        "Could not open file " + diskann_test_256bin.string());
  }

  binary_file.read((char*)&npoints, 4);
  binary_file.read((char*)&ndim, 4);
  REQUIRE(npoints == 256);
  REQUIRE(ndim == 128);

  auto x = ColMajorMatrix<float>(ndim, npoints);

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
      171405,
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
      auto score = sum_of_squares(x[j], x[dst]);
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

  auto&& [top_k_scores, top_k, visited] =
      greedy_search(graph, x, med, x[query_id], k, L);

  CHECK(size(top_k) == 15);
  CHECK(size(top_k_scores) == 15);
  CHECK(size(visited) == 15);

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

TEST_CASE("vamana: greedy grid search", "[vamana]") {
  const bool debug = false;

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
    A.add_edge(src, dst, sum_of_squares_distance{}(vecs[src], vecs[dst]));
  }

  // (2, 3): 17 -> {10, 16, 17, 18, 24}
  // (3, 4): 25 -> {18, 24, 25, 26, 32}
  // (3, 6): 27 -> {20, 26, 27, 34}
  // (4, 5): 33 -> {26, 32, 33, 34}
  // (4, 6): 33 -> {27, 33, 34}

  using expt_type = std::tuple<
      std::vector<id_type>,
      id_type,
      std::vector<id_type>,
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

TEST_CASE("vamana: greedy search hypercube", "[vamana]") {
  const bool debug = false;

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
                  << sum_of_squares_distance{}(nn_hypercube[n], query)
                  << std::endl;
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

TEST_CASE("vamana: greedy search with nn descent", "[vamana]") {
  const bool debug = false;

  size_t k_near = 5;
  size_t k_far = 5;
  size_t L = 7;

  auto nn_hypercube = build_hypercube(k_near, k_far);

  std::vector<std::tuple<float, size_t>> top_k(k_near);
  auto g = detail::graph::init_random_nn_graph<float, uint32_t>(
      nn_hypercube, k_near);

  auto valid = validate_graph(g, nn_hypercube);
  CHECK(valid.size() == 0);

  for (size_t kk : {-1, 1}) {
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

TEST_CASE("vamana: diskann fbin", "[vamana]") {
  size_t k_nn = 5;
  size_t L = 5;

  // should be dim = 128, num = 256
  // npoints, ndims
  uint32_t npoints{0};
  uint32_t ndim{0};

  std::ifstream binary_file(diskann_test_256bin, std::ios::binary);
  if (!binary_file.is_open()) {
    throw std::runtime_error(
        "Could not open file " + diskann_test_256bin.string());
  }

  binary_file.read((char*)&npoints, 4);
  binary_file.read((char*)&ndim, 4);
  REQUIRE(npoints == 256);
  REQUIRE(ndim == 128);

  auto x = ColMajorMatrix<float>(ndim, npoints);

  binary_file.read((char*)x.data(), npoints * ndim);
  binary_file.close();

  SECTION("vary starts") {
    auto start = GENERATE(0, 17, 127, 128, 129, 254, 255);

    auto g = detail::graph::init_random_nn_graph<float, uint32_t>(x, k_nn);

    auto&& [top_k_scores, top_k, V] =
        greedy_search(g, x, start, x[start], k_nn, L);
    std::sort(begin(top_k), end(top_k));

    // There seems to be some ambiguity in the DiskANN code about whether
    // the start point should be included in the top_k list.  It probably
    // should not be, so comment out for now.
    // CHECK(top_k[0] == start);
    CHECK(std::find(begin(top_k), end(top_k), start) != end(top_k));
    CHECK(std::find(begin(V), end(V), start) != end(V));
  }
}

TEST_CASE("vamana: fmnist", "[vamana]") {
  const bool debug = false;

  using feature_type = float;
  using score_type = float;
  using id_type = uint32_t;

  size_t nthreads = 1;
  size_t L = 7;
  size_t k_nn = L;
  size_t num_queries = 10;
  size_t N = 5000;

  tiledb::Context ctx;
  auto db = tdbColMajorMatrix<test_feature_type>(ctx, fmnist_inputs_uri, N);
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

TEST_CASE("vamana: robust prune hypercube", "[vamana]") {
  bool debug = false;

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

TEST_CASE("vamana: robust prune fmnist", "[vamana]") {
  bool debug = false;

  size_t nthreads = 1;
  size_t k_nn = 5;
  size_t L = 7;
  size_t R = 7;
  size_t num_queries = 10;
  size_t N = 500;
  float alpha = 1.0;

  tiledb::Context ctx;
  auto db = tdbColMajorMatrix<test_feature_type>(ctx, fmnist_inputs_uri, N);
  db.load();
  auto g = detail::graph::init_random_nn_graph<float, uint64_t>(db, L);

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

TEST_CASE("vamana: vamana_index vector diskann_test_256bin", "[vamana]") {
  bool debug = false;

  // should be dim = 128, num = 256
  // npoints, ndims
  uint32_t npoints{0};
  uint32_t ndim{0};

  std::ifstream binary_file(diskann_test_256bin, std::ios::binary);
  if (!binary_file.is_open()) {
    throw std::runtime_error(
        "Could not open file " + diskann_test_256bin.string());
  }

  binary_file.read((char*)&npoints, 4);
  binary_file.read((char*)&ndim, 4);
  REQUIRE(npoints == 256);
  REQUIRE(ndim == 128);

  auto x = ColMajorMatrix<float>(ndim, npoints);

  binary_file.read((char*)x.data(), npoints * ndim);
  binary_file.close();

  size_t L = 100;
  size_t R = 100;
  size_t B = 2;
  auto index = vamana_index<float, uint64_t>(num_vectors(x), L, R, B);

  auto x0 = std::vector<float>(ndim);
  std::copy(x.data(), x.data() + ndim, begin(x0));

  index.train(x);

  // x, 5, 7, 7, 1.0, 1, 1);
  // 0
  // 14
  auto&& [s0, v0] = index.query(x0, 5);
  CHECK(v0[0] == 0);
}

TEST_CASE("vamana: vamana by hand random index", "[vamana]") {
  const bool debug = false;

  size_t num_nodes = 20;

  float alpha_0 = 1.0;
  float alpha_1 = 1.2;

  size_t L_build_ = 2;
  size_t R_max_degree_ = 2;

  auto training_set_ = random_geometric_2D(num_nodes);
  dump_coordinates("coords.txt", training_set_);

  auto g = ::detail::graph::init_random_adj_list<float, uint32_t>(
      training_set_, R_max_degree_);

  auto dimension_ = ::dimension(training_set_);
  auto num_vectors_ = ::num_vectors(training_set_);
  auto graph_ = ::detail::graph::init_random_nn_graph<float, uint64_t>(
      training_set_, R_max_degree_);

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
          graph_, training_set_, medioid_, training_set_[p], 1, L_build_);

      if (debug) {
        std::cout << ":::: Post search prune" << std::endl;
      }
      robust_prune(graph_, training_set_, p, visited, alpha, R_max_degree_);

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

        if (size(tmp) > R_max_degree_) {
          if (debug) {
            std::cout << ":::: Pruning neighbor " << j << std::endl;
          }
          robust_prune(graph_, training_set_, j, tmp, alpha, R_max_degree_);
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
TEST_CASE("vamana: vamana_index geometric 2D graph", "[vamana]") {
  const bool debug = false;

  size_t num_nodes = 200;

  float alpha_0 = 1.0;
  float alpha_1 = 1.2;

  size_t L_build = 15;
  size_t R_max_degree = 15;

  size_t k_nn = 5;

  auto training_set = random_geometric_2D(num_nodes);

  auto idx = vamana_index<float, uint64_t>(
      num_vectors(training_set), L_build, R_max_degree, 0);
  idx.train(training_set);

  auto query = training_set[17];
  auto&& [scores, top_k] = idx.query(query, k_nn);
  CHECK(top_k[0] == 17);

  auto query_mat = ColMajorMatrix<float>(dimension(training_set), 7);
  size_t counter{0};
  for (size_t i : {17, 19, 23, 37, 49, 50, 195}) {
    std::copy(
        training_set[i].data(),
        training_set[i].data() + dimension(training_set),
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

TEST_CASE("vamana: vamana_index siftsmall", "[vamana]") {
  bool debug = false;

  size_t num_nodes = 10000;
  size_t num_queries = 200;

  float alpha_0 = 1.0;
  float alpha_1 = 1.2;

  size_t L_build = 15;
  size_t R_max_degree = 12;

  size_t k_nn = 10;

  tiledb::Context ctx;
  auto training_set =
      tdbColMajorMatrix<float>(ctx, siftsmall_inputs_uri, num_nodes);
  training_set.load();
  auto queries =
      tdbColMajorMatrix<float>(ctx, siftsmall_query_uri, num_queries);
  queries.load();

  auto idx = vamana_index<float, uint64_t>(
      num_vectors(training_set), L_build, R_max_degree, 0);
  idx.train(training_set);

  auto&& [qv_scores, qv_top_k] =
      detail::flat::qv_query_heap(training_set, queries, k_nn, 4);
  auto&& [mat_scores, mat_top_k] = idx.query(queries, k_nn);
  size_t total_intersected = count_intersections(mat_top_k, qv_top_k, k_nn);

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

TEST_CASE("vamana: vamana_index write and read", "[vamana]") {
  size_t L_build{37};
  size_t R_max_degree{41};
  size_t k_nn{10};
  size_t Backtrack{3};

  tiledb::Context ctx;
  std::string vamana_index_uri = "/tmp/tmp_vamana_index";
  auto training_set = tdbColMajorMatrix<float>(ctx, siftsmall_inputs_uri, 0);
  load(training_set);

  auto idx = vamana_index<float, uint64_t>(
      num_vectors(training_set), L_build, R_max_degree, Backtrack);
  idx.train(training_set);

  idx.write_index(ctx, vamana_index_uri, true);
  auto idx2 = vamana_index<float, uint64_t>(ctx, vamana_index_uri);

  // Can't compare groups because a write_index does not create a group
  // @todo Should it?
  // CHECK(idx.compare_group(idx2));

  CHECK(idx.compare_cached_metadata(idx2));
  CHECK(idx.compare_feature_vectors(idx2));
  CHECK(idx.compare_adj_scores(idx2));
  CHECK(idx.compare_adj_ids(idx2));
}
