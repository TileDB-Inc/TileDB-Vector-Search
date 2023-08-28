
/**
 * @file  demo_nn-descent.h
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
 * Demonstrate / verify random graph generation.
 *
 */

#include <matplot/matplot.h>
#include <cmath>
#include <random>
#include "detail/graph/vamana.h"
#include "detail/graph/nn-graph.h"
#include "detail/linalg/matrix.h"




auto random_geomtric_2D(size_t N) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> coord(-1.0, 1.0);

  auto X = ColMajorMatrix<float> (2, N);

  for (size_t i = 0; i < N; ++i) {
    X(0, i) = coord(gen);
    X(1, i) = coord(gen);
  }

  return X;
}


int main() {
  size_t k_nn = 5;
  size_t L = 7;
  size_t R = 7;
  float alpha = 1.0;

  auto X = random_geomtric_2D(51);
  auto g = ::detail::graph::init_random_nn_graph<float>(X, k_nn);

  std::vector<std::pair<size_t, size_t>> edges;
  for (size_t i = 0; i < g.num_vertices(); ++i) {
    for (auto&& [_, j] : out_edges(g, i)) {
      edges.emplace_back(i, j);
    }
  }

  matplot::digraph(edges, "-.dr")->show_labels(false);
  //    digraph(edges)->show_labels(false);
  matplot::show();

  auto start = medioid(X);
  for (size_t p = 0; p < X.num_cols(); ++p) {
    auto nbd = std::vector<size_t>(k_nn);
    auto V = greedy_search(g, X, start, X[p], k_nn, L, nbd);
    robust_prune(g, X, p, V, alpha, R);
    if (p % 50 == 0) {
      std::vector<std::pair<size_t, size_t>> edges;
      for (size_t i = 0; i < g.num_vertices(); ++i) {
        for (auto&& [_, j] : out_edges(g, i)) {
          edges.emplace_back(i, j);
        }
      }

      matplot::digraph(edges, "-.dr")->show_labels(false);
      //    digraph(edges)->show_labels(false);
      matplot::show();
    }
  }

}
