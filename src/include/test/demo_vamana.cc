
/**
 * @file  demo_vamana.h
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
#include "detail/graph/nn-graph.h"
#include "detail/graph/vamana.h"
#include "detail/linalg/matrix.h"

#include "gen_graphs.h"

#include <docopt.h>

static constexpr const char USAGE[] =
    R"(demo_vamana: test vamana index
  Usage:
      demo_vamana (-h | --help)
      demo_vamana [--max_degree NN] [--Lbuild NN] [--alpha FF] [--k_nn NN]

  Options:
      -h, --help              show this screen
      -R, --max_degree NN     maximum degree of graph [default: 64]
      -L, --Lbuild NN         size of search list while building [default: 100]
      -a, --alpha FF          pruning parameter [default: 1.2]
      -k, --k_nn NN           number of nearest neighbors [default: 1]
)";



int main(int argc, char* argv[]) {
  std::vector<std::string> strings(argv + 1, argv + argc);
  auto args = docopt::docopt(USAGE, strings, true);

  if (args["--help"].asBool()) {
    std::cout << USAGE << std::endl;
    return 0;
  }
  size_t L = args["--Lbuild"].asLong();
  size_t R = args["--max_degree"].asLong();
  std::string alpha_str = args["--alpha"].asString();
  float alpha_1 = std::atof(alpha_str.c_str());

  size_t k_nn = args["--k_nn"].asLong();
  float alpha_0 = 1.0;
  size_t num_nodes{200};

  auto X = random_geometric_2D(num_nodes);
  dump_coordinates("coords.txt", X);
  auto g = ::detail::graph::init_random_nn_graph<float>(X, 2*R);
  std::cout << "num_vertices " << g.num_vertices() << std::endl;

  std::vector<std::tuple<size_t, size_t>> edges;
  for (size_t i = 0; i < g.num_vertices(); ++i) {
    for (auto&& [_, j] : out_edges(g, i)) {
      edges.emplace_back(i, j);
    }
  }

  dump_edgelist("edges_" + std::to_string(0) + ".txt", edges);

  for (float alpha : {alpha_0, alpha_1}) {
    auto start = medioid(X);
    for (size_t p = 0; p < X.num_cols(); ++p) {

      auto nbd = std::vector<size_t>(k_nn);
      auto V = greedy_search(g, X, start, X[p], k_nn, L, nbd);
      robust_prune(g, X, p, V, alpha, R);
      for (auto&& [i, j] : g.out_edges(p)) {
        if (g.out_degree(j) >= R) {
          robust_prune(g, X, j, X[j], alpha, R);
        }
      }
      if ((p+1) % 20 == 0) {
        std::vector<std::tuple<size_t, size_t>> edges;
        for (size_t i = 0; i < g.num_vertices(); ++i) {
          for (auto&& [_, j] : out_edges(g, i)) {
            edges.emplace_back(i, j);
          }
          dump_edgelist("edges_" + std::to_string(p + 1) + ".txt", edges);
        }
      }
    }
  }
}
