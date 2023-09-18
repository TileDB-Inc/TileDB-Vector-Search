
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
#include "detail/graph/adj_list.h"
#include "detail/graph/nn-graph.h"
#include "detail/graph/vamana.h"
#include "detail/linalg/matrix.h"
#include "scoring.h"

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

#if 1
  auto X = random_geometric_2D(num_nodes);
  dump_coordinates("coords.txt", X);
  auto idx = detail::graph::vamana_index<float, size_t, size_t>(num_nodes, L, R, 0);
  idx.train(X);

#else
#if 1
  auto X = random_geometric_2D(num_nodes);
  dump_coordinates("coords.txt", X);

  auto g = ::detail::graph::init_random_adj_list<float, size_t>(X, R);
  // std::cout << "num_vertices " << g.num_vertices() << std::endl;
#else
  size_t M = 5;
  size_t N = 7;
  int one_two = 3;
  auto&& [X, edges] = ([one_two, M, N]() {
    if (one_two == 1) {
      return gen_uni_grid(M, N);
    } else if (one_two == 2) {
      return gen_bi_grid(M, N);
    } else {
      return gen_star_grid(M, N);
    }
  })();
  dump_coordinates("coords.txt", X);
  detail::graph::adj_list<float, size_t> g(M*N);
  for (auto&& [src, dst] : edges) {
    g.add_edge(src, dst, sum_of_squares_distance{}(X[src], X[dst]));
  }
#endif

  dump_edgelist("edges_" + std::to_string(0) + ".txt", g);

  size_t img_count = 0;
  auto start = medioid(X);

  for (float alpha : {alpha_0, alpha_1}) {
    for (size_t p = 0; p < X.num_cols(); ++p) {
      ++img_count;

      auto&& [top_k_scores, top_k, V] = greedy_search(g, X, start, X[p], 1, L);
      robust_prune(g, X, p, V, alpha, R);

      auto tmp_p = std::vector<size_t>();
      for (auto&& [qq, pp] : g.out_edges(p)) {
        tmp_p.push_back(pp);
      }

      for (auto&& j : tmp_p) {
        if (j == p) {
          continue;
        }

        // out_degree of j \cup p
        std::vector <size_t> tmp;
        tmp.push_back(p);
        for (auto&& [_, k] : g.out_edges(j)) {
          tmp.push_back(k);
        }

        if (tmp.size() > R) {

          // prune Nout(j) \cup p
          robust_prune(g, X, j, tmp, alpha, R);
        } else {
          g.add_edge(j, p, sum_of_squares_distance()(X[p], X[j]));
        }
      }

      if ((img_count) % 10 == 0) {
        dump_edgelist("edges_" + std::to_string(img_count) + ".txt", g);
      }
    }
  }
#endif
}
