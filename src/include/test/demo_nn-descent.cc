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

#include <cmath>
#include <matplot/matplot.h>

#include <vector>
#include "detail/graph/nn-graph.h"
#include "query_common.h"

void show_graph(auto&& g) {
  using namespace matplot;

  std::vector<std::pair<size_t, size_t>> edges;
  for (size_t i = 0; i < g.num_vertices(); ++i) {
    for (auto&& [_, j] : out_edges(g, i)) {
      edges.emplace_back(i, j);
    }
  }
  digraph(edges, "-.dr")->show_labels(false);
  //    digraph(edges)->show_labels(false);
  show();
}


int main() {

  auto g = ::detail::graph::init_random_nn_graph<float>(sift_base, 5);
  show_graph(g);

  for (size_t i = 0; i < 4; ++i) {
    auto num_updates = nn_descent_step_all(g, sift_base);
    std::cout << "num_updates: " << num_updates << std::endl;
    show_graph(g);
  }

  return 0;
}
