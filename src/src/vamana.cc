/**
* @file   flat.cc
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
 * Driver for vamana index.
*/

#include <docopt.h>

#include <cmath>
#include "detail/graph/nn-graph.h"
#include "detail/graph/vamana.h"
#include "detail/linalg/vector.h"
#include "detail/linalg/matrix.h"
#include "detail/linalg/tdb_matrix.h"

bool verbose = false;
bool debug = false;
bool global_debug = false;

bool enable_stats = false;
std::vector<json> core_stats;

#if 1
using db_type = uint8_t;
#else
using db_type = float;
#endif

using groundtruth_type = int32_t;

static constexpr const char USAGE[] =
    R"(vamana: test vamana index
  Usage:
      vamana (-h | --help)
      vamana --db_uri URI --query_uri URI [--groundtruth_uri URI] [--k NN] [--nqueries NN]
             [--max_degree NN] [--Lbuild NN] [--backtrack NN] [--alpha FF] [--k_nn NN]
             [--nthreads NN] [--validate] [--log FILE] [--stats] [-d] [-v] [--dump NN]

  Options:
      -h, --help              show this screen
      --db_uri URI            database URI with feature vectors
      --query_uri URI         query URI with feature vectors to search for
      --groundtruth_uri URI   ground truth URI
      -R, --max_degree NN     maximum degree of graph [default: 64]
      -L, --Lbuild NN         size of search list while building [default: 100]
      -B, --bactrack NN       size of backtrack list [default: 100]
      -a, --alpha FF          pruning parameter [default: 1.2]
      -k, --k NN              number of nearest neighbors [default: 1]
      --nqueries NN           size of queries subset to compare (0 = all) [default: 0]
      --nthreads N            number of threads to use in parallel loops (0 = all) [default: 0]
      -D, --dump NN           dump Nth iteration graph to file (0 = none) [default: 0]
      --log FILE              log info to FILE (- for stdout)
      --stats                 log TileDB stats [default: false]
      -d, --debug             run in debug mode [default: false]
      -v, --verbose           run in verbose mode [default: false]
)";




int main(int argc, char* argv[]) {
  std::vector<std::string> strings(argv + 1, argv + argc);
  auto args = docopt::docopt(USAGE, strings, true);

  if (args["--help"].asBool()) {
    std::cout << USAGE << std::endl;
    return 0;
  }

  float alpha_0 = 1.0;

  size_t Lbuild = args["--Lbuild"].asLong();
  size_t max_degree = args["--max_degree"].asLong();
  size_t backtrack = args["--backtrack"].asLong();
  std::string alpha_str = args["--alpha"].asString();
  float alpha_1 = std::atof(alpha_str.c_str());

  global_debug = debug = args["--debug"].asBool();
  verbose = args["--verbose"].asBool();
  enable_stats = args["--stats"].asBool();

  std::string db_uri = args["--db_uri"].asString();
  std::string query_uri = args["--query_uri"].asString();

  size_t k_nn = args["--k"].asLong();
  size_t nqueries = args["--nqueries"].asLong();
  size_t nthreads = args["--nthreads"].asLong();

  size_t dump = args["--dump"].asLong();

  tiledb::Context ctx;
  auto X = tdbColMajorMatrix<db_type>(ctx, db_uri);
  X.load();

  auto idx = detail::graph::vamana_index<float>(num_vectors(X), Lbuild, max_degree, backtrack);
  idx.train(X);
  nqueries = 1;
  auto&& [top_k_scores, top_k] = idx.query(X[0], k_nn);

  if (args["--groundtruth_uri"]) {
    auto groundtruth_uri = args["--groundtruth_uri"].asString();

    auto groundtruth =
        tdbColMajorMatrix<groundtruth_type>(ctx, groundtruth_uri, nqueries);
    groundtruth.load();

    if (global_debug) {
      std::cout << std::endl;

      debug_matrix(groundtruth, "groundtruth");
      debug_slice(groundtruth, "groundtruth");

      std::cout << std::endl;
      debug_matrix(top_k, "top_k");
      debug_slice(top_k, "top_k");

      std::cout << std::endl;
    }

    size_t total_intersected{0};
    size_t total_groundtruth = num_vectors(top_k) * dimension(top_k);

    if constexpr (feature_vector_array<decltype(top_k)>) {
      for (size_t i = 0; i < num_vectors(top_k); ++i) {
        std::sort(begin(top_k[i]), end(top_k[i]));
        std::sort(begin(groundtruth[i]), begin(groundtruth[i]) + k_nn);
        total_intersected += std::set_intersection(
            begin(top_k[i]),
            end(top_k[i]),
            begin(groundtruth[i]),
            end(groundtruth[i]),
            assignment_counter{});
      }
    } else {
      std::sort(begin(top_k), end(top_k));
      std::sort(begin(groundtruth[0]), begin(groundtruth[0]) + k_nn);
      total_intersected += std::set_intersection(
          begin(top_k),
          end(top_k),
          begin(groundtruth[0]),
          end(groundtruth[0]),
          assignment_counter{});
    }

    float recall = ((float)total_intersected) / ((float)total_groundtruth);
    std::cout << "# total intersected = " << total_intersected << " of "
              << total_groundtruth << " = "
              << "R@" << k_nn << " of " << recall << std::endl;
  }
}

