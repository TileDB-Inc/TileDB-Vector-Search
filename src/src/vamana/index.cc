/**
* @file   vamana/index.cc
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
 * Driver for building vamana index.
 */

#include "docopt.h"

#include <cmath>
#include "detail/graph/nn-graph.h"
#include "detail/graph/vamana.h"
#include "detail/linalg/vector.h"
#include "detail/linalg/matrix.h"
#include "detail/linalg/tdb_matrix.h"

bool verbose = false;
bool debug = false;

bool enable_stats = false;
std::vector<json> core_stats;

// @todo Use type erased version of index
#if 0
using feature_type = uint8_t;
#else
using feature_type = float;
#endif
using id_type = uint64_t;
using score_type = float;

using groundtruth_type = int32_t;

static constexpr const char USAGE[] =
    R"(vamana/index: C++ cli for creating vamana index
  Usage:
      vamana (-h | --help)
      vamana --db_uri URI --index_uri URI [--force]
             [--max_degree NN] [--Lbuild NN] [--backtrack NN] [--alpha FF]
             [--nthreads NN] [--log FILE] [--stats] [-d] [-v] [--dump NN]

  Options:
      -h, --help              show this screen
      --db_uri URI            database URI with feature vectors
      --index_uri URI         group URI for storing vamana index
      -f, --force             overwrite index if it exists [default:false]
      -R, --max_degree NN     maximum degree of graph [default: 64]
      -L, --Lbuild NN         size of search list while building [default: 100]
      -B, --backtrack NN      size of backtrack list [default: 100]
      -a, --alpha FF          pruning parameter [default: 1.2]
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

  bool overwrite = args["--force"].asBool();
  debug = args["--debug"].asBool();
  verbose = args["--verbose"].asBool();
  enable_stats = args["--stats"].asBool();

  std::string db_uri = args["--db_uri"].asString();
  std::string index_uri = args["--index_uri"].asString();
  size_t nthreads = args["--nthreads"].asLong();

  size_t dump = args["--dump"].asLong();

  tiledb::Context ctx;
  auto X = tdbColMajorMatrix<feature_type>(ctx, db_uri);
  X.load();

  auto idx = detail::graph::vamana_index<score_type, id_type>(num_vectors(X), Lbuild, max_degree, backtrack);
  idx.train(X);
  idx.write_index(index_uri, overwrite);

  if (args["--log"]) {
    idx.log_index();
    dump_logs(
        args["--log"].asString(),
        "vamana",
        {},
        {},
        {},
        {},
        {});
  }
  if (enable_stats) {
    std::cout << json{core_stats}.dump() << std::endl;
  }
}

