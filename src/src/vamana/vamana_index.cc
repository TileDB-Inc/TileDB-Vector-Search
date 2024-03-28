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
 *
 * The driver will build a vamana index and store it in the given TileDB group.
 * The type of the feature vectors and ids can be specified using the --ftype
 * and --idtype options. The default feature type is float and the default id
 * type is uint64.
 */

#include "docopt.h"

#include <cmath>
#include "detail/graph/nn-graph.h"
#include "detail/linalg/matrix.h"
#include "detail/linalg/tdb_matrix.h"
#include "detail/linalg/vector.h"
#include "index/vamana_index.h"

bool verbose = false;
bool debug = false;

bool enable_stats = false;
std::vector<json> core_stats;

using score_type = float;

static constexpr const char USAGE[] =
    R"(vamana_index: C++ cli for creating vamana index
  Usage:
      vamana_index (-h | --help)
      vamana_index (-b | --bfs)
      vamana_index --inputs_uri URI --index_uri URI [--ftype TYPE] [--idtype TYPE] [--force]
                  [--max_degree NN] [--Lbuild NN] [--backtrack NN] [--alpha FF]
                  [--nthreads NN] [--log FILE] [--stats] [-d] [-v] [--dump NN]

  Options:
      -h, --help              show this screen
      --inputs_uri URI        database URI with feature vectors
      --index_uri URI         group URI for storing vamana index
      --ftype TYPE            data type of feature vectors [default: float]
      --idtype TYPE           data type of ids [default: uint64]
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

  std::string db_uri = args["--inputs_uri"].asString();
  std::string index_uri = args["--index_uri"].asString();
  size_t nthreads = args["--nthreads"].asLong();

  size_t dump = args["--dump"].asLong();

  auto run_index = [&]<class feature_type, class id_type>() {
    tiledb::Context ctx;
    auto X = tdbColMajorMatrix<feature_type>(ctx, db_uri);
    X.load();

    auto idx = vamana_index<feature_type, id_type>(
        num_vectors(X), Lbuild, max_degree, backtrack);
    idx.train(X);
    idx.write_index(ctx, index_uri, overwrite);

    if (args["--log"]) {
      idx.log_index();
      dump_logs(args["--log"].asString(), "vamana", {}, {}, {}, {}, {});
    }
    if (enable_stats) {
      std::cout << json{core_stats}.dump() << std::endl;
    }
  };
  auto feature_type = args["--ftype"].asString();
  auto id_type = args["--idtype"].asString();

  if (feature_type != "float" && feature_type != "uint8") {
    std::cout << "Unsupported feature type " << feature_type << std::endl;
    return 1;
  }
  if (id_type != "uint64" && id_type != "uint32") {
    std::cout << "Unsupported id type " << id_type << std::endl;
    return 1;
  }

  if (feature_type == "float" && id_type == "uint64") {
    run_index.operator()<float, uint64_t>();
  } else if (feature_type == "float" && id_type == "uint32") {
    run_index.operator()<float, uint32_t>();
  } else if (feature_type == "uint8" && id_type == "uint64") {
    run_index.operator()<uint8_t, uint64_t>();
  } else if (feature_type == "uint8" && id_type == "uint32") {
    run_index.operator()<uint8_t, uint32_t>();
  } else {
    std::cout << "Unsupported feature type " << feature_type;
    std::cout << " and/or unsupported id_type " << id_type << std::endl;
    return 1;
  }
}
