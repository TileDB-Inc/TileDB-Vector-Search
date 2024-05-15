/**
 * @file   ivf_index.cc
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
#include "detail/linalg/matrix.h"
#include "detail/linalg/tdb_matrix.h"
#include "detail/linalg/vector.h"
#include "index/ivf_flat_index.h"

bool verbose = false;
bool debug = false;

bool enable_stats = false;
std::vector<json> core_stats;

using score_type = float;

static constexpr const char USAGE[] =
    R"(ivf_index: C++ cli for creating ivf index
Usage:
    ivf_index (-h | --help)
    ivf_index --db_uri URI --index_uri URI [--ftype TYPE] [--idtype TYPE] [--pxtype TYPE]
                 [--init TYPE] [--num_clusters NN] [--max_iter NN] [--tol NN]
                 [--nthreads NN] [--log FILE] [--stats] [-d] [-v] [--dump NN]

Options:
    -h, --help              show this screen
    --db_uri URI            database URI with feature vectors
    --index_uri URI         group URI for storing ivf index
    --ftype TYPE            data type of feature vectors [default: float]
    --idtype TYPE           data type of ids [default: uint64]
    --pxtype TYPE           data type of partition index [default: uint64]
    -i, --init TYPE         initialization type, kmeans++ or random [default: random]
    --num_clusters NN       number of clusters/partitions, 0 = sqrt(N) [default: 0]
    --max_iter NN           max number of iterations for kmeans [default: 10]
    --tol NN                tolerance for kmeans [default: 1e-4]
    --nthreads N            number of threads to use in parallel loops (0 = all) [default: 0]
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

  debug = args["--debug"].asBool();
  verbose = args["--verbose"].asBool();
  enable_stats = args["--stats"].asBool();

  std::string db_uri = args["--db_uri"].asString();
  std::string index_uri = args["--index_uri"].asString();
  size_t nthreads = args["--nthreads"].asLong();
  size_t max_iter = args["--max_iter"].asLong();
  size_t num_clusters = args["--num_clusters"].asLong();
  std::string tol_str = args["--tol"].asString();
  float tolerance = std::atof(tol_str.c_str());

  auto init_type = args["--init"].asString() == "random" ?
                       kmeans_init::random :
                       kmeans_init::kmeanspp;

  auto run_index = [&]<class feature_type, class id_type, class px_type>() {
    tiledb::Context ctx;
    auto X = tdbColMajorMatrix<feature_type>(ctx, db_uri);
    X.load();

    auto dim = dimensions(X);

    auto idx = ivf_flat_index<feature_type, id_type, px_type>(
        /* dim, */ num_clusters, max_iter, tolerance);
    idx.train(X, init_type);
    idx.add(X);
    idx.write_index(ctx, index_uri);

    if (args["--log"]) {
      //     idx.log_index();
      dump_logs(args["--log"].asString(), "ivf", {}, {}, {}, {}, {});
    }
    if (enable_stats) {
      std::cout << json{core_stats}.dump() << std::endl;
    }
  };
  auto feature_type = args["--ftype"].asString();
  auto id_type = args["--idtype"].asString();
  auto px_type = args["--pxtype"].asString();

  if (feature_type != "float" && feature_type != "uint8") {
    std::cout << "Unsupported feature type " << feature_type << std::endl;
    return 1;
  }
  if (id_type != "uint64" && id_type != "uint32") {
    std::cout << "Unsupported id type " << id_type << std::endl;
    return 1;
  }
  if (px_type != "uint64" && px_type != "uint32") {
    std::cout << "Unsupported px type " << px_type << std::endl;
    return 1;
  }

  if (feature_type == "float" && id_type == "uint64" && px_type == "uint64") {
    run_index.operator()<float, uint64_t, uint64_t>();
  } else if (
      feature_type == "float" && id_type == "uint32" && px_type == "uint64") {
    run_index.operator()<float, uint32_t, uint64_t>();
  } else if (
      feature_type == "uint8" && id_type == "uint64" && px_type == "uint64") {
    run_index.operator()<uint8_t, uint64_t, uint64_t>();
  } else if (
      feature_type == "uint8" && id_type == "uint32" && px_type == "uint64") {
    run_index.operator()<uint8_t, uint32_t, uint64_t>();
  } else if (
      feature_type == "float" && id_type == "uint64" && px_type == "uint32") {
    run_index.operator()<float, uint64_t, uint32_t>();
  } else if (
      feature_type == "float" && id_type == "uint32" && px_type == "uint32") {
    run_index.operator()<float, uint32_t, uint32_t>();
  } else if (
      feature_type == "uint8" && id_type == "uint64" && px_type == "uint32") {
    run_index.operator()<uint8_t, uint64_t, uint32_t>();
  } else if (
      feature_type == "uint8" && id_type == "uint32" && px_type == "uint32") {
    run_index.operator()<uint8_t, uint32_t, uint32_t>();
  } else {
    std::cout << "Unsupported feature type " << feature_type;
    std::cout << " and/or unsupported id_type " << id_type << std::endl;
    return 1;
  }
}
