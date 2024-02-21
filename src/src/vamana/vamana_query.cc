/**
 * @file   vamana/query.cc
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
 * Driver for making a query against a vamana index, as given by the index_uri.
 *
 * The driver searches using a previously-stored index.
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
using groundtruth_type = uint64_t;

static constexpr const char USAGE[] =
    R"(vamana: C++ cli for vamana query
  Usage:
      vamana (-h | --help)
      vamana --index_uri URI --query_uri URI [--ftype TYPE] [--idtype TYPE] [--groundtruth_uri URI]
             [--Lbuild NN] [--nqueries NN] [--k NN]
             [--nthreads NN] [--validate] [--log FILE] [--stats] [-d] [-v] [--dump NN]
             [--bfs]

  Options:
      -h, --help              show this screen
      -b, --bfs               run bfs on the index
      --index_uri URI         group URI ov vamana index
      --query_uri URI         query URI with feature vectors to search for
      --ftype TYPE            data type of feature vectors [default: float]
      --idtype TYPE           data type of ids [default: uint64]
      --groundtruth_uri URI   ground truth URI
      -L, --Lbuild NN         size of search list
      -k, --k NN              number of nearest neighbors [default: 1]
      --nqueries NN           size of queries subset to compare (0 = all) [default: 0]
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

  std::string index_uri = args["--index_uri"].asString();
  std::string query_uri = args["--query_uri"].asString();

  size_t k_nn = args["--k"].asLong();
  size_t nqueries = args["--nqueries"].asLong();
  size_t nthreads = args["--nthreads"].asLong();



  auto run_query = [&]<class feature_type, class id_type>() {
    tiledb::Context ctx;
    auto idx = vamana_index<feature_type, id_type>(ctx, index_uri);
    auto queries = tdbColMajorMatrix<feature_type>(ctx, query_uri, nqueries);
    queries.load();

    auto Lbuild = args["--Lbuild"] ?
                      std::optional<size_t>(args["--Lbuild"].asLong()) :
                      std::nullopt;

    auto query_time = log_timer("query time", true);

    if (args["--bfs"].asBool()) {

      auto query_time = log_timer("bfs time", true);
      tiledb::Context ctx;
      auto idx = vamana_index<feature_type, id_type>(ctx, index_uri);
      /*auto parents = */idx.bfs_O1(queries, k_nn, *Lbuild);
      query_time.stop();

      return 0;
    }


    auto&& [top_k_scores, top_k] = idx.query(queries, k_nn, Lbuild);

    query_time.stop();

    if (args["--groundtruth_uri"]) {
      auto groundtruth_uri = args["--groundtruth_uri"].asString();

      auto groundtruth =
          tdbColMajorMatrix<groundtruth_type>(ctx, groundtruth_uri, nqueries);
      groundtruth.load();

      if (debug) {
        std::cout << std::endl;

        debug_matrix(groundtruth, "groundtruth");
        debug_slice(groundtruth, "groundtruth");

        std::cout << std::endl;
        debug_matrix(top_k, "top_k");
        debug_slice(top_k, "top_k");

        std::cout << std::endl;
      }

      size_t total_groundtruth = num_vectors(top_k) * dimension(top_k);
      size_t total_intersected = count_intersections(top_k, groundtruth, k_nn);

      float recall = ((float)total_intersected) / ((float)total_groundtruth);
      // std::cout << "# total intersected = " << total_intersected << " of "
      //           << total_groundtruth << " = "
      //           << "R@" << k_nn << " of " << recall << std::endl;

      if (args["--log"]) {
        idx.log_index();
        dump_logs(
            args["--log"].asString(),
            "vamana",
            nqueries,
            {},
            k_nn,
            nthreads,
            recall);
      }
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
    run_query.operator()<float, uint64_t>();
  } else if (feature_type == "float" && id_type == "uint32") {
    run_query.operator()<float, uint32_t>();
  } else if (feature_type == "uint8" && id_type == "uint64") {
    run_query.operator()<uint8_t, uint64_t>();
  } else if (feature_type == "uint8" && id_type == "uint32") {
    run_query.operator()<uint8_t, uint32_t>();
  } else {
    std::cout << "Unsupported feature type " << feature_type;
    std::cout << " and/or unsupported id_type " << id_type << std::endl;
    return 1;
  }
}
