/**
 * @file   flat_l2.cc
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
 * Driver program for "flat" feature vector search.  Can read and search
 * from local files in "ANN" format or from simple dense TileDB arrays.
 *
 * The program has a lot of different options to enable exploration of the
 * performance of different formulations of the search algorithms.  It turns
 * out (not surprisingly) that for many of the multi-query problems, that
 * a gemm-based algorithm is fastest.  For other searches, particular with
 * just a small number of query vectors (e.g. 1), a brute-force search is
 * fastest.
 *
 * This program currently uses `sift_db` and `sift_array` structures to
 * hold the data, depending on whether the data comes from local file
 * or from a TileDB array but I have since written some better abstractions.
 *
 * Originally, I represented sets of feature vectors as a `std::vector` of
 * spans over a single allocation of memory.  The very first approach
 * used a `std::vector` to provide that, but I am gradually migrating
 * to using `stdx::mdspan` instead.  There are very efficient ways to
 * allocate memory and `mdspan` is much more lightweight than a `vector`
 * of `span`.
 *
 * Most of the functionality in this driver should be fairly straightforward
 * to follow.  The search algorithms all do a search, find the indices
 * of the top k matches, and compare those results to ground truth.
 *
 * Determining top k is done in one of two ways.  The "hard way" is to compute
 * all scores between the query and the database vectors and then find the
 * top k scores using `nth_element`.  The "easy way" is to use a priority
 * queue to keep a running list of the top k scores.  The easy way is much
 * faster in the qv and vq cases.  The hard way is currently the only way
 * to do top k in gemm, yet gemm tends to be the fastest.
 *
 * The difference between vq vs qv is the ordering of the two nested loops:
 * vq loops over the database vectors and then the queries, while qv loops
 * over the queries and then the database vectors.  There are some
 * ramifications in terms of resource usage and execution time between the
 * two approaches.
 *
 * With the vector of spans approach, each element of the outer std::vector
 * corresponds to a vector.  There isn't really an orientation per se.
 * I.e., A[i] returns a span comprising the ith vector in A.
 *
 *
 */

#include <algorithm>
#include <cmath>
#include <fstream>
// #include <execution>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "docopt.h"

#include "stats.h"
#include "utils/print_types.h"
#include "utils/timer.h"

bool verbose = false;
bool debug = false;

bool enable_stats = false;
std::vector<json> core_stats;

using score_type = float;
using groundtruth_type = int32_t;

static constexpr const char USAGE[] =
    R"(flat_l2: feature vector search with flat index.
  Usage:
      flat_l2 (-h | --help)
      flat_l2 --inputs_uri URI --query_uri URI [--groundtruth_uri URI] [--output_uri URI]
          [--k NN] [--nqueries NN]
          [--alg ALGO] [--finite] [--blocksize NN]
          [--nthreads N] [--region REGION] [--validate] [--log FILE] [--stats] [-d] [-v]

  Options:
      -h, --help              show this screen
      --inputs_uri URI            database URI with feature vectors
      --query_uri URI         query URI with feature vectors to search for
      --groundtruth_uri URI   ground truth URI
      --output_uri URI        output URI for results
      --ftype TYPE            data type of feature vectors [default: float]
      --idtype TYPE           data type of ids [default: uint64]
      --k NN                  number of nearest neighbors to find [default: 10]
      --nqueries NN           size of queries subset to compare (0 = all) [default: 0]
      --alg ALGO              which algorithm to use for comparisons [default: vq_heap]
      --infinite              use infinite RAM algorithm [default: false]
      --finite                (legacy) use finite RAM (out of core) algorithm [default: true]
      --blocksize NN          number of vectors to process in an out of core block (0 = all) [default: 0]
      --nthreads N            number of threads to use in parallel loops (0 = all) [default: 0]
      --region REGION         AWS region [default: us-east-1]
      --log FILE              log info to FILE (- for stdout)
      --stats                 log TileDB stats [default: false]
      -d, --debug             run in debug mode [default: false]
      -v, --verbose           run in verbose mode [default: false]
)";

int main(int argc, char* argv[]) {
  scoped_timer _(tdb_func__ + std::string("all inclusive query time"));

  std::vector<std::string> strings(argv + 1, argv + argc);
  auto args = docopt::docopt(USAGE, strings, true);

  if (args["--help"].asBool()) {
    std::cout << USAGE << std::endl;
    return 0;
  }

  verbose = args["--verbose"].asBool();
  enable_stats = args["--stats"].asBool();

  std::string db_uri = args["--inputs_uri"].asString();
  std::string query_uri = args["--query_uri"].asString();
  std::string groundtruth_uri =
      args["--groundtruth_uri"] ? args["--groundtruth_uri"].asString() : "";

  auto alg_name = args["--alg"].asString();
  std::cout << "# Using " << alg_name << std::endl;

  size_t k = args["--k"].asLong();
  size_t nthreads = args["--nthreads"].asLong();
  size_t nqueries = args["--nqueries"].asLong();
  size_t blocksize = args["--blocksize"].asLong();

  // @todo make global
  if (nthreads == 0) {
    nthreads = std::thread::hardware_concurrency();
  }

  auto run_flat_l2 = [&]<class feature_type, class id_type> {
    ms_timer load_time{"Load database, query, and ground truth arrays"};

    tiledb::Context ctx;

    auto db =
        tdbColMajorMatrix<feature_type>(ctx, db_uri, blocksize);  // blocked
    db.load();

    auto query = tdbColMajorMatrix<feature_type>(
        ctx, query_uri, nqueries);  // just a slice
    query.load();

    load_time.stop();
    std::cout << load_time << std::endl;

    // @todo decide on what the type of top_k::value should be
    auto [top_k_scores, top_k] = [&]() {
      if (alg_name == "vq_heap" || alg_name == "vq") {
        if (verbose) {
          std::cout << "# Using vq_heap" << std::endl;
        }
        return detail::flat::vq_query_heap(db, query, k, nthreads);
      } else if (alg_name == "vq_heap_2" || alg_name == "vq2") {
        if (verbose) {
          std::cout << "# Using vq_heap_2" << std::endl;
        }
        auto foo = detail::flat::vq_query_heap_2(db, query, k, nthreads);
        return detail::flat::vq_query_heap_2(db, query, k, nthreads);
      } else if (alg_name == "qv_tiled") {
        if (verbose) {
          std::cout << "# Using qv_tiled" << std::endl;
        }
        return detail::flat::qv_query_heap_tiled(db, query, k, nthreads);
      } else if (alg_name == "qv_heap" || alg_name == "qv") {
        if (verbose) {
          std::cout << "# Using qv_query (qv_heap)" << std::endl;
        }
        return detail::flat::qv_query_heap(db, query, k, nthreads);
      }
#ifdef TILEDB_VS_ENABLE_BLAS
#if 0
    else if (alg_name == "gemm") {
      if (args["--blocksize"]) {
        std::cout << "# Using blocked_gemm" << std::endl;
        return detail::flat::blocked_gemm_query(db, query, k, nthreads);
      } else {
        std::cout << "# Using gemm" << std::endl;
        return detail::flat::gemm_query(db, query, k, nthreads);
      }
    }
#endif
#endif
      throw std::runtime_error(
          "incorrect or unset algorithm type: " + alg_name);
    }();

    if (!groundtruth_uri.empty()) {
      auto groundtruth =
          tdbColMajorMatrix<groundtruth_type>(ctx, groundtruth_uri);
      groundtruth.load();

      if (!validate_top_k(top_k, groundtruth)) {
        std::cout << "Validation failed" << std::endl;
      } else {
        if (verbose) {
          std::cout << "Validation succeeded" << std::endl;
        }
      }
    }

    if (args["--output_uri"]) {
      write_matrix(ctx, top_k, args["--output_uri"].asString());
    }

    if (args["--log"]) {
      dump_logs(
          args["--log"].asString(), alg_name, nqueries, 0, k, nthreads, 0);
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
    run_flat_l2.operator()<float, uint64_t>();
  } else if (feature_type == "float" && id_type == "uint32") {
    run_flat_l2.operator()<float, uint32_t>();
  } else if (feature_type == "uint8" && id_type == "uint64") {
    run_flat_l2.operator()<uint8_t, uint64_t>();
  } else if (feature_type == "uint8" && id_type == "uint32") {
    run_flat_l2.operator()<uint8_t, uint32_t>();
  } else {
    std::cout << "Unsupported feature type " << feature_type;
    std::cout << " and/or unsupported id_type " << id_type << std::endl;
    return 1;
  }
}
