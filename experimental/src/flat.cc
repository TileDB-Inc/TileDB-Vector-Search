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

#include <docopt.h>

#include "defs.h"
#include "ivf_query.h"
#include "stats.h"
#include "utils/timer.h"

bool        verbose      = false;
bool        debug        = false;
bool        global_debug = false;
std::string global_region { "us-east-1" };

static constexpr const char USAGE[] =
    R"(flat: feature vector search with flat index.
  Usage:
      flat (-h | --help)
      flat --db_uri URI --q_uri URI [--g_uri URI] [--output_uri URI] [--order ORDER] [--k NN]
          [--block N] [--nqueries N] [--nthreads N] [--nth] [--validate] [--log FILE] [-d] [-v]

  Options:
      -h, --help            show this screen
      --db_uri URI          database URI with feature vectors
      --q_uri URI           query URI with feature vectors to search for
      --g_uri URI           ground true URI
      --output_uri URI      output URI for results
      --order ORDER         which ordering to do comparisons [default: gemm]
      --k NN                number of nearest neighbors to find [default: 10]
      --block N             block database with size N (0 = no blocking) [default: 0]
      --nqueries N          size of queries subset to compare (0 = all) [default: 0]
      --nthreads N          number of threads to use in parallel loops (0 = all) [default: 0]
      --nth                 use nth_element for top k [default: false]
      --log FILE            log info to FILE (- for stdout)
      -V, --validate        validate results [default: false]
      -d, --debug           run in debug mode [default: false]
      -v, --verbose         run in verbose mode [default: false]
)";

int main(int argc, char* argv[]) {
  std::vector<std::string> strings(argv + 1, argv + argc);
  auto                     args = docopt::docopt(USAGE, strings, true);

  if (args["--help"].asBool()) {
    std::cout << USAGE << std::endl;
    return 0;
  }

  global_debug = debug = args["--debug"].asBool();
  verbose              = args["--verbose"].asBool();

  std::string db_uri = args["--db_uri"].asString();
  std::string q_uri  = args["--q_uri"].asString();
  std::string g_uri  = args["--g_uri"] ? args["--g_uri"].asString() : "";

  std::cout << "# Using " << args["--order"].asString() << std::endl;

  size_t k        = args["--k"].asLong();
  size_t nthreads = args["--nthreads"].asLong();
  size_t nqueries = args["--nqueries"].asLong();
  size_t block    = args["--block"].asLong();

  auto nth = args["--nth"].asBool();
  auto validate = args["--validate"].asBool();

  if (nthreads == 0) {
    nthreads = std::thread::hardware_concurrency();
  }

  ms_timer                 load_time { "Load database, query, and ground truth arrays" };
  tdbColMajorMatrix<float> db(db_uri, block);     // blocked
  tdbColMajorMatrix<float> q(q_uri, nqueries);    // just a slice

  auto g = g_uri.empty() ? ColMajorMatrix<int>(0, 0) : tdbColMajorMatrix<int>(g_uri);
  load_time.stop();
  std::cout << load_time << std::endl;

  auto top_k = [&]() {

  // @todo reimplement these
#if 0
  if (args["--order"].asString() == "vq") {
    if (verbose) {
      std::cout << "# Using vq loop nesting for query" << std::endl;
      if (nth) {
        std::cout << "# Using nth_element selection" << std::endl;
      }
    }
    return query_vq(db, q, k, nth, nthreads);
  } else if (args["--order"].asString() == "qv") {
    if (verbose) {
      std::cout << "# Using qv nesting for query" << std::endl;
      if (nth) {
        std::cout << "# Using nth element selection" << std::endl;
      }
    }
    return query_qv(db, q, k, nth, nthreads);
  } else
#endif
    if (args["--order"].asString() == "gemm") {
      if (block != 0) {
        std::cout << "# Using blocked gemm for query" << std::endl;
        return blocked_gemm_query(db, q, k, nth, nthreads);
      } else {
        std::cout << "# Using gemm for query" << std::endl;
        return gemm_query(db, q, k, nth, nthreads);
      }
    }
    return ColMajorMatrix<size_t>(0, 0);
  }();

  if (!g_uri.empty() && validate) {
    validate_top_k(top_k, g);
  }

  if (args["--output_uri"]) {
    write_matrix(top_k, args["--output_uri"].asString());
  }

  if (args["--log"]) {

    auto program_args = args_log(args);
    auto config       = config_log(argv[0]);

    json log_log = {
      {"Config",   config       },
      { "Args",    program_args },
      { "Times",   get_timings()}
    };

    if (args["--log"].asString() == "-") {
      std::cout << log_log.dump(2) << std::endl;
    } else {
      std::ofstream outfile(args["--log"].asString(), std::ios_base::app);
      outfile << log_log.dump(2) << std::endl;
    }
  }
}
