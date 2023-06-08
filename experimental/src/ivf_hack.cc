/**
 * @file   ivf_hack.cc
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
 * Driver program for experimenting with algorithms and data structures
 * for kmeans.
 *
 * The program can operate in one of two modes.
 *
 * 1) It takes a set of feature vectors and a set of centroid
 * vectors and creates a new set of feature vectors partitioned according
 * to their nearest centroid.  I then writes the partitioned vectors,
 * the partition index and a vector of the original vector IDs to disk.
 *
 * 2) Given a query vector, it finds the set of nearest centroids and
 * then searches the partitions corresponding to those centroids
 * for the nearest neighbors.
 *
 * @todo This should probably be broken into smaller functions.
 * @todo We need to add a good dose of parallelism.
 * @todo We need to add accuracy reporting as well as QPS.
 */

#include <algorithm>
#include <cmath>
#include <filesystem>
// #include <format>     // Not suppored by Apple clang
// #include <execution>  // Not suppored by Apple clang

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <docopt.h>

#include "array_types.h"
#include "config.h"
#include "defs.h"
#include "ivf_query.h"
#include "linalg.h"
#include "stats.h"
#include "utils/logging.h"
#include "utils/timer.h"
#include "utils/utils.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

bool global_verbose = false;
bool global_debug = false;
std::string global_region;
double global_time_of_interest{0};

static constexpr const char USAGE[] =
    R"(ivf_hack: demo hack feature vector search with kmeans index.
Usage:
    ivf_hack (-h | --help)
    ivf_hack --db_uri URI --centroids_uri URI --index_uri URI --parts_uri URI --ids_uri URI [--alg algo]
            [--output_uri URI] [--query_uri URI] [--groundtruth_uri URI] [--ndb NN] [--nqueries NN] [--blocksize NN]
            [--finite] [--k NN] [--cluster NN] [--nthreads N] [--region REGION] [--nth] [--log FILE] [-d] [-v]

Options:
    -h, --help            show this screen
    --db_uri URI          database URI with feature vectors
    --centroids_uri URI   URI with centroid vectors
    --index_uri URI       URI with the paritioning index
    --parts_uri URI       URI with the partitioned data
    --ids_uri URI         URI with original IDs of vectors
    --alg algo            which algorithm to use for query [default: qv_heap]
    --output_uri URI      URI to store search results
    --query_uri URI       URI storing query vectors
    --groundtruth_uri URI URI storing ground truth vectors
    --nqueries NN         number of query vectors to use (0 = all) [default: 0]
    --ndb NN              number of database vectors to use (0 = all) [default: 0]
    --finite              use finite RAM (out of core) algorithm [default: false]
    --nthreads N          number of threads to use in parallel loops (0 = all) [default: 0]
    --k NN                number of nearest neighbors to search for [default: 10]
    --cluster NN          number of clusters to use [default: 100]
    --blocksize NN        number of vectors to process in a block (0 = all) [default: 0]
    --nth                 use nth_element for top k [default: false]
    --log FILE            log info to FILE (- for stdout)
    --region REGION       AWS S3 region [default: us-east-1]
    -d, --debug           run in debug mode [default: false]
    -v, --verbose         run in verbose mode [default: false]
)";

int main(int argc, char* argv[]) {
  std::vector<std::string> strings(argv + 1, argv + argc);
  auto args = docopt::docopt(USAGE, strings, true);

  auto centroids_uri = args["--centroids_uri"].asString();
  auto db_uri = args["--db_uri"].asString();
  auto ndb = args["--ndb"].asLong();
  auto nthreads = args["--nthreads"].asLong();
  if (nthreads == 0) {
    nthreads = std::thread::hardware_concurrency();
  }
  global_debug = args["--debug"].asBool();
  global_verbose = args["--verbose"].asBool();
  global_region = args["--region"].asString();

  auto part_uri = args["--parts_uri"].asString();
  auto index_uri = args["--index_uri"].asString();
  auto id_uri = args["--ids_uri"].asString();
  size_t nprobe = args["--cluster"].asLong();
  size_t k_nn = args["--k"].asLong();
  auto query_uri = args["--query_uri"] ? args["--query_uri"].asString() : "";
  auto nqueries = (size_t)args["--nqueries"].asLong();
  auto blocksize = (size_t)args["--blocksize"].asLong();
  bool nth = args["--nth"].asBool();
  auto algorithm = args["--alg"].asString();
  bool finite = args["--finite"].asBool();

  if (is_local_array(centroids_uri) &&
      !std::filesystem::exists(centroids_uri)) {
    std::cerr << "Error: centroids URI does not exist: "
              << args["--centroids_uri"] << std::endl;
    return 1;
  }
  auto centroids = tdbColMajorMatrix<centroids_type>(centroids_uri);
  debug_matrix(centroids, "centroids");

  float recall{0.0f};
  json recalls;

  // Find the top k nearest neighbors accelerated by kmeans and do some
  // reporting
  {
    life_timer _("query_time");

    // @todo Encapsulate these arrays in a class
    // auto shuffled_db = tdbColMajorMatrix<shuffled_db_type>(part_uri);
    // auto shuffled_ids = read_vector<shuffled_ids_type>(id_uri);
    // debug_matrix(shuffled_db, "shuffled_db");
    // debug_matrix(shuffled_ids, "shuffled_ids");

    auto indices = read_vector<indices_type>(index_uri);
    debug_matrix(indices, "indices");

    auto q = tdbColMajorMatrix<q_type>(query_uri, nqueries);
    debug_matrix(q, "q");

    auto top_k = [&]() {
      if (finite) {
        return detail::ivf::qv_query_heap_finite_ram(
            part_uri,
            centroids,
            q,
            indices,
            id_uri,
            nprobe,
            k_nn,
            blocksize,
            nth,
            nthreads);
      } else {
        return detail::ivf::qv_query_heap_infinite_ram(
            part_uri,
            centroids,
            q,
            indices,
            id_uri,
            nprobe,
            k_nn,
            nth,
            nthreads);
      }
    }();

    debug_matrix(top_k, "top_k");

    if (args["--groundtruth_uri"]) {
      auto groundtruth_uri = args["--groundtruth_uri"].asString();

      auto groundtruth =
          tdbColMajorMatrix<groundtruth_type>(groundtruth_uri, nqueries);

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
      size_t total_groundtruth = top_k.num_cols() * top_k.num_rows();
      for (size_t i = 0; i < top_k.num_cols(); ++i) {
        std::sort(begin(top_k[i]), end(top_k[i]));
        std::sort(begin(groundtruth[i]), begin(groundtruth[i]) + k_nn);
        debug_matrix(top_k, "top_k");
        debug_slice(top_k, "top_k");
        total_intersected += std::set_intersection(
            begin(top_k[i]),
            end(top_k[i]),
            begin(groundtruth[i]),
            end(groundtruth[i]),
            counter{});
      }

      recall = ((float)total_intersected) / ((float)total_groundtruth);
      std::cout << "# total intersected = " << total_intersected << " of "
                << total_groundtruth << " = "
                << "R@" << k_nn << " of " << recall << std::endl;
    }
  }

  auto timings = get_timings();

  // Quick and dirty way to get query info in summarizable form
  if (true || global_verbose) {
    auto ms = global_time_of_interest;
    auto qps = ((float)nqueries) / ((float)ms / 1000.0);
    std::cout << std::setw(8) << "-|-";
    std::cout << std::setw(8) << algorithm;
    std::cout << std::setw(8) << nqueries;
    std::cout << std::setw(8) << nprobe;
    std::cout << std::setw(8) << k_nn;
    std::cout << std::setw(8) << nthreads;
    std::cout << std::setw(8) << ms;
    std::cout << std::setw(8) << qps;
    std::cout << std::setw(8) << recall;
    std::cout << std::endl;
  }

  if (args["--log"]) {
    auto program_args = args_log(args);
    auto config = config_log(argv[0]);

    json log_log = {
        {"Config", config},
        {"Args", program_args},
        {"Recalls", recalls},
        {"Times", timings}};

    if (args["--log"].asString() == "-") {
      std::cout << log_log.dump(2) << std::endl;
    } else {
      std::ofstream outfile(args["--log"].asString(), std::ios_base::app);
      outfile << log_log.dump(2) << std::endl;
    }
  }
}

// recalls = {}
// i = 1
// while i <= k:
//    recalls[i] = (I[:, :i] == gt[:, :1]).sum() / float(#queries
//    i *= 10
// return (t1 - t0) * 1000.0 / nq, recalls
