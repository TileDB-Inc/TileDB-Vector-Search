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
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <docopt.h>

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

static constexpr const char USAGE[] =
    R"(ivf_hack: demo hack feature vector search with kmeans index.
Usage:
    ivf_hack (-h | --help)
    ivf_hack --db_uri URI --centroids_uri URI --index_uri URI --parts_uri URI --ids_uri URI
            [--output_uri URI] [--query_uri URI] [--groundtruth_uri URI] [--ndb NN] [--nqueries NN] [--blocksize NN]
            [--k NN] [--cluster NN] [--nthreads N] [--region REGION] [--nth] [--log FILE] [-d] [-v]

Options:
    -h, --help            show this screen
    --db_uri URI          database URI with feature vectors
    --centroids_uri URI   URI with centroid vectors
    --index_uri URI       URI with the paritioning index
    --parts_uri URI       URI with the partitioned data
    --ids_uri URI         URI with original IDs of vectors
    --output_uri URI      URI to store search results
    --query_uri URI       URI storing query vectors
    --groundtruth_uri URI URI storing ground truth vectors
    --nqueries NN         number of query vectors to use (0 = all) [default: 0]
    --ndb NN              number of database vectors to use (0 = all) [default: 0]
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


using db_type = uint8_t;
using shuffled_db_type = uint8_t;
using shuffled_ids_type = uint64_t;
using indices_type = uint64_t;
using centroids_type = float;
using q_type = uint8_t;
using groundtruth_type = int32_t;

using original_ids_type = uint64_t;

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
  bool nth = args["--nth"].asBool();

  auto db = tdbColMajorMatrix<db_type>(db_uri, ndb);
  debug_matrix(db, "db");

  if (is_local_array(centroids_uri) &&
      !std::filesystem::exists(centroids_uri)) {
    std::cerr << "Error: centroids URI does not exist: "
              << args["--centroids_uri"] << std::endl;
    return 1;
  }
  auto centroids = tdbColMajorMatrix<centroids_type>(centroids_uri);
  debug_matrix(centroids, "centroids");

  if (is_local_array(centroids_uri) && !std::filesystem::exists(db_uri)) {
    std::cerr << "Error: db URI does not exist: " << args["--centroids_uri"]
              << std::endl;
    return 1;
  }
  json recalls;

  // Function for finding the top k nearest neighbors accelerated by kmeans
  // @todo Move this to a self-contained function
  {
    life_timer _("query_time");

    // auto shuffled_db = tdbColMajorMatrix<shuffled_db_type>(part_uri);
    // auto indices = tdbMatrix<size_t, Kokkos::layout_left>(index_uri);
    auto indices = read_vector<indices_type>(index_uri);
    // auto shuffled_ids = read_vector<shuffled_ids_type>(id_uri);

    //debug_matrix(shuffled_db, "shuffled_db");
    debug_matrix(indices, "indices");
    //debug_matrix(shuffled_ids, "shuffled_ids");


    auto q = [&]() -> ColMajorMatrix<q_type> {
      if (query_uri != "") {
        auto qq = tdbColMajorMatrix<q_type>(query_uri, nqueries);
        return qq;
      } else {
        auto qq = ColMajorMatrix<q_type>{centroids.num_rows(), nqueries};
        for (size_t i = 0; i < centroids.num_rows(); ++i) {
          qq(i, 0) = db(i, 0);
        }
        return qq;
      }
    }();
    debug_matrix(q, "q");

    // What should be returned here?  Maybe a pair with the ids and scores?
    auto&& [kmeans_ids, all_ids] = kmeans_query(
        part_uri,
        centroids,
        q,
        indices,
        id_uri,
        nprobe,
        k_nn,
        nth,
        nthreads);
    debug_matrix(kmeans_ids, "kmeans_ids");

    // Once this is a function, simply return kmeans_ids
    // For now, print the results to std::cout
    // @todo also get scores
    // @todo add an output_uri argument

    if (args["--groundtruth_uri"]) {
      auto groundtruth_uri = args["--groundtruth_uri"].asString();
      auto groundtruth = tdbMatrix<groundtruth_type, Kokkos::layout_left>(groundtruth_uri);
      debug_matrix(groundtruth, "groundtruth");

      Matrix<original_ids_type> original_ids(kmeans_ids.num_rows(), kmeans_ids.num_cols());
      for (size_t i = 0; i < kmeans_ids.num_rows(); ++i) {
        for (size_t j = 0; j < kmeans_ids.num_cols(); ++j) {
          original_ids(i, j) = all_ids[kmeans_ids(i, j)];
        }
      }

      debug_slice(groundtruth, "groundtruth");
      debug_slice(original_ids, "original_ids");
      debug_slice(kmeans_ids, "kmeans_ids");

      // kmeans_ids is k by nqueries

      // foreach query
      // get the top k
      //    get the groundtruth
      //    compare
      //       sort
      //       intersect count

      size_t total_query_in_groundtruth{0};
      // for each query
      std::vector<groundtruth_type> comp(kmeans_ids.num_rows());
      for (size_t i = 0; i < kmeans_ids.num_cols(); ++i) {
        for (size_t j = 0; j < kmeans_ids.num_rows(); ++j) {
          comp[j] = all_ids[kmeans_ids(j, i)];
        }
        std::sort(begin(comp), end(comp));
        std::sort(begin(groundtruth[i]), end(groundtruth[i]));

        static constexpr auto lt = [](auto&& x, auto&& y) {
          return std::get<0>(x) < std::get<0>(y);
        };
        total_query_in_groundtruth += std::set_intersection(
            begin(comp),
            end(comp),
            begin(groundtruth[i]),
            end(groundtruth[i]),
            counter{});
      }
      recalls["queries_in_groundtruth"] = total_query_in_groundtruth;
      recalls["total_queries"] = kmeans_ids.num_cols() * kmeans_ids.num_rows();

      if (global_verbose) {
        std::cout << "total_query_in_groundtruth: "
                  << total_query_in_groundtruth;
        std::cout << " / " << kmeans_ids.num_cols() * kmeans_ids.num_rows();
        std::cout << " = "
                  << (((float)total_query_in_groundtruth /
                       ((float)(kmeans_ids.num_cols()) *
                        kmeans_ids.num_rows())))
                  << std::endl;
      }
    }
  }

  if (args["--log"]) {
    auto program_args = args_log(args);
    auto config = config_log(argv[0]);

    json log_log = {
        {"Config", config},
        {"Args", program_args},
        {"Recalls", recalls},
        {"Times", get_timings()}};

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
