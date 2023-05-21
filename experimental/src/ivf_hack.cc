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
 * @todo This should probably be split into two programs.
 * @todo This should probably be broken into smaller functions.
 * @todo We need to add a good dose of parallelism.
 * @todo We need to add accuracy reporting as well as QPS.
 */

#include <algorithm>
#include <cmath>
#include <filesystem>
// #include <format>     // Not suppored by Apple clang
// #include <execution>  // Not suppored by Apple clang
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <docopt.h>

#include "defs.h"
#include "ivf_query.h"
#include "linalg.h"
#include "timer.h"
#include "utils.h"
#include "config.h"

bool global_verbose = false;
bool global_debug = false;
bool global_dryrun = false;
std::string global_region;

static constexpr const char USAGE[] =
    R"(ivf_hack: demo hack feature vector search with kmeans index.
Usage:
    tdb (-h | --help)
    tdb   --db_uri URI --centroids_uri URI --index_uri URI --part_uri URI --id_uri URI
         [--output_uri URI] [--query_uri URI] [--ndb NN] [--nqueries NN]
         [--k NN] [--cluster NN] [--write] [--nthreads N] [--region REGION] [-n] [-d | -v]

Options:
    -h, --help            show this screen
    --db_uri URI          database URI with feature vectors
    --centroids_uri URI   URI with centroid vectors
    --index_uri URI       URI with the paritioning index
    --part_uri URI        URI with the partitioned data
    --id_uri URI          URI with original IDs of vectors
    --output_uri URI      URI to store search results
    --query_uri URI       URI storing query vectors
    --nqueries NN         number of query vectors to use (0 = all) [default: 1]
    --ndb NN              number of database vectors to use (0 = all) [default: 0]
    --write               write the index to disk [default: false]
    --nthreads N          number of threads to use in parallel loops (0 = all) [default: 0]
    --k NN                number of nearest neighbors to search for [default: 10]
    --cluster NN          number of clusters to use [default: 100]
    --region REGION       AWS S3 region [default: us-east-1]
    -n, --dryrun          perform a dry run (no writes) [default: false]
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
  global_dryrun = args["--dryrun"].asBool();
  global_region = args["--region"].asString();

  if (global_debug) {
    std::cout << "# " << argv[0] << " built at " << CURRENT_DATETIME << "\n";
    std::cout << "# Git branch: " << GIT_BRANCH << "\n";
    std::cout << "# Built from source in " << CMAKE_SOURCE_DIR << "\n";
    auto&& [major, minor, patch] = tiledb::version();
    std::cout << "# TileDB version: " << major << "." << minor << "." << patch
              << std::endl;
  }

  auto db = tdbMatrix<float, Kokkos::layout_left>(db_uri, ndb);
  debug_matrix(db, "db");

  if (is_local_array(centroids_uri) &&
      !std::filesystem::exists(centroids_uri)) {
    std::cerr << "Error: centroids URI does not exist: "
              << args["--centroids_uri"] << std::endl;
    return 1;
  }
  auto centroids = tdbMatrix<float, Kokkos::layout_left>(centroids_uri);
  debug_matrix(centroids, "centroids");

  if (is_local_array(centroids_uri) && !std::filesystem::exists(db_uri)) {
    std::cerr << "Error: db URI does not exist: " << args["--centroids_uri"]
              << std::endl;
    return 1;
  }

  if (args["--write"].asBool()) {
    auto parts = gemm_partition(centroids, db, nthreads);
    debug_matrix(parts, "parts");
    //  auto parts = qv_partition(centroids, db, nthreads);

    // read centroids
    // for each vector in the dataset, find nearest centroid
    // [ D, I ] = query_gemm(centroids, data, top_k, nthreads);
    {
      life_timer _{"shuffling data"};
      std::vector<size_t> degrees(centroids.num_cols());
      std::vector<size_t> indices(centroids.num_cols() + 1);
      for (size_t i = 0; i < db.num_cols(); ++i) {
        auto j = parts[i];
        ++degrees[j];
      }
      indices[0] = 0;
      std::inclusive_scan(begin(degrees), end(degrees), begin(indices) + 1);
      std::vector<size_t> check(indices.size());
      std::copy(begin(indices), end(indices), begin(check));

      debug_matrix(degrees, "degrees");
      debug_matrix(indices, "indices");

      // Some variables for debugging
      // @todo remove these once we are confident in the code
      auto mis = std::max_element(begin(indices), end(indices));
      auto a = std::distance(begin(indices), mis);
      auto b = std::distance(mis, end(indices));
      auto misx = *mis;

      // Array for storing the shuffled data
      auto shuffled_db = ColMajorMatrix<float>{db.num_rows(), db.num_cols()};
      std::vector shuffled_ids = std::vector<uint64_t>(db.num_cols());
      std::iota(begin(shuffled_ids), end(shuffled_ids), 0);

      debug_matrix(shuffled_db, "shuffled_db");
      debug_matrix(shuffled_ids, "shuffled_ids");

      // @todo parallelize
      // Unfortunately this variant of the algorithm is not parallelizable.
      // The other approach involves doing parallel sort on the indices,
      // which will group them nicely -- but a distributed parallel sort may
      // be difficult to implement.  Even this algorithm is not trivial to
      // parallelize, because of the random access to the indices array.
      for (size_t i = 0; i < db.num_cols(); ++i) {
        size_t bin = parts[i];
        size_t ibin = indices[bin];

        shuffled_ids[ibin] = i;

        assert(ibin < shuffled_db.num_cols());
        for (size_t j = 0; j < db.num_rows(); ++j) {
          shuffled_db(j, ibin) = db(j, i);
        }
        ++indices[bin];
      }

      std::shift_right(begin(indices), end(indices), 1);
      indices[0] = 0;

      // A check for debugging
      auto x = std::equal(begin(indices), end(indices), begin(check));

      // Write out the arrays

      // @todo Better checking for existing files and reporting errors
      auto part_uri = args["--part_uri"].asString();
      auto index_uri = args["--index_uri"].asString();
      auto id_uri = args["--id_uri"].asString();

      if (!global_dryrun) {
        if (part_uri != "") {
          if (is_local_array(part_uri) && std::filesystem::exists(part_uri)) {
            // Apple clang does not support std::format yet
            // std::cerr << std::format("Error: URI {} already exists: " ,
            // part_uri) << std::endl;
            std::cerr << "Error: URI " << part_uri
                      << " already exists: " << std::endl;
            std::cerr << "This is a dangerous operation, so we will not "
                         "overwrite the file."
                      << std::endl;
            std::cerr << "Please delete the file manually and try again."
                      << std::endl;
            return 1;
            // Too dangerous to have this ability
            // std::filesystem::remove_all(part_uri);
          }
          write_matrix(shuffled_db, part_uri);
        }
        if (index_uri != "") {
          if (is_local_array(index_uri) && std::filesystem::exists(index_uri)) {
            // std::filesystem::remove(index_uri);
            std::cerr << "Error: URI " << index_uri
                      << " already exists: " << std::endl;
            std::cerr << "This is a dangerous operation, so we will not "
                         "overwrite the file."
                      << std::endl;
            std::cerr << "Please delete the file manually and try again."
                      << std::endl;
            return 1;
          }
          write_vector(indices, index_uri);
        }
        if (id_uri != "") {
          if (is_local_array(id_uri) && std::filesystem::exists(id_uri)) {
            std::cerr << "Error: URI " << id_uri
                      << " already exists: " << std::endl;
            std::cerr << "This is a dangerous operation, so we will not "
                         "overwrite the file."
                      << std::endl;
            std::cerr << "Please delete the file manually and try again."
                      << std::endl;
            return 1;
            // std::filesystem::remove(id_uri);
          }
          write_vector(shuffled_ids, id_uri);
        }
      }
    }
  } else {
    // Read all of the precomputed data
    auto part_uri = args["--part_uri"].asString();
    auto index_uri = args["--index_uri"].asString();
    auto id_uri = args["--id_uri"].asString();
    size_t nprobe = args["--cluster"].asLong();
    size_t k_nn = args["--k"].asLong();
    auto query_uri = args["--query_uri"] ? args["--query_uri"].asString() : "";
    auto nqueries = (size_t)args["--nqueries"].asLong();

    // Function for finding the top k nearest neighbors accelerated by kmeans
    // @todo Move this to a self-contained function
    {
      life_timer _("query_time");

      auto shuffled_db = tdbMatrix<float, Kokkos::layout_left>(part_uri);
      // auto indices = tdbMatrix<size_t, Kokkos::layout_left>(index_uri);
      auto indices = read_vector<size_t>(index_uri);
      auto shuffled_ids = read_vector<uint64_t>(id_uri);

      debug_matrix(shuffled_db, "shuffled_db");
      debug_matrix(indices, "indices");
      debug_matrix(shuffled_ids, "shuffled_ids");

      // Some variables for debugging
      auto mv = *std::max_element(begin(shuffled_ids), end(shuffled_ids));

      auto q = [&]() -> ColMajorMatrix<float> {
        if (query_uri != "") {
          auto qq = tdbMatrix<float, Kokkos::layout_left>(query_uri, nqueries);
          return qq;
        } else {
          auto qq = ColMajorMatrix<float>{centroids.num_rows(), nqueries};
          for (size_t i = 0; i < centroids.num_rows(); ++i) {
            qq(i, 0) = db(i, 0);
          }
          return qq;
        }
      }();
      debug_matrix(q, "q");

      // What should be returned here?  Maybe a pair with the ids and scores?
      auto&& [kmeans_ids, all_ids] = kmeans_query(
          db,
          shuffled_db,
          centroids,
          q,
          indices,
          shuffled_ids,
          nprobe,
          k_nn,
          nthreads);
      debug_matrix(kmeans_ids, "kmeans_ids");
      // Once this is a function, simply return kmeans_ids
      // For now, print the results to std::cout
      // @todo also get scores
      // @todo add an output_uri argument
      for (size_t i = 0; i < kmeans_ids.num_rows(); ++i) {
        std::cout << all_ids[kmeans_ids(i, 0)] << ": " << std::endl;
      }
    }
  }
}
