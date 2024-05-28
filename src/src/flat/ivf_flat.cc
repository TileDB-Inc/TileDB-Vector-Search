/**
 * @file   ivf_flat.cc
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

#include "docopt.h"

#include "config.h"
#include "ivf_query.h"
#include "linalg.h"
#include "stats.h"
#include "utils/logging.h"
#include "utils/timer.h"
#include "utils/utils.h"

bool enable_stats = false;
std::vector<json> core_stats;

#include <cstdint>

/**
 * Specify some types for the demo.  For now the types associated with the
 * vector db to be queried are hard-coded.
 */
using score_type = float;
using groundtruth_type = int32_t;
using centroids_type = float;
using indices_type = uint64_t;

static constexpr const char USAGE[] =
    R"(ivf_flat: demo CLI program for performing feature vector search with kmeans index.
Usage:
    ivf_flat (-h | --help)
    ivf_flat --centroids_uri URI --parts_uri URI (--index_uri URI | --sizes_uri URI)
             --ids_uri URI --query_uri URI [--groundtruth_uri URI] [--output_uri URI]
            [--k NN][--nprobe NN] [--nqueries NN] [--alg ALGO] [--infinite] [--finite] [--blocksize NN]
            [--nthreads NN] [--ppt NN] [--vpt NN] [--nodes NN] [--region REGION] [--stats] [--log FILE] [-d] [-v]

Options:
    -h, --help            show this screen
    --centroids_uri URI   URI with centroid vectors
    --index_uri URI       URI with the paritioning index
    --sizes_uri URI       URI with the parition sizes
    --parts_uri URI       URI with the partitioned data
    --ids_uri URI         URI with original IDs of vectors
    --query_uri URI       URI storing query vectors
    --groundtruth_uri URI URI storing ground truth vectors
    --output_uri URI      URI to store search results
    --ftype TYPE          data type of feature vectors [default: float]
    --idtype TYPE         data type of ids [default: uint64]
    --k NN                number of nearest neighbors to search for [default: 10]
    --nprobe NN           number of centroid partitions to use [default: 100]
    --nqueries NN         number of query vectors to use (0 = all) [default: 0]
    --alg ALGO            which algorithm to use for query [default: qv_heap]
    --infinite            use infinite RAM algorithm [default: false]
    --finite              (legacy) use finite RAM (out of core) algorithm [default: true]
    --blocksize NN        number of vectors to process in an out of core block (0 = all) [default: 0]
    --nthreads NN         number of threads to use (0 = hardware concurrency) [default: 0]
    --ppt NN              minimum number of partitions to assign to a thread (0 = no min) [default: 0]
    --vpt NN              minimum number of vectors to assign to a thread (0 = no min) [default: 0]
    --nodes NN            number of nodes to use for (emulated) distributed query [default: 1]
    --region REGION       AWS S3 region [default: us-east-1]
    --log FILE            log info to FILE (- for stdout)
    --stats               log TileDB stats [default: false]
    -d, --debug           run in debug mode [default: false]
    -v, --verbose         run in verbose mode [default: false]
)";

int main(int argc, char* argv[]) {
  log_timer _(tdb_func__ + std::string(" all inclusive query time"));

  std::vector<std::string> strings(argv + 1, argv + argc);
  auto args = docopt::docopt(USAGE, strings, true);

  auto centroids_uri = args["--centroids_uri"].asString();
  auto nthreads = args["--nthreads"].asLong();
  if (nthreads == 0) {
    nthreads = std::thread::hardware_concurrency();
  }
  enable_stats = args["--stats"].asBool();

  auto part_uri = args["--parts_uri"].asString();

  std::string index_uri;
  bool size_index{false};
  if (args["--index_uri"]) {
    if (args["--sizes_uri"]) {
      std::cerr << "Cannot specify both --index_uri and --sizes_uri\n";
      return 1;
    }
    index_uri = args["--index_uri"].asString();
  }
  if (args["--sizes_uri"]) {
    index_uri = args["--sizes_uri"].asString();
    size_index = true;
  }
  if (index_uri.empty()) {
    std::cerr << "Must specify either --index_uri or --sizes_uri\n";
    return 1;
  }

  auto id_uri = args["--ids_uri"].asString();
  size_t nprobe = args["--nprobe"].asLong();
  size_t k_nn = args["--k"].asLong();
  auto query_uri = args["--query_uri"] ? args["--query_uri"].asString() : "";
  auto nqueries = (size_t)args["--nqueries"].asLong();
  auto blocksize = (size_t)args["--blocksize"].asLong();
  auto num_nodes = (size_t)args["--nodes"].asLong();
  auto ppt = args["--ppt"].asLong();
  auto vpt = args["--vpt"].asLong();
  auto algorithm = args["--alg"].asString();
  // bool finite = args["--finite"].asBool();
  bool finite = !(args["--infinite"].asBool());

  float recall{0.0f};
  tiledb::Context ctx;

  if (is_local_array(centroids_uri) &&
      !std::filesystem::exists(centroids_uri)) {
    std::cerr << "Error: centroids URI does not exist: "
              << args["--centroids_uri"] << std::endl;
    return 1;
  }

  auto centroids = tdbColMajorMatrix<centroids_type>(ctx, centroids_uri);
  centroids.load();
  debug_matrix(centroids, "centroids");

  // Find the top k nearest neighbors accelerated by kmeans and do some
  // reporting

  // @todo Encapsulate these arrays into a single ivf_index class (WIP)
  // auto shuffled_db = tdbColMajorMatrix<feature_type>(part_uri);
  // auto shuffled_ids = read_vector<id_type>(id_uri);
  // debug_matrix(shuffled_db, "shuffled_db");
  // debug_matrix(shuffled_ids, "shuffled_ids");

  auto indices = read_vector<indices_type>(ctx, index_uri);
  if (size_index) {
    indices = sizes_to_indices(indices);
  }
  debug_matrix(indices, "indices");

  auto run_query = [&]<class feature_type, class id_type>() {
    auto q = tdbColMajorMatrix<feature_type>(ctx, query_uri, nqueries);
    q.load();
    debug_matrix(q, "q");

    auto&& [top_k_scores, top_k] = [&]() {
      if (algorithm == "reg") {
        if (finite) {
          return detail::ivf::
              nuv_query_heap_finite_ram_reg_blocked<feature_type, id_type>(
                  ctx,
                  part_uri,
                  centroids,
                  q,
                  indices,
                  id_uri,
                  nprobe,
                  k_nn,
                  blocksize,
                  nthreads);
        } else {
          return detail::ivf::
              nuv_query_heap_infinite_ram_reg_blocked<feature_type, id_type>(
                  ctx,
                  part_uri,
                  centroids,
                  q,
                  indices,
                  id_uri,
                  nprobe,
                  k_nn,
                  nthreads);
        }
      } else if (algorithm == "final" || algorithm == "fin") {
        if (finite) {
          return detail::ivf::query_finite_ram<feature_type, id_type>(
              ctx,
              part_uri,
              centroids,
              q,
              indices,
              id_uri,
              nprobe,
              k_nn,
              blocksize,
              nthreads,
              ppt);
        } else {
          return detail::ivf::query_infinite_ram<feature_type, id_type>(
              ctx,
              part_uri,
              centroids,
              q,
              indices,
              id_uri,
              nprobe,
              k_nn,
              nthreads);
        }
      } else if (algorithm == "nuv_heap" || algorithm == "nuv") {
        if (finite) {
          return detail::ivf::nuv_query_heap_finite_ram<feature_type, id_type>(
              ctx,
              part_uri,
              centroids,
              q,
              indices,
              id_uri,
              nprobe,
              k_nn,
              blocksize,
              nthreads);
        } else {
          return detail::ivf::
              nuv_query_heap_infinite_ram<feature_type, id_type>(
                  ctx,
                  part_uri,
                  centroids,
                  q,
                  indices,
                  id_uri,
                  nprobe,
                  k_nn,
                  nthreads);
        }
      } else if (algorithm == "qv_heap" || algorithm == "qv") {
        if (finite) {
          return detail::ivf::qv_query_heap_finite_ram<feature_type, id_type>(
              ctx,
              part_uri,
              centroids,
              q,
              indices,
              id_uri,
              nprobe,
              k_nn,
              blocksize,
              nthreads);
        } else {
          return detail::ivf::qv_query_heap_infinite_ram<feature_type, id_type>(
              ctx,
              part_uri,
              centroids,
              q,
              indices,
              id_uri,
              nprobe,
              k_nn,
              nthreads);
        }
      } else if (algorithm == "vq_heap" || algorithm == "vq") {
        if (finite) {
          return detail::ivf::vq_query_finite_ram<feature_type, id_type>(
              ctx,
              part_uri,
              centroids,
              q,
              indices,
              id_uri,
              nprobe,
              k_nn,
              blocksize,
              nthreads);
        } else {
          return detail::ivf::vq_query_infinite_ram<feature_type, id_type>(
              ctx,
              part_uri,
              centroids,
              q,
              indices,
              id_uri,
              nprobe,
              k_nn,
              nthreads);
        }
#if 0
      else {
        return detail::ivf::
            vq_query_heap_infinite_ram<feature_type, id_type>(
                ctx,
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
#endif
      } else if (algorithm == "vq_heap_2" || algorithm == "vq2") {
        if (finite) {
          return detail::ivf::vq_query_finite_ram_2<feature_type, id_type>(
              ctx,
              part_uri,
              centroids,
              q,
              indices,
              id_uri,
              nprobe,
              k_nn,
              blocksize,
              nthreads);
        } else {
          {
            return detail::ivf::vq_query_infinite_ram_2<feature_type, id_type>(
                ctx,
                part_uri,
                centroids,
                q,
                indices,
                id_uri,
                nprobe,
                k_nn,
                nthreads);
          }
        }
      } else if (algorithm == "dist_nuv_heap" || algorithm == "dist") {
        return detail::ivf::dist_qv_finite_ram<feature_type, id_type>(
            ctx,
            part_uri,
            centroids,
            q,
            indices,
            id_uri,
            nprobe,
            k_nn,
            blocksize,
            nthreads,
            num_nodes);
      }
      throw std::runtime_error(
          "incorrect or unset algorithm type: " + algorithm);
    }();

    debug_matrix(top_k, "top_k");

    // @todo encapsulate as a function
    if (args["--groundtruth_uri"]) {
      auto groundtruth_uri = args["--groundtruth_uri"].asString();

      auto groundtruth =
          tdbColMajorMatrix<groundtruth_type>(ctx, groundtruth_uri, nqueries);
      groundtruth.load();

      if (false) {
        std::cout << std::endl;

        debug_matrix(groundtruth, "groundtruth");

        std::cout << std::endl;
        debug_matrix(top_k, "top_k");
        debug_matrix(top_k, "top_k");

        std::cout << std::endl;
      }

      size_t total_intersected{0};
      size_t total_groundtruth = top_k.num_cols() * top_k.num_rows();

      for (size_t i = 0; i < top_k.num_cols(); ++i) {
        std::sort(begin(top_k[i]), end(top_k[i]));
        std::sort(begin(groundtruth[i]), begin(groundtruth[i]) + k_nn);
        total_intersected += std::set_intersection(
            begin(top_k[i]),
            end(top_k[i]),
            begin(groundtruth[i]),
            end(groundtruth[i]),
            assignment_counter{});
      }

      recall = ((float)total_intersected) / ((float)total_groundtruth);
      std::cout << "# total intersected = " << total_intersected << " of "
                << total_groundtruth << " = "
                << "R@" << k_nn << " of " << recall << std::endl;
    }

    if (args["--output_uri"]) {
      write_matrix(ctx, top_k, args["--output_uri"].asString());
    }

    _.stop();

    if (args["--log"]) {
      dump_logs(
          args["--log"].asString(),
          algorithm,
          (nqueries == 0 ? num_vectors(q) : nqueries),
          nprobe,
          k_nn,
          nthreads,
          recall);
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
