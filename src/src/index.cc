/**
 * @file   index.cc
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
 * Driver program to compute kmeans ivf search index.
 *
 * WIP.
 *
 */
#include <filesystem>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <docopt.h>

#include "flat_query.h"
#include "ivf_query.h"
#include "linalg.h"
#include "utils/utils.h"

bool global_verbose = false;
bool global_debug = false;

using namespace detail::flat;

/**
 * Specify some types for the demo.  For now the types associated with the
 * vector db to be queried are hard-coded.
 */
#if 1
using db_type = uint8_t;
#else
using db_type = float;
#endif

using groundtruth_type = int32_t;
using centroids_type = float;
using shuffled_ids_type = uint64_t;
using indices_type = uint64_t;

static constexpr const char USAGE[] =
    R"(index: demo CLI program to create kmeans ivf search index.
Usage:
    index (-h | --help)
    index [--kmeans] [--index]
           --db_uri URI --centroids_uri URI [--index_uri URI] [--parts_uri URI] [--ids_uri URI]
          [--blocksize NN] [--nthreads N] [--nth] [--log FILE] [--force] [--dryrun] [-d] [-v]

Options:
    -h, --help            show this screen
    --db_uri URI          database URI with feature vectors
    --kmeans              run kmeans clustering, computing centroids (default: false)
    --index               create indexing data structures, given centroids (default: false)
    --centroids_uri URI   URI for centroid vectors.  May be input or output.
    --index_uri URI       URI with the paritioning index.  Output.
    --parts_uri URI       URI with the partitioned data.  Output.
    --ids_uri URI         URI with original IDs of vectors.  Output.
    --blocksize NN        number of vectors to process in a block (0 = all) [default: 0]
    --nthreads N          number of threads to use in parallel loops (0 = all) [default: 0]
    --nth                 use nth_element for top k [default: false]
    --log FILE            log info to FILE (- for stdout)
    --force               overwrite output file if it exists [default: false]
    --dryrun              do not write output file [default: false]
    -d, --debug           run in debug mode [default: false]
    -v, --verbose         run in verbose mode [default: false]
)";

int main(int argc, char* argv[]) {
  std::vector<std::string> strings(argv + 1, argv + argc);
  auto args = docopt::docopt(USAGE, strings, true);

  auto centroids_uri = args["--centroids_uri"].asString();
  auto db_uri = args["--db_uri"].asString();

  auto nthreads = args["--nthreads"].asLong();
  if (nthreads == 0) {
    nthreads = std::thread::hardware_concurrency();
  }
  global_debug = args["--debug"].asBool();
  global_verbose = args["--verbose"].asBool();
  bool dryrun = args["--dryrun"].asBool();

  auto parts_uri = args["--parts_uri"] ? args["--parts_uri"].asString() : "";
  auto index_uri = args["--index_uri"] ? args["--index_uri"].asString() : "";
  auto id_uri = args["--ids_uri"] ? args["--ids_uri"].asString() : "";
  bool nth = args["--nth"].asBool();

  tiledb::Context ctx;

  // For now we train the index on the same data as we search

  bool do_kmeans = args["--kmeans"].asBool();
  bool do_index = args["--index"].asBool();

  if (do_kmeans) {
    // compute centroids
  }

  auto centroids = tdbColMajorMatrix<centroids_type>(ctx, centroids_uri);

  auto db = tdbColMajorMatrix<db_type>(ctx, db_uri);

  auto parts = qv_partition(centroids, db, nthreads);
  debug_matrix(parts, "parts");

  {
    scoped_timer _{"shuffling data"};
    std::vector<size_t> degrees(centroids.num_cols());
    std::vector<indices_type> indices(centroids.num_cols() + 1);
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
    auto shuffled_db = ColMajorMatrix<db_type>{db.num_rows(), db.num_cols()};
    std::vector shuffled_ids = std::vector<shuffled_ids_type>(db.num_cols());
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

    if (dryrun) {
      std::cout << "Dry run, not writing output files." << std::endl;
      return 0;
    }
    if (parts_uri != "") {
      if (is_local_array(parts_uri) && std::filesystem::exists(parts_uri)) {
        // Apple clang does not support std::format yet
        // std::cerr << std::format("Error: URI {} already exists: " ,
        // parts_uri) << std::endl;
        std::cerr << "Error: URI " << parts_uri
                  << " already exists: " << std::endl;
        std::cerr << "This is a dangerous operation, so we will not "
                     "overwrite the file."
                  << std::endl;
        std::cerr << "Please delete the file manually and try again."
                  << std::endl;
        return 1;
        // Too dangerous to have this ability
        // std::filesystem::remove_all(parts_uri);
      }
      write_matrix(ctx, shuffled_db, parts_uri);
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
      write_vector(ctx, indices, index_uri);
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
      write_vector(ctx, shuffled_ids, id_uri);
    }
  }
}

#if 0

--write               write the index to disk [default: false]
    -n, --dryrun          perform a dry run (no writes) [default: false]

int main() {
}
#endif
