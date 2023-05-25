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
 */

if (args["--write"].asBool()) {
  auto parts = blocked_gemm_partition(centroids, db, nthreads);
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
}


--write               write the index to disk [default: false]
    -n, --dryrun          perform a dry run (no writes) [default: false]
