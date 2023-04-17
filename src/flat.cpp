//
// Created by Andrew Lumsdaine on 4/12/23.
//

#include <algorithm>
#include <cmath>
// #include <execution>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <docopt.h>

#include "defs.h"
#include "query.h"
#include "sift_db.h"
#include "timer.h"



bool verbose = false;
bool debug = false;

static constexpr const char USAGE[] =
        R"(flat: feature vector search with flat index.
  Usage:
      tdb (-h | --help)
      tdb (--db_file FILE | --db_uri URI) (--q_file FILE | --q_uri URI) (--g_file FILE | --g_uri URI) [--dim D] [--k NN] [--L2 | --cosine] [--order ORDER] [--hardway] [-d | -v]

  Options:
      -h, --help            show this screen
      --db_file FILE        database file with feature vectors
      --db_uri URI          database URI with feature vectors
      --q_file FILE         query file with feature vectors to search for
      --q_uri URI           query URI with feature vectors to search for
      --g_file FILE         ground truth file
      --g_uri URI           ground true URI
      --dim D               dimension of feature vectors [default: 128]
      --k NN                number of nearest neighbors to find [default: 10]
      --L2                  use L2 distance (Euclidean)
      --cosine              use cosine distance [default]
      --order ORDER         which ordering to do comparisons [default: qv]
      --hardway             use hard way to compute distances [default: false]
      -d, --debug           run in debug mode [default: false]
      -v, --verbose         run in verbose mode [default: false]
)";

int main(int argc, char *argv[]) {
  std::vector<std::string> strings(argv + 1, argv + argc);
  auto args = docopt::docopt(USAGE, strings, true);

  if (args["--help"].asBool()) {
    std::cout << USAGE << std::endl;
    return 0;
  }

  debug = args["--debug"].asBool();
  verbose = args["--verbose"].asBool();
  auto hardway = args["--hardway"].asBool();

  std::string db_file{};
  std::string db_uri{};
  if (args["--db_file"]) {
    db_file = args["--db_file"].asString();
  } else if (args["--db_uri"]) {
    db_uri = args["--db_uri"].asString();
  } else {
    std::cout << "Must specify either --db_file or --db_uri" << std::endl;
    return 1;
  }

  std::string q_file{};
  std::string q_uri{};
  if (args["--q_file"]) {
    q_file = args["--q_file"].asString();
  } else if (args["--q_uri"]) {
    q_uri = args["--q_uri"].asString();
  } else {
    std::cout << "Must specify either --q_file or --q_uri" << std::endl;
    return 1;
  }

  std::string g_file{};
  std::string g_uri{};
  if (args["--g_file"]) {
    g_file = args["--g_file"].asString();
  } else if (args["--g_uri"]) {
    g_uri = args["--g_uri"].asString();
  } else {
    std::cout << "Must specify either --g_file or --q_uri" << std::endl;
    return 1;
  }

  if (!db_file.empty() && !q_file.empty() && !g_file.empty()) {
    if (db_file == q_file) {
      std::cout << "db_file and q_file must be different" << std::endl;
      return 1;
    }
    size_t dimension = args["--dim"].asLong();

    ms_timer load_time{"Load database, query, and ground truth"};
    sift_db<float> db(db_file, dimension);
    sift_db<float> q(q_file, dimension);
    sift_db<int> g(g_file, 100);
    load_time.stop();
    std::cout << load_time << std::endl;

    assert(size(db[0]) == dimension);

    size_t k = args["--k"].asLong();
    std::vector<std::vector<int>> top_k(size(q), std::vector<int>(k, 0));

    std::cout << "Using " << args["--order"].asString() << std::endl;

    /**
     * vq: for each vector in the database, compare with each query vector
     */
    if (args["--order"].asString() == "vq") {
      if (verbose) {
        std::cout << "Using vq loop nesting for query" << std::endl;
        if (hardway) {
          std::cout << "Doing it the hard way" << std::endl;
        }
      }
      query_vq(db, q, g, top_k, k, hardway);
    } else if (args["--order"].asString() == "qv") {
      if (verbose) {
        std::cout << "Using qv nesting for query" << std::endl;
        if (hardway) {
          std::cout << "Doing it the hard way" << std::endl;
        }
      }
      query_qv(db, q, g, top_k, k, hardway);
    } else if (args["--order"].asString() == "gemm") {
      if (verbose) {
        std::cout << "Using gemm for query" << std::endl;
      }
      query_gemm(db, q, g, top_k, k, hardway);
    } else {
      std::cout << "Unknown ordering: " << args["--order"].asString() << std::endl;
      return 1;
    }
  }
}
