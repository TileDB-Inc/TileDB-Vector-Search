//
// Created by Andrew Lumsdaine on 4/12/23.
//

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <docopt.h>

#include "defs.h"
#include "sift_db.h"

bool verbose = false;
bool debug = false;

static constexpr const char USAGE[] =
        R"(flat: feature vector search with flat index.
  Usage:
      tdb (-h | --help)
      tdb (--db_file FILE | --db_uri URI) (--q_file FILE | --q_uri URI) (--g_file FILE | --g_uri URI) [--dim D] [--k NN] [--L2 | --cosine] [--order ORDER] [-d | -v]

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
    sift_db<float> db(db_file, args["--dim"].asLong());
    sift_db<float> q(q_file, args["--dim"].asLong());
    sift_db<int> g(g_file, 100);

    /** Matrix-matrix multiplication
     *
     * @todo Use BLAS DGEMM
     */
    size_t k = args["--k"].asLong();
    std::cout << args["--order"].asString() << std::endl;
    if (args["--order"].asString() == "vq") {
      if (verbose) {
        std::cout << "Using vq ordering" << std::endl;
      }
      std::vector<std::vector<size_t>> top_k(q.size(), std::vector<size_t>(k, 0UL));
      
      /**
       * For each query
       */
      for (size_t j = 0; j < q.size(); ++j) {
        std::vector<size_t> index(size(db));
        std::iota(begin(index), end(index), 0);
        std::vector<float> scores(size(db));

        /**
         * Compare with each database vector
         */
        for (size_t i = 0; i < size(db); ++i) {
          scores[i] = L2(q[j], db[i]);
        }
        std::nth_element(begin(index), begin(index) + k, end(index), [&](auto a, auto b) {
          return scores[a] < scores[b];
        });
        std::copy(begin(index), begin(index) + k, begin(top_k[j]));

        std::sort(begin(top_k[j]), end(top_k[j]), [&](auto a, auto b) {
          return scores[a] < scores[b];
        });

        if (!std::equal(top_k[j].begin(), top_k[j].end(), g[j].begin(), [&](auto a, auto b) {
              return scores[a] == scores[b];
            })) {
          std::cout << "Query " << j << " is incorrect" << std::endl;
          for (size_t i = 0; i < k; ++i) {
            std::cout << "  (" << top_k[j][i] << " " << scores[top_k[j][i]] << ") ";
          }
          std::cout << std::endl;
          for (size_t i = 0; i < k; ++i) {
            std::cout << "  (" << g[j][i] << " " << scores[g[j][i]] << ") ";
          }
          std::cout << std::endl;
          std::cout << std::endl;
        }
      }
    } else if (args["--order"].asString() == "qv") {
      if (verbose) {
        std::cout << "Using qv ordering" << std::endl;
      }
      std::vector<std::vector<float>> scores(q.size(), std::vector<float>(size(db), 0.0f));
      std::vector<std::vector<size_t>> top_k(q.size(), std::vector<size_t>(k, 0UL));

      /**
       * For each database vector
       */
      for (size_t i = 0; i < size(db); ++i) {
        /**
         * Compare with each query
         */
        for (size_t j = 0; j < q.size(); ++j) {
          scores[j][i] = L2(q[j], db[i]);
        }
      }
      for (size_t j = 0; j < q.size(); ++j) {
        std::vector<size_t> index(size(db));
        std::iota(begin(index), end(index), 0);

        std::nth_element(begin(index), begin(index) + k, end(index), [&](auto a, auto b) {
          return scores[j][a] < scores[j][b];
        });
        std::copy(begin(index), begin(index) + k, top_k[j].begin());

        std::sort(top_k[j].begin(), top_k[j].end(), [&](auto a, auto b) {
          return scores[j][a] < scores[j][b];
        });

        if (!std::equal(top_k[j].begin(), top_k[j].end(), g[j].begin(), [&](auto a, auto b) {
              return scores[j][a] == scores[j][b];
            })) {
          std::cout << "Query " << j << " is incorrect" << std::endl;
          for (size_t i = 0; i < k; ++i) {
            std::cout << "  (" << top_k[j][i] << " " << scores[j][top_k[j][i]] << ") ";
          }
          std::cout << std::endl;
          for (size_t i = 0; i < k; ++i) {
            std::cout << "  (" << g[j][i] << " " << scores[j][g[j][i]] << ") ";
          }
          std::cout << std::endl;
          std::cout << std::endl;
        }
      }

    } else {
      std::cout << "Unknown ordering: " << args["--order"].asString() << std::endl;
      return 1;
    }
  }
}