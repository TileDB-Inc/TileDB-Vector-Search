//
// Created by Andrew Lumsdaine on 4/12/23.
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <docopt.h>

#include "defs.h"
#include "sift_db.h"
#include "timer.h"

// If apple, use Accelerate
#include <Accelerate/Accelerate.h>


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
    size_t dimension = args["--dim"].asLong();


    ms_timer load_time{"Load database, query, and ground truth"};
    sift_db<float> db(db_file, dimension);
    sift_db<float> q(q_file, dimension);
    sift_db<int> g(g_file, 100);
    load_time.stop();
    std::cout << load_time << std::endl;

    size_t k = args["--k"].asLong();
    std::vector<std::vector<size_t>> top_k(size(q), std::vector<size_t>(k, 0UL));

    /** Matrix-matrix multiplication
     */
    std::cout << args["--order"].asString() << std::endl;
    if (args["--order"].asString() == "vq") {
      if (verbose) {
        std::cout << "Using vq ordering" << std::endl;
      }

      std::vector<size_t> index(size(db));
      std::vector<float> scores(size(db));
      /**
       * For each query
       */
      for (size_t j = 0; j < size(q); ++j) {

        /**
         * Compare with each database vector
         */
        for (size_t i = 0; i < size(db); ++i) {
          scores[i] = L2(q[j], db[i]);
        }
        get_top_k(scores, top_k[j], index, k);
        verify_top_k(scores, top_k[j], g[j], k, j);
      }
    } else if (args["--order"].asString() == "qv") {
      if (verbose) {
        std::cout << "Using qv ordering" << std::endl;
      }
      std::vector<std::vector<float>> scores(size(q), std::vector<float>(size(db), 0.0f));

      {
        life_timer _{"L2 comparisons"};

        /**
         * For each database vector
         */
        for (size_t i = 0; i < size(db); ++i) {
          /**
          * Compare with each query
          */
          for (size_t j = 0; j < size(q); ++j) {
            scores[j][i] = L2(q[j], db[i]);
          }
        }
      }

      /**
       * For each query, get indices of top k
       */
      {
        life_timer _{"Get top k"};

        std::vector<size_t> index(size(db));
        for (size_t j = 0; j < size(q); ++j) {
          get_top_k(scores[j], top_k[j], index, k);
        }
      }

      {
        life_timer _{"Checking results"};
        for (size_t j = 0; j < size(q); ++j) {
          verify_top_k(scores[j], top_k[j], g[j], k, j);
        }
      }

    } else if (args["--order"].asString() == "gemm") {
      if (verbose) {
        std::cout << "Using gemm ordering" << std::endl;
      }

      /**
       * scores is nsamples X nq
       * db is dimension X nsamples
       * q is vsize X dimension
       * scores <- db^T * q
       */
      std::vector<std::span<float>> scores(size(q));
      std::vector<float> _score_data(size(q) * size(db));
      size_t M = size(db);
      size_t N = size(q);
      size_t K = size(db[0]);
      assert(size(db[0]) == size(q[0]));
      assert(size(db[0]) == dimension);

      // Each score[j] is a column of the score matrix
      for (size_t j = 0; j < size(q); ++j) {
        scores[j] = std::span<float>(_score_data.data() + j * M, M);
      }

      std::vector<float> alpha(M, 0.0f);
      std::vector<float> beta(N, 0.0f);

      {
        life_timer _{"L2 comparisons"};

        col_sum(db, alpha, [](auto a) { return a * a; });
        col_sum(q, beta, [](auto a) { return a * a; });

        for (size_t j = 0; j < N; ++j) {
          for (size_t i = 0; i < M; ++i) {
            scores[j][i] += alpha[i] + beta[j];
          }
        }
        cblas_sgemm(
                CblasColMajor,
                CblasTrans,  // db^T
                CblasNoTrans,// q
                (int32_t) M, // number of samples
                (int32_t) N, // number of queries
                (int32_t) K, // dimension of vectors
                -2.0,
                db[0].data(),// A: K x M -> A^T: M x K
                K,
                q[0].data(),// B: K x N
                K,
                1.0,
                _score_data.data(),// C: M x N
                M);

        for (size_t j = 0; j < N; ++j) {
          for (size_t i = 0; i < M; ++i) {
            scores[j][i] = std::sqrtf(scores[j][i]);
          }
        }
      }

      {
        life_timer _{"Get top k"};

        std::vector<size_t> index(size(db));
        for (size_t j = 0; j < size(q); ++j) {
          get_top_k(scores[j], top_k[j], index, k);
        }
      }

      {
        life_timer _{"Checking results"};
        for (size_t j = 0; j < size(q); ++j) {
          verify_top_k(scores[j], top_k[j], g[j], k, j);
        }
      }
    } else {
      std::cout << "Unknown ordering: " << args["--order"].asString() << std::endl;
      return 1;
    }
  }
}