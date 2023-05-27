

#include <catch2/catch_all.hpp>
#include <thread>
#include "utils/timer.h"

#include "flat_query.h"
#include "ivf_query.h"
#include "linalg.h"

// Cases:
//    Big db / small db
//    Big q / small q
//    big k / small k
//    nth = true / false

TEST_CASE("time queries", "[queries]") {
  unsigned dimension = 128;
  unsigned nthreads = std::thread::hardware_concurrency();
#if 1
  unsigned small_threads = 1;
  unsigned med_threads = nthreads / 4;
  unsigned big_threads = nthreads;

  unsigned small_q = 1;
  unsigned med_q = 100;
  unsigned big_q = 10000;

  unsigned small_db = 1000;
  unsigned med_db = 100000;
  unsigned big_db = 10000000;

  unsigned small_k = 1;
  unsigned med_k = 10;
  unsigned big_k = 100;

  unsigned true_nth = true;
  unsigned false_nth = false;
#else
  unsigned small_threads = 1;
  unsigned med_threads = 2;
  unsigned big_threads = 4;

  unsigned small_q = 1;
  unsigned med_q = 10;
  unsigned big_q = 100;

  unsigned small_db = 100;
  unsigned med_db = 1000;
  unsigned big_db = 10000;

  unsigned small_k = 1;
  unsigned med_k = 5;
  unsigned big_k = 10;

  unsigned true_nth = true;
  unsigned false_nth = false;

#endif
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-128, 128);

  for (auto db : {small_db, med_db, big_db}) {
    for (auto q : {small_q, med_q, big_q}) {
      auto db_mat = ColMajorMatrix<float>(dimension, db);
      for (auto& x : raveled(db_mat)) {
        x = dist(gen);
      }
      auto q_mat = ColMajorMatrix<float>(dimension, q);
      for (auto& x : raveled(q_mat)) {
        x = dist(gen);
      }
      for (auto nthreads : {small_threads, med_threads, big_threads}) {
        for (auto k : {small_k, med_k, big_k}) {

          std::cout << "\n# [ Experiment ]: nthreads: " << nthreads
                    << " q: " << q << " db: " << db << " k: " << k << std::endl;
          {
            life_timer _outer{"qv_query"};
            qv_query(db_mat, q_mat, k, nthreads);
          }
          {
            life_timer _outer{"vq_query_heap"};
            vq_query_heap(db_mat, q_mat, k, nthreads);
          }

          for (auto nth : {true_nth, false_nth}) {
            std::cout << "\n# [ Experiment: ]: nthreads: " << nthreads
                      << " q: " << q << " db: " << db << " k: " << k
                      << " nth: " << nth << std::endl;
            {
              life_timer _outer{"qv_query_nth"};
              qv_query_nth(db_mat, q_mat, k, nth, nthreads);
            }
            {
              life_timer _outer{"vq_query_nth"};
              vq_query_nth(db_mat, q_mat, k, nth, nthreads);
            }
            {
              life_timer _outer{"gemm_query"};
              gemm_query(db_mat, q_mat, k, nth, nthreads);
            }
          }
        }
      }
    }
  }
}
