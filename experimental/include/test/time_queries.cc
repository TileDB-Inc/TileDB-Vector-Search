

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
  size_t dimension = 128;
  size_t nthreads = std::thread::hardware_concurrency();
#if 1
  size_t small_threads = 1;
  size_t med_threads = nthreads / 4;
  size_t big_threads = nthreads;

  size_t small_q = 1;
  size_t med_q = 100;
  size_t big_q = 10000;

  size_t small_db = 1000;
  size_t med_db = 100000;
  size_t big_db = 10000000;

  size_t small_k = 1;
  size_t med_k = 10;
  size_t big_k = 100;

  size_t true_nth = true;
  size_t false_nth = false;
#else
  size_t small_threads = 1;
  size_t med_threads = 2;
  size_t big_threads = 4;

  size_t small_q = 1;
  size_t med_q = 10;
  size_t big_q = 100;

  size_t small_db = 100;
  size_t med_db = 1000;
  size_t big_db = 10000;

  size_t small_k = 1;
  size_t med_k = 5;
  size_t big_k = 10;

  size_t true_nth = true;
  size_t false_nth = false;

#endif
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-128, 128);

  for (auto db : {small_db, med_db, big_db}) {
    for (auto q : {small_q, med_q, big_q}) {
      
      if (db * q * 128 > 8'000'000'000) {
        continue;
      }
      
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
