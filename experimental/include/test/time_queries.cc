

#include <catch2/catch_all.hpp>
#include <thread>

#include <iostream>
#include <filesystem>
#include "utils/timer.h"

#include "flat_query.h"
#include "ivf_query.h"
#include "linalg.h"

bool global_debug = false;
std::string global_region = "us-east-1";

TEST_CASE("time queries", "[queries]") {
  size_t dimension = 128;

  size_t nthreads = GENERATE(1, 8);

  size_t k = GENERATE(1, 10, 100);

  std::tuple<size_t, size_t> expts = GENERATE(

      std::make_tuple(10000000, 1),
      std::make_tuple(1, 10000),

      std::make_tuple(1000000, 10),
      std::make_tuple(10, 10000),

      std::make_tuple(1000000, 100),
      std::make_tuple(100, 1000),

      std::make_tuple(100000, 1000),
      std::make_tuple(1000, 1000));
  size_t nth = GENERATE(true, false);

  std::cout << "\n=======================================================\n";

  auto&& [db, q] = expts;
  if (k <= db) {
    //  std::filesystem::path currentPath = std::filesystem::current_path();
    // std::cout << "Current working directory: " << currentPath << std::endl;

    std::string db_name = "db_" + std::to_string(db) + ".tdb";
    std::string q_name = "q_" + std::to_string(q) + ".tdb";

    // std::cout << "db_name: " << db_name << " q_name: " << q_name << "\n";

    auto db_mat = tdbColMajorMatrix<float>(db_name);

    auto q_mat = tdbColMajorMatrix<float>(q_name);

    std::cout << "\n# [ Experiment: ]: nthreads: " << nthreads << " q: " << q
              << " db: " << db << " k: " << k << " nth: " << nth << "\n\n";

    //    if constexpr (false)
    {
      life_timer _outer{"qv_query"};
      qv_query(db_mat, q_mat, k, nthreads);
    }
    //    if constexpr (false)
    {
      life_timer _outer{"qv_query_nth"};
      qv_query_nth(db_mat, q_mat, k, nth, nthreads);
    }
    {
      life_timer _outer{"vq_query_heap"};
      vq_query_heap(db_mat, q_mat, k, nthreads);
    }
    {
      life_timer _outer{"vq_query_nth"};
      vq_query_nth(db_mat, q_mat, k, nth, nthreads);
    }
    {
      life_timer _outer{"gemm_query"};
      gemm_query(db_mat, q_mat, k, nth, nthreads);
    }
#if 0
  {
    life_timer _outer{"blocked_gemm_query"};
    blocked_gemm_query(b_db_mat, q_mat, k, nth, nthreads);
  }
#endif
  }
}
