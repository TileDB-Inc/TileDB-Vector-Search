#include "utils/fixed_min_heap.h"
#include "algorithm.h"
#include "scoring.h"
#include "tdb_defs.h"
#include "concepts.h"
#include "cpos.h"

template <feature_vector_range DB, feature_vector_range Q, std::integral Index>
auto qv_query_heap(const DB& db, const Q& query, int k_nn, unsigned nthreads) {

  auto top_k = ColMajorMatrix<size_t>(k_nn, num_vectors(query));
  auto top_k_scores = ColMajorMatrix<float>(k_nn, num_vectors(query));

  auto min_scores = k_min_heap<float, Index>(k_nn);

  for (size_t j = 0; j < num_vectors(query); ++j) {
    feature_vector auto q_vec = query[j];
    for (size_t i = 0; i < num_vectors(db); ++i) {
      auto score = L2(q_vec, db[i]);
      min_scores.insert(score, i);
    }
    get_top_k_with_scores_from_heap(min_scores, top_k[j], top_k_scores[j]);
  }

  return std::make_tuple(std::move(top_k_scores), std::move(top_k));
}