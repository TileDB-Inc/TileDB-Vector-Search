#if 0
template <class DB, class Q, class G, class TK>
void blocked_query_gemm(DB& db, Q& q, const G& g, TK& top_k, int k, [[maybe_unused]] bool hw, size_t nthreads) {
  auto scores = blocked_gemm_compute_scores(db, q, top_k, k, nthreads);
  if (g.num_rows() > 0) {
    life_timer _ { "Checking results" };

    size_t size_q = size(q);
    for (size_t j = 0; j < size_q; ++j) {
      // verify_top_k(scores[j], top_k[j], g[j], k, j);
      std::sort(begin(g[j]), begin(g[j]) + k);
      std::sort(begin(top_k[j]), end(top_k[j]));
      if (!std::equal(begin(top_k[j]), end(top_k[j]), g[j])) {
        std::cout << "Solution vector " << top_k[j] << " != " << g[j] << std::endl;
      }
    }
  }
}
#else


template <class DB, class Q, class G>
void query_gemm(const DB& db, const Q& q, const G& g, int k, [[maybe_unused]] bool hw, size_t nthreads) {
  auto top_k = gemm_compute_scores(db, q, k, nthreads);
  if (g.num_rows() > 0) {
    life_timer _ { "Checking results" };

    size_t size_q = size(q);
    for (size_t j = 0; j < size_q; ++j) {
      verify_top_k(scores[j], top_k[j], g[j], k, j);
    }
  }
}


template <class DB, class Q, class G>
void blocked_query_gemm(DB& db, Q& q, const G& g, int k, [[maybe_unused]] bool hw, size_t nthreads) {
  auto top_k = blocked_gemm_compute_scores(db, q, k, nthreads);
  verify_top_k(top_k, g);
}


#else

// Either of these seems okay maybe
#if 0
  using Comparator = std::function<bool(unsigned, unsigned)>;

  fixed_min_set<unsigned, Comparator> s(k, [&](unsigned a, unsigned b) {
    return scores[a] < scores[b];
  });
  for (auto i : index) {
    s.insert(i);
  }
  // std::sort_heap(begin(s), end(s), [&](unsigned a, unsigned b) {
    // return scores[a] < scores[b];
  //});
  std::copy(begin(s), end(s), begin(top_k));
  std::sort(begin(top_k), end(top_k), [&](unsigned a, unsigned b) {
    return scores[a] < scores[b];
  });
#else

#endif
#endif
}



#if 0
  if (!std::equal(begin(top_k), begin(top_k) + k, g.begin())) {
    std::cout << "Query " << qno << " is incorrect" << std::endl;
    for (int i = 0; i < std::min(k, 10); ++i) {
      std::cout << "  (" << top_k[i] << " " << g[i] << ")";
    }
    std::cout << std::endl;
    return false;
  }
#endif