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
void query_gemm(
    const DB& db,
    const Q& q,
    const G& g,
    int k,
    [[maybe_unused]] bool hw,
    size_t nthreads) {
  auto top_k = gemm_compute_scores(db, q, k, nthreads);
  if (g.num_rows() > 0) {
    life_timer _{"Checking results"};

    size_t size_q = size(q);
    for (size_t j = 0; j < size_q; ++j) {
      verify_top_k(scores[j], top_k[j], g[j], k, j);
    }
  }
}

template <class DB, class Q, class G>
void blocked_query_gemm(
    DB& db,
    Q& q,
    const G& g,
    int k,
    [[maybe_unused]] bool hw,
    size_t nthreads) {
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

/**
 * @brief Query a set of vectors against a vector database, returning the
 * indices of the best matches for each query vector.  The difference between
 * partition and query is that query returns the indices for the top k
 * scores, whereas partition returns just the top index.
 *
 * @tparam DB
 * @tparam Q
 * @param db
 * @param q
 * @param k
 * @param nthreads
 * @return
 */
template <class DB, class Q>
auto gemm_query(const DB& db, const Q& q, size_t k, unsigned nthreads) {
  life_timer _outer{"Total time gemm_query"};

  auto scores = gemm_scores(db, q, nthreads);

  ColMajorMatrix<size_t> top_k(k, q.num_cols());
  {
    life_timer _{"top k"};
    get_top_k(scores, top_k, k, size(q), size(db), nthreads);
  }

  return top_k;
}

/**
 * @brief Query a set of vectors against a vector database, returning the
 * indices of the best matches for each query vector.  The difference between
 * partition and query is that query returns the indices for the top k
 * scores, whereas partition returns just the top index.
 *
 * @tparam DB
 * @tparam Q
 * @param db
 * @param q
 * @param k
 * @param nthreads
 * @return
 */

/**
 * @brief Query a set of vectors against a vector database, returning the
 * indices of the best matches for each query vector.  The difference between
 * partition and query is that query returns the indices for the top k
 * scores, whereas partition returns just the top index.
 *
 * @tparam DB
 * @tparam Q
 * @param db
 * @param q
 * @param k
 * @param nthreads
 * @return
 */
template <class DB, class Q>
auto blocked_gemm_query(DB& db, Q& q, size_t k, unsigned nthreads) {
  life_timer _outer{"Total time blocked_gemm_query"};
  using element = std::pair<float, unsigned>;

  const auto block_db = db.is_blocked();
  const auto block_q = q.is_blocked();
  if (block_db && block_q) {
    throw std::runtime_error("Can't block both db and q");
  }

  ColMajorMatrix<float> scores(db.num_cols(), q.num_cols());
  std::vector<fixed_min_set<element>> min_scores(
      size(q), fixed_min_set<element>(k));

  for (;;) {
    gemm_scores(db, q, scores, nthreads);

    auto par = stdx::execution::indexed_parallel_policy{nthreads};
    stdx::range_for_each(
        std::move(par), scores, [&](auto&& q_vec, auto&& n = 0, auto&& i = 0) {
          if (block_db) {
            for (int j = 0; j < scores.num_rows(); ++j) {
              min_scores[i].insert({scores(j, i), j + db.offset()});
            }
          } else if (block_q) {
            for (int j = 0; j < scores.num_rows(); ++j) {
              min_scores[i + q.offset()].insert({scores(j, i), j});
            }
          } else {
            for (int j = 0; j < scores.num_rows(); ++j) {
              min_scores[i].insert({scores(j, i), j});
            }
          }
        });

    bool done = true;
    if (block_db) {
      done = !db.advance();
    } else if (block_q) {
      done = !q.advance();
    }
    if (done) {
      break;
    }
  }

  ColMajorMatrix<size_t> top_k(k, q.num_cols());

  for (int j = 0; j < scores.num_rows(); ++j) {
    std::transform(
        min_scores[j].begin(),
        min_scores[j].end(),
        top_k[j].begin(),
        ([](auto&& e) { return e.second; }));
  }

  return top_k;

  std::fill(begin(alpha), end(alpha), 0.0f);
  std::fill(begin(beta), end(beta), 0.0f);

  mat_col_sum(
      db, alpha, [](auto a) { return a * a; });  // @todo optimize somehow
  mat_col_sum(q, beta, [](auto a) { return a * a; });

  // A += alpha * x * transpose(y)

  // This should be more parallelizable -- but seems to be completely
  // memory-bound
  cblas_sger(
      CblasColMajor,
      M,
      N,
      1.0,
      &alpha[0],
      1,
      &alpha_ones[0],
      1,
      &scores(0, 0),
      M);
  cblas_sger(
      CblasColMajor,
      M,
      N,
      1.0,
      &beta_ones[0],
      1,
      &beta[0],
      1,
      &scores(0, 0),
      M);

  stdx::execution::parallel_policy par{nthreads};
  stdx::for_each(
      std::move(par), begin(_score_data), end(_score_data), [](auto& a) {
        a = sqrt(a);
      });
}

cblas_sgemm(
    CblasColMajor,
    CblasTrans,
    CblasNoTrans,
    M,
    N,
    K,
    -2.0,
    &db(0, 0),
    K,
    &q(0, 0),
    K,
    0.0,
    &scores(0, 0),
    M);



/**
 * Derived from `Matrix`.  Initialized in construction by filling from a given
 * TileDB array.
 *
 * @todo Evaluate whether or not we really need to do things this way or if
 * it is sufficient to simply have one Matrix class and have a factory that
 * creates them by reading from TileDB.
 */
template <class T, class LayoutPolicy = stdx::layout_right, class I = size_t>
class BlockedMatrix : public Matrix<T, LayoutPolicy, I> {
  using Base = Matrix<T, LayoutPolicy, I>;
  using Base::Base;

 public:
  using index_type = typename Base::index_type;
  using size_type = typename Base::size_type;
  using reference = typename Base::reference;

  using view_type = Base;

  constexpr static auto matrix_order_{order_v<LayoutPolicy>};

 private:
  size_t num_array_rows_{0};
  size_t num_array_cols_{0};

  std::tuple<index_type, index_type> row_view_;
  std::tuple<index_type, index_type> col_view_;

  index_type row_offset_{0};
  index_type col_offset_{0};

 public:
  // One-D blocking
  BlockedMatrix(size_t array_rows, size_t array_cols, size_t block) {
#ifndef __APPLE__
    auto data_ = std::make_unique_for_overwrite<T[]>(array_rows * array_cols);
#else
    // auto data_ = std::make_unique<T[]>(new T[mat_rows_ * mat_cols_]);
    auto data_ = std::unique_ptr<T[]>(new T[array_rows * array_cols]);
#endif

    num_array_rows_ = array_rows;
    num_array_cols_ = array_cols;

    size_t num_rows{0};
    size_t num_cols{0};

    if (matrix_order_ == TILEDB_ROW_MAJOR) {
      num_rows = std::min(block, array_rows);
      num_cols = array_cols;
    } else if (matrix_order_ == TILEDB_COL_MAJOR) {
      num_rows = array_rows;
      num_cols = std::min(block, array_cols);
    }

    std::get<0>(row_view_) = 0;
    std::get<1>(row_view_) = num_rows;
    std::get<0>(col_view_) = 0;
    std::get<1>(col_view_) = num_cols;

    Base::operator=(Base{std::move(data_), num_rows, num_cols});
  }

  bool advance(size_t num_elts = 0) {
    if (matrix_order_ == TILEDB_ROW_MAJOR) {
      if (num_elts == 0) {
        num_elts = std::get<1>(row_view_) - std::get<0>(row_view_);
      }
      num_elts = std::min(num_elts, num_array_rows_ - row_offset_);
      row_offset_ += num_elts;
      std::get<0>(row_view_) += num_elts;
      std::get<1>(row_view_) += num_elts;

      if (std::get<0>(row_view_) >= num_array_rows_) {
        return false;
      }
    } else if (matrix_order_ == TILEDB_COL_MAJOR) {
      if (num_elts == 0) {
        num_elts = std::get<1>(col_view_) - std::get<0>(col_view_);
      }
      num_elts = std::min(num_elts, num_array_cols_ - col_offset_);
      col_offset_ += num_elts;
      std::get<0>(col_view_) += num_elts;
      std::get<1>(col_view_) += num_elts;

      if (std::get<0>(col_view_) >= num_array_cols_) {
        return false;
      }
    } else {
      throw std::runtime_error("Unknown cell order");
    }
    return true;
  }

  auto raveled()
    requires(matrix_order_ == TILEDB_COL_MAJOR)
  {
    auto num_elts = std::get<1>(col_view_) - std::get<0>(col_view_);
    return std::span(
        this->storage_.get() + col_offset_ * this->num_rows(),
        this->nrows * num_elts);
  }

  auto raveled()
    requires(matrix_order_ == TILEDB_ROW_MAJOR)
  {
    auto num_elts = std::get<1>(row_view_) - std::get<0>(row_view_);
    return std::span(
        this->storage_.get() + row_offset_ * this->num_cols(),
        this->ncols * num_elts);
  }

  auto raveled() const
    requires(matrix_order_ == TILEDB_COL_MAJOR)
  {
    auto num_elts = std::get<1>(col_view_) - std::get<0>(col_view_);
    return std::span(
        this->storage_.get() + col_offset_ * this->num_rows(),
        this->nrows * num_elts);
  }

  auto raveled() const
    requires(matrix_order_ == TILEDB_ROW_MAJOR)
  {
    auto num_elts = std::get<1>(row_view_) - std::get<0>(row_view_);
    return std::span(
        this->storage_.get() + row_offset_ * this->num_cols(),
        this->ncols * num_elts);
  }

  size_t offset() const
    requires(std::is_same_v<LayoutPolicy, stdx::layout_right>)
  {
    return row_offset_;
  }

  size_t offset() const
    requires(std::is_same_v<LayoutPolicy, stdx::layout_left>)
  {
    return col_offset_;
  }

  constexpr static bool is_blocked() {
    return true;
  }
};
