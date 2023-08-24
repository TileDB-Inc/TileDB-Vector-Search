/**
* @file   csr.h
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
*/

#ifndef TILEDB_CSR_H
#define TILEDB_CSR_H

#include "coo.h"
#include "utils/logging.h"
#include "detail/linalg/vector.h"
#include <algorithm>
#include <cassert>
#include <future>
#include <initializer_list>
#include <iostream>
#include <map>
#include <set>
#include <thread>
#include <tuple>

#include "algorithm.h"
#include "execution_policy.h"


// @todo Parallelize
template <class V>
auto make_translation_table(const V& vec, bool noisy = false) {
 scoped_timer _ { "make_translation_table", noisy };

 using index_type = typename V::index_type;
 using value_type = typename V::value_type;

 auto par      = stdx::execution::indexed_parallel_policy();
 auto nthreads = par.nthreads_;

 auto sets = std::vector<std::set<value_type>>(nthreads);

 stdx::range_for_each(std::move(par), vec, [&sets](auto&& v, auto&& n, auto&& i) { sets[n].insert({ v }); });

 // Size of #unique elements in each set
 auto index_set = std::set<value_type> {};
 for (auto&& s : sets) {
   index_set.insert(begin(s), end(s));
 }

 // Create a map to associate each element with a unique integer
 // @todo Does ordering matter here?  Maybe use unordered_map?
 auto index_map = std::map<value_type, index_type> {};

 index_type unique_integer = 0;
 for (const auto& element : index_set) {
   index_map[element] = unique_integer++;
 }

 std::vector<value_type> unique_ids(index_map.size());
 for (auto&& [k, v] : index_map) {
   unique_ids[v] = k;
 }

 return std::make_tuple(index_map, unique_ids);
}

template <class ValueType, class IndexType>
class coo_matrix;

/**
* @brief A CSR matrix class.  The row pointers, column indices, and values are stored in separate `Vector`
* structures.  The primary means of constructing a CSR matrix is via a COO matrix.
*
* @tparam ValueType
* @tparam IndexType
*
* @note There are currently a number of debugging and diagnostic statements in the code.
*/
// @todo: Add a PointerType which may be different from IndexType
template <class ValueType, class IndexType = size_t>
class csr_matrix {
 friend class coo_matrix<ValueType, IndexType>;

 using index_type = IndexType;
 using value_type = ValueType;

 size_t             num_rows_ { 0 };
 size_t             nnz_ { 0 };
 Vector<index_type> row_ptr_;
 Vector<index_type> col_idx_;
 Vector<value_type> values_;

 bool relabeled_ { false };
 bool noisy_ { true };
 bool debug_ { false };

 // @todo This isn't quite right -- we really want the types of the row_idx_ array
 std::map<typename decltype(row_ptr_)::value_type, typename decltype(col_idx_)::index_type> row_index_map_;
 std::vector<typename decltype(row_ptr_)::value_type>                                       row_unique_ids_;
 std::map<typename decltype(col_idx_)::value_type, typename decltype(col_idx_)::index_type> col_index_map_;
 std::vector<index_type>                                                                    col_unique_ids_;


public:
 csr_matrix(size_t num_rows) : row_ptr_(num_rows + 1), col_idx_(0), values_(0) {
 }

 /**
  * Move constructor from COO matrix.  The column index and value arrays are moved from the COO matrix and the row index is relabeled.
  * @param coo
  * @param relabel
  */
 csr_matrix(coo_matrix<ValueType, IndexType>&& coo, bool relabel = false)
     : nnz_(coo.nnz_), col_idx_ { std::move(coo.col_idx_) }, values_ { std::move(coo.values_) } {


   /*
    * Sort all three arrays of the COO matrix.  By sorting the row indices we
    * enable the use of std::adjacent_find to compute the row_ptr_ array.
    *
    * @todo Need to evaluate the tradeoffs of different ways of sorting multiple arrays at once.
    * @todo Also need to profile the sort in depth to make sure there are no inadvertent copies.
    *
    * The sort seems to be a win when there are a modest number of elements per row (on average)
    * and there is a fairly high degree of parallelism (number of threads).
    *
    */
   auto z = zipped(coo.row_idx_, col_idx_, values_);

   {
     auto a = scoped_timer("sort", noisy_);

     // @todo: Better sort that takes advantage of repeated elements
     stdx::sort(stdx::execution::par_unseq, begin(z), end(z), [](auto&& a, auto&& b) { return std::get<0>(a) < std::get<0>(b); });
   }

   if (relabel) {
     relabel_indices(coo.row_idx_);
   } else {
     auto&& [lo, hi] = std::minmax_element(stdx::execution::par_unseq, begin(coo.row_idx_), end(coo.row_idx_));
     num_rows_       = *hi - *lo + 1;
   }

   if (debug_) {
     std::cout << "num_rows_: " << num_rows_ << "\n";
   }

   row_ptr_ = std::move(Vector<index_type>(num_rows_ + 1));
   stdx::fill(stdx::execution::par_unseq, begin(row_ptr_), end(row_ptr_), 0);

#if 1
   // @todo: Parallelize this with divide and conquer
   auto it = begin(coo.row_idx_);
   while (it != end(coo.row_idx_)) {

     // Do NOT parallelize this with execution policy -- it will fail (though should not)
     auto   next_it = stdx::adjacent_find(stdx::execution::seq, it, end(coo.row_idx_), std::not_equal_to<>());
     size_t length  = std::distance(it, next_it == end(coo.row_idx_) ? next_it : std::next(next_it));

     row_ptr_(*it) = length;
     if (next_it == end(coo.row_idx_)) {
       break;
     }
     it = next(next_it);
   }

   // @todo Parallelize (though row_ptr_ might not be large enough to benefit
   // Standard allows in place scan, but g++ does not correctly implement it
   std::exclusive_scan(begin(row_ptr_), end(row_ptr_), begin(row_ptr_), 0);
#else

   // Set up row_ptr_ to be the number of nonzeros in previous row
   for (int i = 0; i < nnz_; ++i) {
     ++row_ptr_(coo.row_idx_(i) + 1);
   }

   std::inclusive_scan(PAR begin(row_ptr_), end(row_ptr_), begin(row_ptr_));
   row_ptr_(0) = 0;

#endif

   if (debug_ && num_rows_ < 20) {
     printf_debug(coo.row_idx_);
   }
 }

 /**
  * Copying constructor from COO matrix.  Demonstrates non-sorting approach to constructing CSR matrix.
  */
 csr_matrix(const coo_matrix<ValueType, IndexType>& coo, bool relabel = false) : nnz_(coo.nnz_), col_idx_(nnz_), values_(nnz_) {
   scoped_timer _ { "csr_matrix copy constructor", noisy_ };
   // Make a copy of the row index array -- we only need this in the case of relabeling
   // @todo Be more clever -- don't copy if we don't relabel
   auto row_idx = Vector<index_type>(nnz_);
   stdx::copy(stdx::execution::par_unseq, coo.row_idx_.begin(), coo.row_idx_.end(), row_idx.begin());

   if (relabel) {
     relabel_indices(row_idx);
   } else {
     auto&& [lo, hi] = stdx::minmax_element(stdx::execution::par_unseq, begin(coo.row_idx_), end(coo.row_idx_));
     num_rows_       = *hi - *lo + 1;
   }

   if (debug_) {
     std::cout << "num_rows_: " << num_rows_ << "\n";
   }

#if 0
   auto max_row_el = *std::max_element(PAR begin(row_idx), end(row_idx));
   auto min_row_el = *std::min_element(PAR begin(row_idx), end(row_idx));
   auto max_col_el = *std::max_element(PAR begin(coo.col_idx_), end(coo.col_idx_));
   auto min_col_el = *std::min_element(PAR begin(coo.col_idx_), end(coo.col_idx_));

   std::cout << "max_row_el: " << max_row_el << " ";
   std::cout << "min_row_el: " << min_row_el << " ";
   std::cout << "max_col_el: " << max_col_el << " ";
   std::cout << "min_col_el: " << min_col_el << "\n";
#endif

   row_ptr_ = std::move(Vector<index_type>(num_rows_ + 1));
   stdx::fill(stdx::execution::par_unseq, begin(row_ptr_), end(row_ptr_), 0);

   /*
    * The below is kind of a counting sort, which should be parallelizable.
    * @todo Parallelize this without requiring temporary storage
    * (though if size(row_ptr) << nthreads * size(row_idx)then even with temp storage it could be a win)
    */
#if 1
   {
     scoped_timer _ { "count sort 1", noisy_ };
     // Set up row_ptr_ to be the number of nonzeros in previous row
     for (int i = 0; i < nnz_; ++i) {
       ++row_ptr_(row_idx(i) + 1);
     }
   }
#else
   auto par    = stdx::execution::indexed_parallel_policy();

   std::cout << "nthreads: " << par.nthreads_ << "\n";

   std::vector<Vector<value_type>> tmps(par.nthreads_);
   for (size_t n = 0; n < par.nthreads_; ++n) {
     tmps[n] = std::move(Vector<value_type>(size(row_ptr_)));
     stdx::fill(stdx::execution::par_unseq, begin(tmps[n]), end(tmps[n]), 0);
   }

   stdx::range_for_each(std::move(par), row_idx, [&tmps](auto&& v, auto&& n, auto&& i) { ++tmps[n](v + 1); });
   for (size_t n = 0; n < par.nthreads_; ++n) {
     for (size_t i = 0; i < size(row_ptr_); ++i) {
       row_ptr_(i) += tmps[n](i);
     }
   }
#endif

   stdx:inclusive_scan(stdx::execution::par_unseq, begin(row_ptr_), end(row_ptr_), begin(row_ptr_));

   {
     scoped_timer cs2("count sort 2", noisy_);
     // Place values and col_idx_ in correct location
     // @todo parallelize?
     for (int i = 0; i < nnz_; ++i) {
       col_idx_(row_ptr_(row_idx(i))) = coo.col_idx_(i);
       values_(row_ptr_(row_idx(i)))  = coo.values_(i);
       ++row_ptr_(row_idx(i));
     }
   }

   // @todo: Use execution policy
   // Parallel overload does not seem to be implemented at all yet in g++
   std::shift_right(begin(row_ptr_), end(row_ptr_), 1);

   row_ptr_(0) = 0;

   // Yes, printf debugging
   if (noisy_ && num_rows_ < 20) {
     printf_debug(row_idx);
   }
 }

 auto nnz() const noexcept {
   return nnz_;
 }

 auto num_rows() const noexcept {
   return num_rows_;
 }


private:
 /**
  * relabel the row and column indices in the COO matrix to be contiguous from zero.
  * This is necessary for the CSR matrix (at least for the rows) since we index into
  * the row_ptr_ array with values between zero and num_rows_.  It may not be necessary
  * for the columns, but we do both for now.
  *
  * @param row_idx -- the row indices of the COO matrix
  *
  */
 auto relabel_indices(auto&& row_idx) {
   {
     scoped_timer _ { "get maps", noisy_ };
     /*
      * Get maps from unique ids to indices and table from indices to unique ids
      */
     std::tie(row_index_map_, row_unique_ids_) = make_translation_table(row_idx, noisy_);
     std::tie(col_index_map_, col_unique_ids_) = make_translation_table(col_idx_, noisy_);
   }

   assert(size(row_index_map_) == size(row_unique_ids_));
   num_rows_ = size(row_index_map_);
   {
     scoped_timer _ { "relabel", noisy_ };
     // auto         par = stdx::execution::indexed_parallel_policy();
     //      stdx::range_for_each(std::move(par), row_idx, [this](auto&& v, auto&& n, auto&& i) { v = row_index_map_[v]; });
     //      stdx::range_for_each(std::move(par), col_idx_, [this](auto&& v, auto&& n, auto&& i) { v = col_index_map_[v]; });
     stdx::for_each(stdx::execution::par_unseq, begin(row_idx), end(row_idx), [this](auto&& v) { v = row_index_map_[v]; });
     stdx::for_each(stdx::execution::par_unseq, begin(col_idx_), end(col_idx_), [this](auto&& v) { v = col_index_map_[v]; });
   }

   relabeled_ = true;
 }

 void printf_debug(auto&& coo_row_idx) {
   for (auto j : row_ptr_) {
     std::cout << j << " ";
   }
   std::cout << std::endl;

   for (auto j : coo_row_idx) {
     std::cout << j << " ";
   }
   std::cout << std::endl;

   for (auto j : col_idx_) {
     std::cout << j << " ";
   }
   std::cout << std::endl;

   for (auto j : values_) {
     std::cout << j << " ";
   }
   std::cout << std::endl;
 }
};
#endif