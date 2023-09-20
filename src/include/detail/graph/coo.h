/**
* @file   coo.h
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

#ifndef TILEDB_COO_H
#define TILEDB_COO_H

#include "csr.h"
#include "detail/linalg/vector.h"
#include <initializer_list>
#include <random>
#include <tiledb/tiledb>

template <class ValueType, class IndexType>
class csr_matrix;

/**
 * Simple COO matrix class.  This is a simple wrapper around three vectors that respectively store row, column, and
 * value data.  The vectors are assumed to be the same size, and the i-th element of each vector is assumed to
 * correspond to the (row(i), col(i)) element of the matrix.
 * @tparam ValueType
 * @tparam IndexType
 */
template <class ValueType, class IndexType = size_t>
class coo_matrix {
 public:
  // copilot scares me sometimes
  using index_type = IndexType;
  using value_type = ValueType;

 private:
  friend class csr_matrix<ValueType, IndexType>;

  size_t             nnz_ { 0 };
  Vector<index_type> row_idx_;
  Vector<index_type> col_idx_;
  Vector<value_type> values_;

 public:
  coo_matrix() = default;

  coo_matrix(size_t nnz) : nnz_(nnz), row_idx_(nnz), col_idx_(nnz), values_(nnz) {
  }

  /**
   * @brief Construct a new coo matrix object from three vectors
   * @param row_idx
   * @param col_idx
   * @param values
   */
  coo_matrix(Vector<index_type>&& row_idx, Vector<index_type>&& col_idx, Vector<value_type>&& values)
      : nnz_(row_idx.size()), row_idx_(std::move(row_idx)), col_idx_(std::move(col_idx)), values_(std::move(values)) {
    assert(row_idx.size() == col_idx.size() && col_idx.size() == values.size());
  }

  /**
   * @brief Construct a new coo matrix object from a csr matrix
   * @param csr
   * @param unlabel
   */
  coo_matrix(csr_matrix<value_type, index_type>&& csr, bool unlabel = false)
      : nnz_(csr.nnz_), col_idx_(std::move(csr.col_idx_)), values_(std::move(csr.values_)), row_idx_(nnz_) {


    if (unlabel) {
      // @todo Undo relabeling (WIP)
    }

    // @todo: parallelize this
    for (size_t i = 0; i < csr.row_ptr_.size() - 1; ++i) {
      for (size_t j = csr.row_ptr_(i); j < csr.row_ptr_(i + 1); ++j) {
        row_idx_(j) = i;
      }
    }
  }

  /**
   * @brief Construct a new coo matrix object by moving from another coo matrix
   * @param rhs
   */
  coo_matrix(coo_matrix&& rhs) noexcept
      : nnz_(rhs.nnz_), row_idx_(std::move(rhs.row_idx_)), col_idx_(std::move(rhs.col_idx_)), values_(std::move(rhs.values_)) {
  }

  /**
   * @brief Construct a new coo matrix object from an initializer list
   * @param lst
   */
  coo_matrix(const std::initializer_list<std::tuple<IndexType, IndexType, ValueType>>& lst)
      : nnz_(lst.size()), row_idx_(lst.size()), col_idx_(lst.size()), values_(lst.size()) {

    auto row_ptr = row_idx_.data();
    auto col_ptr = col_idx_.data();
    auto val_ptr = values_.data();

    // Initializer lists are presumed to be small, so no point parallelizing this
    for (auto& [row, col, val] : lst) {
      *row_ptr++ = row;
      *col_ptr++ = col;
      *val_ptr++ = val;
    }
  }

  /**
   * @brief Construct a new coo matrix object from a container of tuples
   * @param lst
   */
  coo_matrix(const std::vector<std::tuple<IndexType, IndexType, ValueType>>& lst)
      : nnz_(lst.size()), row_idx_(lst.size()), col_idx_(lst.size()), values_(lst.size()) {

    auto row_ptr = row_idx_.data();
    auto col_ptr = col_idx_.data();
    auto val_ptr = values_.data();

    // Initializer lists are presumed to be small, so no point parallelizing this
    for (auto& [row, col, val] : lst) {
      *row_ptr++ = row;
      *col_ptr++ = col;
      *val_ptr++ = val;
    }
  }

  /**
   * @brief Move equality operator
   * @param rhs
   * @return
   */
  auto& operator=(coo_matrix&& rhs) noexcept {
    nnz_     = rhs.nnz_;
    row_idx_ = std::move(rhs.row_idx_);
    col_idx_ = std::move(rhs.col_idx_);
    values_  = std::move(rhs.values_);

    return *this;
  }

  /**
   * @brief Function operator, used for indexing
   * @param i
   * @return A tuple of elements comprising the i-th row of each vector
   */
  auto operator()(size_t i) {
    return std::tuple { row_idx_(i), col_idx_(i), values_(i) };
  }

  auto operator()(size_t i) const {
    return std::tuple { row_idx_(i), col_idx_(i), values_(i) };
  }

  /**
   * @brief Access pointers to underlying data
   * @return A tuple of elements comprising pointers of the underlying
   */
  auto data() const noexcept {
    return std::tuple { row_idx_.data(), col_idx_.data(), values_.data() };
  }

  /**
   * @brief Get the number of elements in each vector (the sizes of each vector are the same)
   * @return
   */
  auto nnz() const noexcept {
    //  assert(row_idx_.size() == col_idx_.size() && col_idx_.size() == values_.size());
    //    return row_idx_.size();
    return nnz_;
  }

  /**
   * @brief Shuffle the elements in the row vector.  Used only for debugging and testing.
   * @return
   */
  auto shuffle_rows() {
    std::random_device rd;
    std::mt19937       gen(rd());

    // Shuffle the elements in the vector
    std::shuffle(begin(row_idx_), end(row_idx_), gen);
  }
};

/**
 * @brief Simple COO class that reads its data from a sparse TileDB array.  The row and column
 * vectors are read from the array dimensions, and the values are read from the array attribute.
 * @tparam ValueType
 * @tparam IndexType
 */
template <class ValueType, class IndexType = size_t>
class tdb_coo_matrix : public coo_matrix<ValueType, IndexType> {

  using Base = coo_matrix<ValueType, IndexType>;
  using Base::Base;

  using index_type = typename Base::index_type;
  using value_type = typename Base::value_type;

  std::string uri_;
  std::string attr_name_;

 public:
  /**
   * @brief Construct a new tdb coo matrix object from a sparse TileDB array
   * @param ctx
   * @param uri
   */
  tdb_coo_matrix(const tiledb::Context& ctx, const std::string& uri)
      : uri_(uri) {
    tiledb::Array array_ = tiledb::Array(ctx, uri, TILEDB_READ);
    auto schema_ = array_.schema();

    /*
 *     schema_.dump(stdout);

    ArraySchema(
        domain=Domain(*[
          Dim(name='soma_dim_0', domain=(0, 2147483645), tile=2048,
 dtype='int64', filters=FilterList([ZstdFilter(level=3), ])),
          Dim(name='soma_dim_1', domain=(0, 2147483645), tile=2048,
 dtype='int64', filters=FilterList([ZstdFilter(level=3), ])),
        ]),
        attrs=[
          Attr(name='soma_data', dtype='float32', var=False, nullable=False,
 filters=FilterList([ZstdFilter(level=-1), ])),
        ],
        cell_order='row-major',
        tile_order='row-major',
        capacity=100000,
        sparse=True,
        allows_duplicates=False,
        )
*/

    // using dim_type  = int64_t;
    // using attr_type = float;

    auto domain{schema_.domain()};

    auto dim_num{domain.ndim()};
    if (dim_num != 2) {
      throw std::runtime_error(
          "Expected 2 dimensions, got " + std::to_string(dim_num));
    }

    auto attr_num{schema_.attribute_num()};
    if (attr_num != 1) {
      throw std::runtime_error(
          "Expected 1 attribute, got " + std::to_string(attr_num));
    }

    auto row_dim = domain.dimension(0);
    auto col_dim = domain.dimension(1);
    auto val_attr = schema_.attribute(0);

    auto row_dim_name = row_dim.name();
    auto col_dim_name = col_dim.name();
    auto val_attr_name = val_attr.name();
    if (row_dim_name != "soma_dim_0" && col_dim_name != "soma_dim_1" &&
        val_attr_name != "soma_data") {
      throw std::runtime_error("Attribute name mismatch");
    }

    tiledb_datatype_t row_dim_type = row_dim.type();
    tiledb_datatype_t col_dim_type = col_dim.type();
    tiledb_datatype_t val_attr_type = val_attr.type();
    if (row_dim_type != tiledb::impl::type_to_tiledb<index_type>::tiledb_type) {
      throw std::runtime_error(
          "Attribute type mismatch: " + std::to_string(row_dim_type) + " != " +
          std::to_string(
              tiledb::impl::type_to_tiledb<index_type>::tiledb_type));
    }
    if (col_dim_type != tiledb::impl::type_to_tiledb<index_type>::tiledb_type) {
      throw std::runtime_error(
          "Attribute type mismatch: " + std::to_string(col_dim_type) + " != " +
          std::to_string(
              tiledb::impl::type_to_tiledb<index_type>::tiledb_type));
    }
    if (val_attr_type !=
        tiledb::impl::type_to_tiledb<value_type>::tiledb_type) {
      throw std::runtime_error(
          "Attribute type mismatch: " + std::to_string(val_attr_type) + " != " +
          std::to_string(
              tiledb::impl::type_to_tiledb<value_type>::tiledb_type));
    }

    tiledb::Query query(ctx, array_);
    auto row_result_size = query.est_result_size(row_dim_name);
    auto col_result_size = query.est_result_size(col_dim_name);
    auto val_result_size = query.est_result_size(val_attr_name);

    assert(
        row_result_size == col_result_size &&
        col_result_size == 2 * val_result_size);

    auto row_ptr = Vector<index_type>(row_result_size / sizeof(index_type));
    auto col_ptr = Vector<index_type>(col_result_size / sizeof(index_type));
    auto val_ptr = Vector<value_type>(val_result_size / sizeof(value_type));

    std::cout << "row_result_size = " << size(row_ptr) << std::endl;

    auto subarray = tiledb::Subarray(ctx, array_);

    auto row_extent = row_dim.domain<index_type>();
    auto col_extent = row_dim.domain<index_type>();

    subarray.add_range(0, (index_type)0, (index_type)row_extent.second);
    subarray.add_range(1, (index_type)0, (index_type)col_extent.second);

    query.set_subarray(subarray)
        .set_layout(TILEDB_ROW_MAJOR)
        .set_data_buffer(row_dim_name, row_ptr.data(), row_ptr.size())
        .set_data_buffer(col_dim_name, col_ptr.data(), col_ptr.size())
        .set_data_buffer(val_attr_name, val_ptr.data(), val_ptr.size());

    query.submit();
    array_.close();

// printf debugging
#if 0
    std::sort(begin(row_ptr), end(row_ptr), std::less<index_type>());
    auto last_row = std::unique(begin(row_ptr), end(row_ptr));
    std::cout << "unique row elements = " << last_row - begin(row_ptr) << std::endl;
    std::sort(begin(col_ptr), end(col_ptr), std::less<index_type>());
    auto last_col = std::unique(begin(col_ptr), end(col_ptr));
    std::cout << "unique col elements = " << last_col - begin(col_ptr) << std::endl;

    auto result_el = query.result_buffer_elements();
    auto row_read = result_el[row_dim_name].second;
    auto col_read = result_el[col_dim_name].second;
    auto val_read = result_el[val_attr_name].second;

    std::cout << "row_read = " << row_read << std::endl;
    std::cout << "col_read = " << col_read << std::endl;
    std::cout << "val_read = " << val_read << std::endl;
#endif

    assert(tiledb::Query::Status::COMPLETE == query.query_status());

    Base::operator=
        (Base{std::move(row_ptr), std::move(col_ptr), std::move(val_ptr)});
  }
};

#endif  // TILEDB_COO_H
