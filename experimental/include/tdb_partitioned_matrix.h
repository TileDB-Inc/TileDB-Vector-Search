


#ifndef TILEDB_PARTITIONED_MATRIX_H
#define TILEDB_PARTITIONED_MATRIX_H
#include <cstddef>
#include <future>
#include <memory>
#include <span>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#include <tiledb/tiledb>
#include "mdspan/mdspan.hpp"

#include "linalg.h"

#include "array_types.h"
#include "utils/timer.h"

namespace stdx {
using namespace Kokkos;
using namespace Kokkos::Experimental;
}  // namespace stdx

extern bool global_verbose;
extern bool global_debug;
extern std::string global_region;


template <class T, class LayoutPolicy = stdx::layout_right, class I = size_t>
class tdbPartitionedMatrix : public Matrix<T, LayoutPolicy, I> {

  /****************************************************************************
   *
   * Duplication from tdbMatrix
   *
   * @todo Unify duplicated code
   *
   ****************************************************************************/
  using Base = Matrix<T, LayoutPolicy, I>;
  using Base::Base;

  using typename Base::index_type;
  using typename Base::size_type;
  using typename Base::reference;

  using view_type = Base;

  constexpr static auto matrix_order_{order_v<LayoutPolicy>};

 private:
  using row_domain_type = int32_t;
  using col_domain_type = int32_t;


  // @todo: Make this configurable
  std::map<std::string, std::string> init_{
      {"vfs.s3.region", global_region.c_str()}};
  tiledb::Config config_{init_};
  tiledb::Context ctx_{config_};

  tiledb::Array array_;
  tiledb::ArraySchema schema_;
  std::unique_ptr<T[]> backing_data_;
  size_t num_array_rows_{0};
  size_t num_array_cols_{0};

  std::tuple<index_type, index_type> row_view_;
  std::tuple<index_type, index_type> col_view_;
  index_type row_offset_{0};
  index_type col_offset_{0};

  std::future<bool> fut_;
  size_t pending_row_offset{0};
  size_t pending_col_offset{0};

  /****************************************************************************
   *
   * Stuff for partitioned (reshuffled) matrix
   *
   * @todo This needs to go into its own class
   *
   ****************************************************************************/
  tiledb::Array ids_array_;
  tiledb::ArraySchema ids_schema_;
  std::vector<shuffled_ids_type> indices_;  // @todo pointer and span?
  std::vector<parts_type> parts_;           // @todo pointer and span?
  std::vector<shuffled_ids_type> ids_;      // @todo pointer and span?
  std::tuple<index_type, index_type> row_part_view_;
  std::tuple<index_type, index_type> col_part_view_;

 public:

  tdbPartitionedMatrix(
      const std::string& uri,
      std::vector<indices_type>& indices,
      const std::vector<parts_type>& parts,
      const std::string& id_uri,
      std::vector<shuffled_ids_type>& shuffled_ids,
      size_t nthreads) : tdbPartitionedMatrix(uri, indices, parts, id_uri, shuffled_ids, 0, nthreads) {}

  /**
   * Gather pieces of a partitioned array into a single array (along with the
   * vector ids into a corresponding 1D array)
   *
   * @todo Column major is kind of baked in here.  Need to generalize.
   */
  tdbPartitionedMatrix(
      const std::string& uri,
      std::vector<indices_type>& indices,
      const std::vector<parts_type>& parts,
      const std::string& ids_uri,
      std::vector<shuffled_ids_type>& shuffled_ids,
      size_t upper_bound,
      size_t nthreads)
      : array_{ctx_, uri, TILEDB_READ}
      , schema_{array_.schema()}
      , ids_array_{ctx_, ids_uri, TILEDB_READ}
      , ids_schema_{ids_array_.schema()}
      , row_part_view_{0, 0}
      , col_part_view_{0, 0} {

    size_t total_num_parts = size(parts);
    size_t num_cols = 0;

    {
      life_timer _{"read partitioned matrix " + uri};

      auto cell_order = schema_.cell_order();
      auto tile_order = schema_.tile_order();

      // @todo Maybe throw an exception here?  Have to properly handle since
      // this is a constructor.
      assert(cell_order == tile_order);

      const size_t attr_idx = 0;

      auto attr_num{schema_.attribute_num()};
      auto attr = schema_.attribute(attr_idx);

      std::string attr_name = attr.name();
      tiledb_datatype_t attr_type = attr.type();
      if (attr_type != tiledb::impl::type_to_tiledb<T>::tiledb_type) {
        throw std::runtime_error(
            "Attribute type mismatch: " + std::to_string(attr_type) + " != " +
            std::to_string(tiledb::impl::type_to_tiledb<T>::tiledb_type));
      }

      auto domain_{schema_.domain()};

      auto array_rows_{domain_.dimension(0)};
      auto array_cols_{domain_.dimension(1)};

      num_array_rows_ =
          (array_rows_.template domain<row_domain_type>().second -
           array_rows_.template domain<row_domain_type>().first + 1);
      num_array_cols_ =
          (array_cols_.template domain<col_domain_type>().second -
           array_cols_.template domain<col_domain_type>().first + 1);

      if ((matrix_order_ == TILEDB_ROW_MAJOR &&
           cell_order == TILEDB_COL_MAJOR) ||
          (matrix_order_ == TILEDB_COL_MAJOR &&
           cell_order == TILEDB_ROW_MAJOR)) {
        throw std::runtime_error("Cell order and matrix order must match");
      }

      size_t dimension = num_array_rows_;

      if (indices[size(indices) - 1] == indices[size(indices) - 2]) {
        if (indices[size(indices) - 1] > num_array_cols_) {
          throw std::runtime_error("Indices are not valid");
        }
        indices[size(indices) - 1] = num_array_cols_;
      }

      for (size_t i = 0; i < total_num_parts; ++i) {
        num_cols += indices[parts[i] + 1] - indices[parts[i]];
      }

#ifndef __APPLE__
      auto data_ = std::make_unique_for_overwrite<T[]>(dimension * num_cols);
#else
      auto data_ = std::unique_ptr<T[]>(new T[dimension * num_cols]);
#endif

      /**
       * Read in the partitions
       */
      tiledb::Subarray subarray(ctx_, this->array_);

      // Dimension 0 goes from 0 to 127
      subarray.add_range(0, 0, (int)dimension - 1);

      for (size_t j = 0; j < total_num_parts; ++j) {
        size_t start = indices[parts[j]];
        size_t stop = indices[parts[j] + 1];
        size_t len = stop - start;
        if (len == 0) {
          continue;
        }
        size_t num_elements = len * dimension;
        subarray.add_range(1, (int)start, (int)stop - 1);
      }
      auto layout_order = cell_order;

      tiledb::Query query(ctx_, this->array_);

      auto ptr = data_.get();
      query.set_subarray(subarray)
          .set_layout(layout_order)
          .set_data_buffer(attr_name, ptr, num_cols * dimension);
      query.submit();

      // assert(tiledb::Query::Status::COMPLETE == query.query_status());
      if (tiledb::Query::Status::COMPLETE != query.query_status()) {
        throw std::runtime_error("Query status is not complete -- fix me");
      }
      Base::operator=(Base{std::move(data_), dimension, num_cols});
    }

    auto part_ids = std::vector<uint64_t>(num_cols);

    {
      life_timer _{"read partitioned vector" + ids_uri};
      /**
       * Now deal with ids
       */
      auto attr_idx = 0;

      auto ids_array_ = tiledb::Array{ctx_, ids_uri, TILEDB_READ};
      auto ids_schema_ = ids_array_.schema();

      auto attr_num{ids_schema_.attribute_num()};
      auto attr = ids_schema_.attribute(attr_idx);
      std::string attr_name = attr.name();

      tiledb::Subarray subarray(ctx_, ids_array_);

      for (size_t j = 0; j < total_num_parts; ++j) {
        size_t start = indices[parts[j]];
        size_t stop = indices[parts[j] + 1];
        size_t len = stop - start;
        if (len == 0) {
          continue;
        }
        size_t num_elements = len;
        subarray.add_range(0, (int)start, (int)stop - 1);
      }
      tiledb::Query query(ctx_, ids_array_);

      auto ptr = part_ids.data();
      query.set_subarray(subarray)
          .set_data_buffer(attr_name, ptr, num_cols);
      query.submit();

      // assert(tiledb::Query::Status::COMPLETE == query.query_status());
      if (tiledb::Query::Status::COMPLETE != query.query_status()) {
        throw std::runtime_error("Query status is not complete -- fix me");
      }
      ids_array_.close();
    }
    shuffled_ids = std::move(part_ids);
  }

};

/**
 * Convenience class for row-major matrices.
 */
template <class T, class I = size_t>
using tdbRowMajorPartitionedMatrix = tdbPartitionedMatrix<T, stdx::layout_right, I>;

/**
 * Convenience class for column-major matrices.
 */
template <class T, class I = size_t>
using tdbColMajorPartitionedMatrix = tdbPartitionedMatrix<T, stdx::layout_left, I>;



#endif //TILEDB_PARTITIONED_MATRIX_H