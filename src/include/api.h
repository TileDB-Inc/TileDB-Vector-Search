/**
 * @file   api.h
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
 * Nascent C++ API.
 *
 */

#ifndef TDB_API_H
#define TDB_API_H

#include <memory>
#include <vector>

#include "concepts.h"
#include "cpos.h"
#include "index.h"
#include "detail/linalg/tdb_vector.h"

#include "utils/print_types.h"

//------------------------------------------------------------------------------
// Type erasure in two parts: a generic type erased wrapper and a specific
// type erased wrapper for feature vectors arrays.
//------------------------------------------------------------------------------
/**
 * Basic wrapper to type-erase a class.  It has a shared pointer to
 * a non-template base class that has virtual methods for accessing member
 * functions of the wrapped type.  It provides data(), dimension(), and
 * num_vectors() methods that delegate to the wrapped type.
 *
 * @todo Instrument carefully to make sure we are not doing any copying.
 */
class FeatureVectorArrayWrapper {
 public:
  template <typename T>
  explicit FeatureVectorArrayWrapper(T&& obj)
      : vector_array(std::make_unique<vector_array_impl<T>>(
            std::move(std::forward<T>(obj)))) {
  }

  [[nodiscard]] auto data() const {
    // return _cpo::data(*vector_array);
    return vector_array->data();
  }

  [[nodiscard]] auto dimension() const {
    return _cpo::dimension(*vector_array);
  }

  [[nodiscard]] auto num_vectors() const {
    return _cpo::num_vectors(*vector_array);
  }

  struct vector_array_base {
    virtual ~vector_array_base() = default;
    [[nodiscard]] virtual size_t dimension() const = 0;
    [[nodiscard]] virtual size_t num_vectors() const = 0;
    [[nodiscard]] virtual void* data() = 0;
    [[nodiscard]] virtual const void* data() const = 0;
  };

  // @todo Create move constructors for Matrix and tdbMatrix
  template <typename T>
  struct vector_array_impl : vector_array_base {
    explicit vector_array_impl(T&& t)
        : vector_array(std::move(t)) {
    }
    [[nodiscard]] void* data() override {
      return _cpo::data(vector_array);
      // return vector_array.data();
    }
    [[nodiscard]] const void* data() const override {
      return _cpo::data(vector_array);
      // return vector_array.data();
    }
    [[nodiscard]] size_t dimension() const override {
      return _cpo::dimension(vector_array);
    }
    [[nodiscard]] size_t num_vectors() const override {
      return _cpo::num_vectors(vector_array);
    }

   private:
    T vector_array;
  };

  std::unique_ptr<const vector_array_base> vector_array;
};

// Put these here for now to enforce separation between matrix aware
// and matrix unaware code.  We could (and probably should) merge these
// into a single class.  Although we could use the above for
// wrapping feature vectors as well (we would just use a concept
// to elide num_vectors).  The separation may also be useful for
// dealing with in-memory arrays vs arrays on disk.

#include <tiledb/tiledb>
#include "detail/linalg/tdb_helpers.h"
#include "detail/linalg/tdb_matrix.h"
#include "detail/linalg/tdb_vector.h"

class ProtoFeatureVectorArray {
 public:
  FeatureVectorArrayWrapper open(
      const tiledb::Context& ctx, const std::string& uri) {
    auto array = tiledb_helpers::open_array(tdb_func__, ctx, uri, TILEDB_READ);
    auto schema = array.schema();
    auto attr = schema.attribute(0);  // @todo Is there a better way to look up
                                      // the attribute we're interested in?
    datatype_ = attr.type();

    array.close();  // @todo create Matrix constructor that takes opened array

    /**
     * Row and column orientation are kind of irrelevant?  We could dispatch
     * on the layout in the schema, but that might not be necessary.  What is
     * important is that the vectors are along the major axis, which should
     * happen with either orientation, and so will work at the other end with
     * either orientation since we are just passing a pointer to the data.
     */
    switch (datatype_) {
      case TILEDB_FLOAT32:
        return FeatureVectorArrayWrapper(tdbColMajorMatrix<float>(ctx, uri));
      case TILEDB_UINT8:
        return FeatureVectorArrayWrapper(tdbColMajorMatrix<uint8_t>(ctx, uri));
      default:
        throw std::runtime_error("Unsupported attribute type");
    }
  }

  ProtoFeatureVectorArray(const tiledb::Context& ctx, const std::string& uri)
      : vector_array{open(ctx, uri)} {
  }

  [[nodiscard]] tiledb_datatype_t datatype() const {
    return datatype_;
  }

  [[nodiscard]] void* data() const {
    return (void*)_cpo::data(vector_array);
  }

  [[nodiscard]] auto dimension() const {
    return _cpo::dimension(vector_array);
  }

  [[nodiscard]] auto num_vectors() const {
    return _cpo::dimension(vector_array);
  }

  tiledb_datatype_t datatype_;
  FeatureVectorArrayWrapper vector_array;
};

namespace {
auto get_array_datatype(const tiledb::Array& array) {
  auto schema = array.schema();
  auto num_attributes = schema.attribute_num();
  if (num_attributes == 1) {
    return schema.attribute(0).type();
  }
  if (schema.has_attribute("values")) {
    return schema.attribute("values").type();
  }
  if (schema.has_attribute("a")) {
    return schema.attribute("a").type();
  }
  throw std::runtime_error("Could not determine datatype of array attributes");
}
}  // namespace

/**
 * Unified wrapper for feature vector arrays.
 */
class FeatureVector {
 public:
  template <feature_vector T>
  FeatureVector(T&& vec)
      : vector_(std::make_unique<vector_impl<T>>(std::forward<T>(vec))) {
  }

  FeatureVector(const tiledb::Context& ctx, const std::string& uri) {
    auto array = tiledb_helpers::open_array(tdb_func__, ctx, uri, TILEDB_READ);
    datatype_ = get_array_datatype(array);
    array.close();  // @todo create Matrix constructor that takes opened array

    /**
     * Row and column orientation are kind of irrelevant?  We could dispatch
     * on the layout in the schema, but that might not be necessary.  What is
     * important is that the vectors are along the major axis, which should
     * happen with either orientation, and so will work at the other end with
     * either orientation since we are just passing a pointer to the data.
     */
    switch (datatype_) {
      case TILEDB_FLOAT32:
        vector_ = std::make_unique<vector_impl<tdbVector<float>>>(ctx, uri);
        break;
      case TILEDB_UINT8:
        vector_ = std::make_unique<vector_impl<tdbVector<uint8_t>>>(ctx, uri);
        break;
      case TILEDB_INT32:
        vector_ = std::make_unique<vector_impl<tdbVector<int32_t>>>(ctx, uri);
        break;
      case TILEDB_UINT32:
        vector_ = std::make_unique<vector_impl<tdbVector<uint32_t>>>(ctx, uri);
        break;
      case TILEDB_UINT64:
        vector_ = std::make_unique<vector_impl<tdbVector<uint64_t>>>(ctx, uri);
        break;
      default:
        throw std::runtime_error("Unsupported attribute type");
    }
  }

  [[nodiscard]] auto data() const {
    // return _cpo::data(*vector);
    return vector_->data();
  }

  [[nodiscard]] auto dimension() const {
    return _cpo::dimension(*vector_);
  }

  [[nodiscard]] tiledb_datatype_t datatype() const {
    return datatype_;
  }

  /**
   * Non-type parameterized base class (for type erasure).
   */
  struct vector_base {
    virtual ~vector_base() = default;
    [[nodiscard]] virtual size_t dimension() const = 0;
    [[nodiscard]] virtual void* data() = 0;
    [[nodiscard]] virtual const void* data() const = 0;
  };

  // @todo Create move constructors for Matrix and tdbMatrix?
  template <typename T>
  struct vector_impl : vector_base {
    explicit vector_impl(T&& t)
        : vector_(std::move(t)) {
    }
    vector_impl(const tiledb::Context& ctx, const std::string& uri)
        : vector_(ctx, uri) {
    }
    [[nodiscard]] void* data() override {
      return _cpo::data(vector_);
      // return vector_.data();
    }
    [[nodiscard]] const void* data() const override {
      return _cpo::data(vector_);
      // return vector_.data();
    }
    [[nodiscard]] size_t dimension() const override {
      return _cpo::dimension(vector_);
    }

   private:
    T vector_;
  };

 private:
  tiledb_datatype_t datatype_{TILEDB_ANY};
  std::unique_ptr<const vector_base> vector_;
};

using QueryVector = FeatureVector;
using IdVector = FeatureVector;

/**
 * Unified wrapper for feature vector arrays.
 * @todo Lots of duplicated code from FeatureVector.  Can we factor this out?
 */
class FeatureVectorArray {
 public:
  template <feature_vector_array T>
  explicit FeatureVectorArray(T&& obj)
      : vector_array(std::make_unique<vector_array_impl<T>>(
            std::move(std::forward<T>(obj)))) {
  }

  FeatureVectorArray(const tiledb::Context& ctx, const std::string& uri) {
    auto array = tiledb_helpers::open_array(tdb_func__, ctx, uri, TILEDB_READ);
    datatype_ = get_array_datatype(array);
    array.close();  // @todo create Matrix constructor that takes opened array

    /**
     * Row and column orientation are kind of irrelevant?  We could dispatch
     * on the layout in the schema, but that might not be necessary.  What is
     * important is that the vectors are along the major axis, which should
     * happen with either orientation, and so will work at the other end with
     * either orientation since we are just passing a pointer to the data.
     */
    switch (datatype_) {
      case TILEDB_FLOAT32:
        vector_array =
            std::make_unique<vector_array_impl<tdbColMajorMatrix<float>>>(
                ctx, uri);
        break;
      case TILEDB_UINT8:
        vector_array =
            std::make_unique<vector_array_impl<tdbColMajorMatrix<uint8_t>>>(
                ctx, uri);
        break;
      default:
        throw std::runtime_error("Unsupported attribute type");
    }
  }

  // @todo fix so niebloids work
  [[nodiscard]] auto data() const {
    // return _cpo::data(*vector_array);
    return vector_array->data();
  }

  [[nodiscard]] auto extents() const {
    return _cpo::extents(*vector_array);
  }

  [[nodiscard]] auto dimension() const {
    return _cpo::dimension(*vector_array);
  }

  [[nodiscard]] auto num_vectors() const {
    return _cpo::num_vectors(*vector_array);
  }

  [[nodiscard]] tiledb_datatype_t datatype() const {
    return datatype_;
  }

  /**
   * Non-type parameterized base class (for type erasure).
   */
  struct vector_array_base {
    virtual ~vector_array_base() = default;
    [[nodiscard]] virtual size_t dimension() const = 0;
    [[nodiscard]] virtual size_t num_vectors() const = 0;
    [[nodiscard]] virtual void* data() const = 0;
    [[nodiscard]] virtual std::vector<size_t> extents() const = 0;
  };

  // @todo Create move constructors for Matrix and tdbMatrix?
  template <typename T>
  struct vector_array_impl : vector_array_base {
    explicit vector_array_impl(T&& t)
        : vector_array(std::move(t)) {
    }
    vector_array_impl(const tiledb::Context& ctx, const std::string& uri)
        : vector_array(ctx, uri) {
    }
    [[nodiscard]] void* data() const override {
      return _cpo::data(vector_array);
    }
    [[nodiscard]] size_t dimension() const override {
      return _cpo::dimension(vector_array);
    }
    [[nodiscard]] size_t num_vectors() const override {
      return _cpo::num_vectors(vector_array);
    }
    [[nodiscard]] std::vector<size_t> extents() const override {
      return _cpo::extents(vector_array);
    }

   private:
    T vector_array;
  };

 private:
  tiledb_datatype_t datatype_{TILEDB_ANY};
  std::unique_ptr<const vector_array_base> vector_array;
};

using QueryVectorArray = FeatureVectorArray;


//------------------------------------------------------------------------------
// Index
//------------------------------------------------------------------------------

// Some fake types and type aliases for now
using URI = std::string;
using StringMap = std::map<std::string, std::string>;

using IndexOptions = std::map<std::string, std::string>;
using UpdateOptions = std::map<std::string, std::string>;

// @todo Context?
class Index {

 public:
  Index(const URI& index_uri, std::optional<StringMap> config = std::nullopt)
      : Index(tiledb::Context{}, index_uri, config) {}

  Index(const tiledb::Context ctx, const URI& index_uri, std::optional<StringMap> config = std::nullopt)
  : ctx_{ctx} {
    auto array = tiledb_helpers::open_array(tdb_func__, ctx_, index_uri, TILEDB_READ);
    datatype_ = get_array_datatype(array);
    array.close();

    //@todo Constructors that take open arrays
    switch(datatype_) {
      case TILEDB_FLOAT32:
        index_ = std::make_unique<index_impl<flat_index<float>>>(ctx_, index_uri, config);
        break;
      case TILEDB_UINT8:
        index_ = std::make_unique<index_impl<flat_index<uint8_t>>>(ctx_, index_uri, config);
        break;
      default:
        throw std::runtime_error("Unsupported attribute type");
    }
  }

  template <feature_vector_array V>
  Index(
      const URI& index_uri,
      const V& vectors,
      const IndexOptions& options,
      std::optional<StringMap> config = std::nullopt) {
    // @todo
  }

  // Create from input URI
  Index(
      const URI& index_uri,
      const URI& vectors_uri,
      const IndexOptions& options,
      std::optional<StringMap> config = std::nullopt) {
    // @todo
  }

  auto query(const QueryVectorArray& vectors, size_t top_k) const {
    return index_->query(vectors, top_k);
  }

  void update(
      const FeatureVectorArray& vectors,
      const std::optional<IdVector>& ids = std::nullopt,
      const std::optional<UpdateOptions>& options = std::nullopt) const {
    index_->update(vectors, ids, options);
  }

  void update(
      URI vectors_uri,
      const std::optional<IdVector>& ids = std::nullopt,
      const std::optional<UpdateOptions>& options = std::nullopt) const {
    index_->update(vectors_uri, ids, options);
  }

  virtual void remove(const IdVector& ids) const {
    index_->remove(ids);
  }

  struct index_base {
    virtual ~index_base() = default;

    virtual std::tuple<FeatureVectorArray, FeatureVectorArray> query(
        const QueryVectorArray& vectors, size_t top_k) const = 0;

    virtual void update(
        const FeatureVectorArray&,
        const std::optional<IdVector>& ids = std::nullopt,
        const std::optional<UpdateOptions>& options = std::nullopt) const = 0;

    virtual void update(
        URI vectors_uri,
        const std::optional<IdVector>& ids = std::nullopt,
        const std::optional<UpdateOptions>& options = std::nullopt) const = 0;

    virtual void remove(const IdVector& ids) const = 0;

  };

  template <typename T>
  struct index_impl : index_base {
    explicit index_impl(T&& t)
        : index_(std::move(t)) {
    }

    index_impl(
        const tiledb::Context& ctx,
        const URI& index_uri,
        const std::optional<StringMap>& config = std::nullopt)
        : index_(ctx, index_uri, config) {
    }

    template <feature_vector_array V>
    index_impl(
        const URI& index_uri,
        const V& vectors,
        const IndexOptions& options,
        const std::optional<StringMap>& config = std::nullopt)
        : index_(index_uri, vectors, options, config) {
    }

    // Create from input URI
    index_impl(
        const URI& index_uri,
        const URI& vectors_uri,
        const IndexOptions& options,
        std::optional<StringMap> config = std::nullopt)
        : index_(index_uri, vectors_uri, options, config) {
    }

    auto query(tiledb::Context ctx, const URI& uri, size_t top_k) const {
      return index_.query(ctx, uri, top_k);
    }

    std::tuple<FeatureVectorArray, FeatureVectorArray>
    query(const QueryVectorArray& vectors, size_t k_nn) const {
      using index_type = size_t; // @todo Parameterize?

      auto dtype = vectors.datatype();

      switch (dtype) {
        case TILEDB_FLOAT32: {
          // auto qspan = MatrixView{stdx::mdspan<float, matrix_extents<index_type>, stdx::layout_left>}
          auto qspan = MatrixView{
              (float*)vectors.data(),
              extents(vectors)[0],
              extents(vectors)[1]};  // @todo ??
          auto [s, t] = index_.query(qspan, k_nn);
          auto x = FeatureVectorArray{std::move(s)};
          auto y = FeatureVectorArray{std::move(t)};
          //return std::make_tuple( std::move(x), std::move(y) );
          return { std::move(x), std::move(y) };
        }
        case TILEDB_UINT8: {
          auto qspan = MatrixView{
              (float*)vectors.data(),
              extents(vectors)[0],
              extents(vectors)[1]};  // @todo ??
          auto [s, t] = index_.query(qspan, k_nn);
          auto x = FeatureVectorArray{std::move(s)};
          auto y = FeatureVectorArray{std::move(t)};
          //return std::make_tuple( std::move(x), std::move(y) );
          return { std::move(x), std::move(y) };
        }
        default:
          throw std::runtime_error("Unsupported attribute type");
      }
    }

    void update(
        const FeatureVectorArray& vectors,
        const std::optional<IdVector>& ids = std::nullopt,
        const std::optional<UpdateOptions>& options = std::nullopt) const {
//      index_.update(vectors, ids, options);
    }

    void update(
        URI vectors_uri,
        const std::optional<IdVector>& ids = std::nullopt,
        const std::optional<UpdateOptions>& options = std::nullopt) const {
//      index_.update(vectors_uri, ids, options);
    }

    virtual void remove(const IdVector& ids) const {
//      index_.remove(ids);
    }

   private:
    T index_;
  };

  tiledb::Context ctx_{};
  tiledb_datatype_t datatype_{TILEDB_ANY};
  std::unique_ptr<const index_base> index_;
};

#endif
