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
 * Nascent C++ API.  Type-erased classes are provided here as an interface
 * between the C++ vector search library and the Python bindings.
 *
 * Type erasure is accomplished with the following pattern.
 *   - The type-erased class (the outer class) provides the API that is invoked
 * by Python
 *   - It defines an abstract base class that is not a template and a derived
 * implementation class that is a template.
 *   - Constructors for the outer class use information from how they are
 * constructed (perhaps from reading the schema of an array) to determine the
 * type of the implementation class. The unique_ptr member of the outer class is
 * constructed with the derived implementation class.
 *   - The member functions comprising the outer class API invoke the
 * corresponding member functions of the base class object stored in the
 * unique_ptr (which in turn invoke members of the concrete class stored by the
 * implementation class).
 *
 */

#ifndef TDB_API_H
#define TDB_API_H

#include <memory>
#include <vector>

#include "concepts.h"
#include "cpos.h"
#include "detail/linalg/tdb_vector.h"
#include "flat_index.h"
#include "index_defs.h"
#include "ivf_index.h"

#include "utils/print_types.h"

#include <tiledb/tiledb>
#include "detail/linalg/tdb_helpers.h"
#include "detail/linalg/tdb_matrix.h"
#include "detail/linalg/tdb_vector.h"

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

//------------------------------------------------------------------------------
// FeatureVector
//------------------------------------------------------------------------------

/**
 * @brief Outer class defining the API for feature vectors.
 */
class FeatureVector {
 public:
  /**
   * @brief Construct from a class meeting the requirements of feature_vector.
   * @tparam T
   * @param vec
   */
  template <feature_vector T>
  explicit FeatureVector(T&& vec)
      : vector_(std::make_unique<vector_impl<T>>(std::forward<T>(vec))) {
    feature_type_ = tiledb::impl::type_to_tiledb<
        typename std::remove_cvref_t<T>::value_type>::tiledb_type;
  }

  /**
   * @brief Constructs a feature vector from an array URI.
   * @param ctx
   * @param uri
   */
  FeatureVector(const tiledb::Context& ctx, const std::string& uri) {
    auto array = tiledb_helpers::open_array(tdb_func__, ctx, uri, TILEDB_READ);
    feature_type_ = get_array_datatype(array);
    array.close();  // @todo create Matrix constructor that takes opened array

    /*
     * Dispatch to the appropriate concrete class based on the datatype.
     */
    switch (feature_type_) {
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

  /**
   * @brief Returns a pointer to the underlying data.
   * @return
   */
  [[nodiscard]] auto data() const {
    // return _cpo::data(*vector);
    return vector_->data();
  }

  /**
   * @brief Returns the dimension (number of elements) of the vector
   * @return
   */
  [[nodiscard]] auto dimension() const {
    return _cpo::dimension(*vector_);
  }

  /**
   * @brief Returns the TileDB datatype of the vector
   * @return
   */
  [[nodiscard]] tiledb_datatype_t feature_type() const {
    return feature_type_;
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

  /**
   * @brief Type-parameterize implementation class.
   * @tparam T Type of the concrete class that is being type-erased.
   */
  template <typename T>
  struct vector_impl : vector_base {
    explicit vector_impl(T&& t)
        : vector_(std::forward<T>(t)) {
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
  tiledb_datatype_t feature_type_{TILEDB_ANY};
  std::unique_ptr<const vector_base> vector_;
};

using QueryVector = FeatureVector;
using IdVector = FeatureVector;

//------------------------------------------------------------------------------
// FeatureVectorArray
//------------------------------------------------------------------------------

class FeatureVectorArray {
 public:
  FeatureVectorArray(const FeatureVectorArray& other) = delete;
  FeatureVectorArray& operator=(const FeatureVectorArray& other) = delete;
  FeatureVectorArray() = delete;
  virtual ~FeatureVectorArray() = default;

  FeatureVectorArray(FeatureVectorArray&& other) = default;
  FeatureVectorArray& operator=(FeatureVectorArray&& other) = default;

  template <feature_vector_array T>
  explicit FeatureVectorArray(T&& obj)
      : vector_array(
            std::make_unique<vector_array_impl<T>>(std::forward<T>(obj))) {
    feature_type_ = tiledb::impl::type_to_tiledb<
        typename std::remove_cvref_t<T>::value_type>::tiledb_type;
  }

  FeatureVectorArray(
      const tiledb::Context& ctx,
      const std::string& uri,
      size_t num_vectors = 0) {
    auto array = tiledb_helpers::open_array(tdb_func__, ctx, uri, TILEDB_READ);
    feature_type_ = get_array_datatype(array);
    array.close();  // @todo create Matrix constructor that takes opened array

    /**
     * Row and column orientation are kind of irrelevant?  We could dispatch
     * on the layout in the schema, but that might not be necessary.  What is
     * important is that the vectors are along the major axis, which should
     * happen with either orientation, and so will work at the other end with
     * either orientation since we are just passing a pointer to the data.
     */
    switch (feature_type_) {
      case TILEDB_FLOAT32:
        vector_array =
            std::make_unique<vector_array_impl<tdbColMajorMatrix<float>>>(
                ctx, uri, num_vectors);
        break;
      case TILEDB_UINT8:
        vector_array =
            std::make_unique<vector_array_impl<tdbColMajorMatrix<uint8_t>>>(
                ctx, uri, num_vectors);
        break;
      case TILEDB_INT32:
        vector_array =
            std::make_unique<vector_array_impl<tdbColMajorMatrix<int32_t>>>(
                ctx, uri, num_vectors);
        break;
      case TILEDB_UINT32:
        vector_array =
            std::make_unique<vector_array_impl<tdbColMajorMatrix<uint32_t>>>(
                ctx, uri, num_vectors);
        break;
      case TILEDB_INT64:
        vector_array =
            std::make_unique<vector_array_impl<tdbColMajorMatrix<int64_t>>>(
                ctx, uri, num_vectors);
        break;
      case TILEDB_UINT64:
        vector_array =
            std::make_unique<vector_array_impl<tdbColMajorMatrix<uint64_t>>>(
                ctx, uri, num_vectors);
        break;
      default:
        throw std::runtime_error("Unsupported attribute type");
    }
  }

  auto load() const {
    // return _cpo::load(*vector_array);
    return vector_array->load();
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

  [[nodiscard]] tiledb_datatype_t feature_type() const {
    return feature_type_;
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
    [[nodiscard]] virtual bool load() = 0;
  };

  // @todo Create move constructors for Matrix and tdbMatrix?
  template <typename T>
  struct vector_array_impl : vector_array_base {
    explicit vector_array_impl(T&& t)
        : impl_vector_array(std::forward<T>(t)) {
    }
    vector_array_impl(
        const tiledb::Context& ctx, const std::string& uri, size_t num_vectors)
        : impl_vector_array(ctx, uri, num_vectors) {
    }
    [[nodiscard]] void* data() const override {
      return _cpo::data(impl_vector_array);
    }
    [[nodiscard]] size_t dimension() const override {
      return _cpo::dimension(impl_vector_array);
    }
    [[nodiscard]] size_t num_vectors() const override {
      return _cpo::num_vectors(impl_vector_array);
    }
    [[nodiscard]] std::vector<size_t> extents() const override {
      return _cpo::extents(impl_vector_array);
    }
    bool load() override {
      return _cpo::load(impl_vector_array);
    }

   private:
    T impl_vector_array;
  };

 private:
  tiledb_datatype_t feature_type_{TILEDB_ANY};

  // @todo const????
  std::unique_ptr</*const*/ vector_array_base> vector_array;
};

using QueryVectorArray = FeatureVectorArray;

//------------------------------------------------------------------------------
// Type-erased index classes
//------------------------------------------------------------------------------

// Some fake types and type aliases for now
using URI = std::string;
using StringMap = std::map<std::string, std::string>;

using IndexOptions = std::map<std::string, std::string>;
using UpdateOptions = std::map<std::string, std::string>;

/**
 * A type-erased index class. An index class is provides
 *   - URI-based constructor
 *   - Array-based constructor
 *   - A train method
 *   - An add method
 *   - A query method
 *   - An update method
 *   - A remove method
 */
class IndexFlatL2 {
 public:
  // @todo Who owns the context?
  IndexFlatL2(
      const URI& index_uri,
      const std::optional<IndexOptions>& config = std::nullopt)
      : IndexFlatL2(tiledb::Context{}, index_uri, config) {
  }

  // @todo Use group metadata to determine index type and associated array types
  IndexFlatL2(
      const tiledb::Context& ctx,
      const URI& index_uri,
      const std::optional<IndexOptions>& config = std::nullopt)
      : ctx_{ctx} {
    auto array =
        tiledb_helpers::open_array(tdb_func__, ctx_, index_uri, TILEDB_READ);
    feature_type_ = get_array_datatype(array);
    array.close();

    switch (feature_type_) {
      case TILEDB_FLOAT32:
        index_ = std::make_unique<index_impl<flat_index<float>>>(
            ctx_, index_uri, config);
        break;
      case TILEDB_UINT8:
        index_ = std::make_unique<index_impl<flat_index<uint8_t>>>(
            ctx_, index_uri, config);
        break;
      default:
        throw std::runtime_error("Unsupported attribute type");
    }
  };

  template <feature_vector_array V>
  IndexFlatL2(
      const URI& index_uri,
      const V& vectors,
      const std::optional<IndexOptions>& config = std::nullopt) {
  }

  // Create from input URI
  IndexFlatL2(
      const URI& index_uri,
      const URI& vectors_uri,
      const std::optional<IndexOptions>& config = std::nullopt) {
    // @todo
  }

  void add() const {
    // @todo
  }

  void add_with_ids() const {
    // @todo
  }

  void train() const {
    // @todo
  }

  void save() const {
    // @todo
  }

  // todo query() or search() -- or both?
  [[nodiscard]] auto query(
      const QueryVectorArray& vectors, size_t top_k) const {
    return index_->query(vectors, top_k);
  }

  void update(
      const FeatureVectorArray& vectors,
      const std::optional<IdVector>& ids = std::nullopt,
      const std::optional<UpdateOptions>& options = std::nullopt) const {
    index_->update(vectors, ids, options);
  }

  void update(
      const URI& vectors_uri,
      const std::optional<IdVector>& ids = std::nullopt,
      const std::optional<UpdateOptions>& options = std::nullopt) const {
    index_->update(vectors_uri, ids, options);
  }

  virtual void remove(const IdVector& ids) const {
    index_->remove(ids);
  }

  auto dimension() {
    return _cpo::dimension(*index_);
  }

  size_t ntotal() const {
    // @todo
    return 0;
  }

  auto num_vectors() {
    return _cpo::num_vectors(*index_);
  }

  auto feature_type() {
    return feature_type_;
  }

  /**
   * Non-type parameterized base class (for type erasure).
   */
  struct index_base {
    virtual ~index_base() = default;

    [[nodiscard]] virtual std::tuple<FeatureVectorArray, FeatureVectorArray>
    query(const QueryVectorArray& vectors, size_t top_k) const = 0;

    virtual void update(
        const FeatureVectorArray&,
        const std::optional<IdVector>& ids,
        const std::optional<UpdateOptions>& options) const = 0;

    virtual void update(
        const URI& vectors_uri,
        const std::optional<IdVector>& ids,
        const std::optional<UpdateOptions>& options) const = 0;

    virtual void remove(const IdVector& ids) const = 0;

    virtual size_t dimension() const = 0;

    virtual size_t ntotal() const = 0;

    virtual size_t num_vectors() const = 0;
  };

  /**
   * @brief Type-parameterize implementation class.
   * @tparam T Type of the concrete class that is being type-erased.
   */
  template <typename T>
  struct index_impl : index_base {
    explicit index_impl(T&& t)
        : impl_index_(std::forward<T>(t)) {
    }

    index_impl(
        const tiledb::Context& ctx,
        const URI& index_uri,
        const std::optional<StringMap>& config = std::nullopt)
        : impl_index_(ctx, index_uri) {
    }

    template <feature_vector_array V>
    index_impl(
        const URI& index_uri,
        const V& vectors,
        const IndexOptions& options,
        const std::optional<StringMap>& config = std::nullopt)
        : impl_index_(index_uri, vectors, options, config) {
    }

    // Create from input URI
    index_impl(
        const URI& index_uri,
        const URI& vectors_uri,
        const IndexOptions& options,
        std::optional<StringMap> config = std::nullopt)
        : impl_index_(index_uri, vectors_uri, options, config) {
    }

    [[nodiscard]] auto query(
        tiledb::Context ctx, const URI& uri, size_t top_k) const {
      return impl_index_.query(ctx, uri, top_k);
    }

    /**
     * @brief Query the index with the given vectors.  The concrete query
     * function returns a tuple of arrays, which are type erased and returned as
     * a tuple of FeatureVectorArrays.
     * @param vectors
     * @param k_nn
     * @return
     *
     * @todo Make sure the extents of the returned arrays are used correctly.
     */
    [[nodiscard]] std::tuple<FeatureVectorArray, FeatureVectorArray> query(
        const QueryVectorArray& vectors, size_t k_nn) const override {
      // @todo using index_type = size_t;

      auto dtype = vectors.feature_type();

      // @note We need to maintain same layout -> or swap extents
      switch (dtype) {
        case TILEDB_FLOAT32: {
          auto qspan = MatrixView<float, stdx::layout_left>{
              (float*)vectors.data(),
              extents(vectors)[0],
              extents(vectors)[1]};  // @todo ??
          auto [s, t] = impl_index_.query(qspan, k_nn);
          debug_slice(t);
          auto& ss = s;
          auto& tt = t;
          auto x = FeatureVectorArray{std::move(s)};
          auto y = FeatureVectorArray{std::move(t)};
          return {std::move(x), std::move(y)};
        }
        case TILEDB_UINT8: {
          auto qspan = MatrixView<uint8_t, stdx::layout_left>{
              (uint8_t*)vectors.data(),
              extents(vectors)[0],
              extents(vectors)[1]};  // @todo ??
          auto [s, t] = impl_index_.query(qspan, k_nn);
          auto x = FeatureVectorArray{std::move(s)};
          auto y = FeatureVectorArray{std::move(t)};
          return {std::move(x), std::move(y)};
        }
        default:
          throw std::runtime_error("Unsupported attribute type");
      }
    }

    // WIP
    void update(
        const FeatureVectorArray& vectors,
        const std::optional<IdVector>& ids,
        const std::optional<UpdateOptions>& options) const override {
      //      index_.update(vectors, ids, options);
    }

    // WIP
    void update(
        const URI& vectors_uri,
        const std::optional<IdVector>& ids,
        const std::optional<UpdateOptions>& options) const override {
      //      index_.update(vectors_uri, ids, options);
    }

    // WIP
    void remove(const IdVector& ids) const override {
      //      index_.remove(ids);
    }

    size_t dimension() const override {
      return _cpo::dimension(impl_index_);
    }

    size_t ntotal() const override {
      return _cpo::num_vectors(impl_index_);
    }

    size_t num_vectors() const override {
      return _cpo::num_vectors(impl_index_);
    }

   private:
    /**
     * @brief Instance of the concrete class.
     */
    T impl_index_;
  };

  // @todo Who should own the context?
  tiledb::Context ctx_{};
  tiledb_datatype_t feature_type_{TILEDB_ANY};
  tiledb_datatype_t id_type_{TILEDB_ANY};
  tiledb_datatype_t ptx_type_{TILEDB_ANY};
  std::unique_ptr<const index_base> index_;
};

/*******************************************************************************
 * IndexFlatPQ
 ******************************************************************************/

/*******************************************************************************
 * IndexIVFFlat
 ******************************************************************************/
// OK -- this one weirded me out
/**
 * A type-erased index class. An index class is provides
 *   - URI-based constructor
 *   - Array-based constructor
 *   - A train method
 *   - An add method
 *   - A query method
 *   - An update method
 *   - A remove method
 */
class IndexIVFFlat {
 public:
  // @todo Who owns the context?
  IndexIVFFlat(
      const URI& group_uri,
      const std::optional<IndexOptions>& config = std::nullopt)
      : IndexIVFFlat(tiledb::Context{}, group_uri, config) {
  }

  /**
   * @brief Open an existing index.
   *
   * @note This will be able to infer all of its types using the group metadata
   * to create the internal ivf_index object.
   *
   * @param ctx
   * @param group_uri
   * @param config
   */
  IndexIVFFlat(
      const tiledb::Context& ctx,
      const URI& group_uri,
      const std::optional<IndexOptions>& config = std::nullopt)
      : ctx_{ctx} {
    using metadata_element = std::tuple<std::string, void*, tiledb_datatype_t>;
    std::vector<metadata_element> metadata{
        {"feature_datatype", &feature_datatype_, TILEDB_UINT32},
        {"id_datatype", &id_datatype_, TILEDB_UINT32},
        {"ptx_datatype", &ptx_datatype_, TILEDB_UINT32}};

    tiledb::Config cfg;
    tiledb::Group read_group(ctx_, group_uri, TILEDB_READ, cfg);

    for (auto& [name, value, datatype] : metadata) {
      if (!read_group.has_metadata(name, &datatype)) {
        throw std::runtime_error("Missing metadata: " + name);
      }
      uint32_t count;
      void* addr;
      read_group.get_metadata(name, &datatype, &count, (const void**)&addr);
      if (datatype == TILEDB_UINT32) {
        *reinterpret_cast<uint32_t*>(value) =
            *reinterpret_cast<uint32_t*>(addr);
      } else {
        throw std::runtime_error("Unsupported datatype for metadata: " + name);
      }
    }

    if (feature_datatype_ == TILEDB_UINT8 && id_datatype_ == TILEDB_UINT32 &&
        ptx_datatype_ == TILEDB_UINT32) {
      index_ =
          std::make_unique<index_impl<ivf_index<uint8_t, uint32_t, uint32_t>>>(
              ctx_, group_uri, config);
    } else if (
        feature_datatype_ == TILEDB_FLOAT32 && id_datatype_ == TILEDB_UINT32 &&
        ptx_datatype_ == TILEDB_UINT32) {
      index_ =
          std::make_unique<index_impl<ivf_index<float, uint32_t, uint32_t>>>(
              ctx_, group_uri, config);
    } else if (
        feature_datatype_ == TILEDB_UINT8 && id_datatype_ == TILEDB_UINT32 &&
        ptx_datatype_ == TILEDB_UINT64) {
      index_ =
          std::make_unique<index_impl<ivf_index<uint8_t, uint32_t, uint64_t>>>(
              ctx_, group_uri, config);
    } else if (
        feature_datatype_ == TILEDB_FLOAT32 && id_datatype_ == TILEDB_UINT32 &&
        ptx_datatype_ == TILEDB_UINT64) {
      index_ =
          std::make_unique<index_impl<ivf_index<float, uint32_t, uint64_t>>>(
              ctx_, group_uri, config);
    } else if (
        feature_datatype_ == TILEDB_UINT8 && id_datatype_ == TILEDB_UINT64 &&
        ptx_datatype_ == TILEDB_UINT32) {
      index_ =
          std::make_unique<index_impl<ivf_index<uint8_t, uint64_t, uint32_t>>>(
              ctx_, group_uri, config);
    } else if (
        feature_datatype_ == TILEDB_FLOAT32 && id_datatype_ == TILEDB_UINT64 &&
        ptx_datatype_ == TILEDB_UINT32) {
      index_ =
          std::make_unique<index_impl<ivf_index<float, uint64_t, uint32_t>>>(
              ctx_, group_uri, config);
    } else if (
        feature_datatype_ == TILEDB_UINT8 && id_datatype_ == TILEDB_UINT64 &&
        ptx_datatype_ == TILEDB_UINT64) {
      index_ =
          std::make_unique<index_impl<ivf_index<uint8_t, uint64_t, uint64_t>>>(
              ctx_, group_uri, config);
    } else if (
        feature_datatype_ == TILEDB_FLOAT32 && id_datatype_ == TILEDB_UINT64 &&
        ptx_datatype_ == TILEDB_UINT64) {
      index_ =
          std::make_unique<index_impl<ivf_index<float, uint64_t, uint64_t>>>(
              ctx_, group_uri, config);
    }
  }

  template <feature_vector_array V>
  IndexIVFFlat(
      const URI& group_uri,
      const V& vectors,
      const std::optional<IndexOptions>& config = std::nullopt) {
  }

  // Create from input URI
  IndexIVFFlat(
      const URI& group_uri,
      const URI& vectors_uri,
      const std::optional<IndexOptions>& config = std::nullopt) {
    // @todo
  }

  void add() const {
    // @todo
  }

  void add_with_ids() const {
    // @todo
  }

  void train() const {
    // @todo
  }

  void save() const {
    // @todo
  }

  // todo query() or search() -- or both?
  [[nodiscard]] auto query_infinite_ram(
      const QueryVectorArray& vectors, size_t top_k, size_t nprobe)  {
    return index_->query_infinite_ram(vectors, top_k, nprobe);
  }

  [[nodiscard]] auto query_finite_ram(
      const QueryVectorArray& vectors, size_t top_k, size_t nprobe)  {
    return index_->query_finite_ram(vectors, top_k, nprobe);
  }

  void update(
      const FeatureVectorArray& vectors,
      const std::optional<IdVector>& ids = std::nullopt,
      const std::optional<UpdateOptions>& options = std::nullopt) const {
    index_->update(vectors, ids, options);
  }

  void update(
      const URI& vectors_uri,
      const std::optional<IdVector>& ids = std::nullopt,
      const std::optional<UpdateOptions>& options = std::nullopt) const {
    index_->update(vectors_uri, ids, options);
  }

  virtual void remove(const IdVector& ids) const {
    index_->remove(ids);
  }

  auto dimension() {
    return _cpo::dimension(*index_);
  }

// Don't think we need thi
#if 0
  size_t ntotal() const {
    // @todo
    return 0;
  }

  auto num_vectors() {
    return _cpo::num_vectors(*index_);
  }
#endif

  auto feature_type() {
    return feature_datatype_;
  }

  /**
   * Non-type parameterized base class (for type erasure).
   */
  struct index_base {
    virtual ~index_base() = default;

    [[nodiscard]] virtual std::tuple<FeatureVectorArray, FeatureVectorArray>
    query_infinite_ram(const QueryVectorArray& vectors, size_t top_k, size_t nprobe)  = 0;

    [[nodiscard]] virtual std::tuple<FeatureVectorArray, FeatureVectorArray>
    query_finite_ram(const QueryVectorArray& vectors, size_t top_k, size_t nprobe)  = 0;

    virtual void update(
        const FeatureVectorArray&,
        const std::optional<IdVector>& ids,
        const std::optional<UpdateOptions>& options) const = 0;

    virtual void update(
        const URI& vectors_uri,
        const std::optional<IdVector>& ids,
        const std::optional<UpdateOptions>& options) const = 0;

    virtual void remove(const IdVector& ids) const = 0;

    virtual size_t dimension() const = 0;

// Don't think we need these
#if 0
    virtual size_t ntotal() const = 0;

    virtual size_t num_vectors() const = 0;
#endif
  };

  /**
   * @brief Type-parameterize implementation class.
   * @tparam T Type of the concrete class that is being type-erased.
   */
  template <typename T>
  struct index_impl : index_base {
    explicit index_impl(T&& t)
        : impl_index_(std::forward<T>(t)) {
    }

    index_impl(
        const tiledb::Context& ctx,
        const URI& index_uri,
        const std::optional<StringMap>& config = std::nullopt)
        : impl_index_(ctx, index_uri) {
    }

    template <feature_vector_array V>
    index_impl(
        const URI& index_uri,
        const V& vectors,
        const IndexOptions& options,
        const std::optional<StringMap>& config = std::nullopt)
        : impl_index_(index_uri, vectors, options, config) {
    }

    // Create from input URI
    index_impl(
        const URI& index_uri,
        const URI& vectors_uri,
        const IndexOptions& options,
        std::optional<StringMap> config = std::nullopt)
        : impl_index_(index_uri, vectors_uri, options, config) {
    }

    [[nodiscard]] auto query_infinite_ram(
        tiledb::Context ctx, const URI& uri, size_t top_k, size_t nprobe)  {
      return impl_index_.query_infinite_ram(ctx, uri, top_k, nprobe);
    }

    [[nodiscard]] auto query_finite_ram(
        tiledb::Context ctx, const URI& uri, size_t top_k, size_t nprobe)  {
      return impl_index_.query_finite_ram(ctx, uri, top_k, nprobe);
    }
    /**
     * @brief Query the index with the given vectors.  The concrete query
     * function returns a tuple of arrays, which are type erased and returned as
     * a tuple of FeatureVectorArrays.
     * @param vectors
     * @param k_nn
     * @return
     *
     * @todo Make sure the extents of the returned arrays are used correctly.
     */
    [[nodiscard]] std::tuple<FeatureVectorArray, FeatureVectorArray> query_infinite_ram(
        const QueryVectorArray& vectors, size_t k_nn, size_t nprobe)  override {
      // @todo using index_type = size_t;

      auto dtype = vectors.feature_type();

      // @note We need to maintain same layout -> or swap extents
      switch (dtype) {
        case TILEDB_FLOAT32: {
          auto qspan = MatrixView<float, stdx::layout_left>{
              (float*)vectors.data(),
              extents(vectors)[0],
              extents(vectors)[1]};  // @todo ??
          auto [s, t] = impl_index_.query_infinite_ram(qspan, k_nn, nprobe);
          debug_slice(t);
          auto& ss = s;
          auto& tt = t;
          auto x = FeatureVectorArray{std::move(s)};
          auto y = FeatureVectorArray{std::move(t)};
          return {std::move(x), std::move(y)};
        }
        case TILEDB_UINT8: {
          auto qspan = MatrixView<uint8_t, stdx::layout_left>{
              (uint8_t*)vectors.data(),
              extents(vectors)[0],
              extents(vectors)[1]};  // @todo ??
          auto [s, t] = impl_index_.query_infinite_ram(qspan, k_nn, nprobe);
          auto x = FeatureVectorArray{std::move(s)};
          auto y = FeatureVectorArray{std::move(t)};
          return {std::move(x), std::move(y)};
        }
        default:
          throw std::runtime_error("Unsupported attribute type");
      }
    }

    [[nodiscard]] std::tuple<FeatureVectorArray, FeatureVectorArray> query_finite_ram(
        const QueryVectorArray& vectors, size_t k_nn, size_t nprobe)  override {
      // @todo using index_type = size_t;

      auto dtype = vectors.feature_type();

      // @note We need to maintain same layout -> or swap extents
      switch (dtype) {
        case TILEDB_FLOAT32: {
          auto qspan = MatrixView<float, stdx::layout_left>{
              (float*)vectors.data(),
              extents(vectors)[0],
              extents(vectors)[1]};  // @todo ??
          auto [s, t] = impl_index_.query_finite_ram(qspan, k_nn, nprobe);
          debug_slice(t);
          auto& ss = s;
          auto& tt = t;
          auto x = FeatureVectorArray{std::move(s)};
          auto y = FeatureVectorArray{std::move(t)};
          return {std::move(x), std::move(y)};
        }
        case TILEDB_UINT8: {
          auto qspan = MatrixView<uint8_t, stdx::layout_left>{
              (uint8_t*)vectors.data(),
              extents(vectors)[0],
              extents(vectors)[1]};  // @todo ??
          auto [s, t] = impl_index_.query_finite_ram(qspan, k_nn, nprobe);
          auto x = FeatureVectorArray{std::move(s)};
          auto y = FeatureVectorArray{std::move(t)};
          return {std::move(x), std::move(y)};
        }
        default:
          throw std::runtime_error("Unsupported attribute type");
      }
    }

    // WIP
    void update(
        const FeatureVectorArray& vectors,
        const std::optional<IdVector>& ids,
        const std::optional<UpdateOptions>& options) const override {
      //      index_.update(vectors, ids, options);
    }

    // WIP
    void update(
        const URI& vectors_uri,
        const std::optional<IdVector>& ids,
        const std::optional<UpdateOptions>& options) const override {
      //      index_.update(vectors_uri, ids, options);
    }

    // WIP
    void remove(const IdVector& ids) const override {
      //      index_.remove(ids);
    }

    size_t dimension() const override {
      return ::dimension(impl_index_);
    }

    // Don't think we need these
#if 0
    size_t ntotal() const override {
      return ::num_vectors(impl_index_);
    }

    size_t num_vectors() const override {
      return ::num_vectors(impl_index_);
    }
#endif

   private:
    /**
     * @brief Instance of the concrete class.
     */
    T impl_index_;
  };

  // @todo Who should own the context?
  tiledb::Context ctx_{};
  tiledb_datatype_t feature_datatype_{TILEDB_ANY};
  tiledb_datatype_t id_datatype_{TILEDB_ANY};
  tiledb_datatype_t ptx_datatype_{TILEDB_ANY};
  std::unique_ptr</* const */ index_base> index_;
};

/*******************************************************************************
 * IndexIVFPQ
 ******************************************************************************/
// OMG -- this one is even weirder

/*******************************************************************************
 * IndexVamana
 ******************************************************************************/

/*******************************************************************************
 * Testing functions
 ******************************************************************************/

bool validate_top_k(const FeatureVectorArray& a, const FeatureVectorArray& b) {
  // assert(a.datatype() == b.datatype());

  auto proc_b = [&b](auto& aview) {
    switch (b.feature_type()) {
      case TILEDB_INT32: {
        auto bview = MatrixView<int32_t, stdx::layout_left>{
            (int32_t*)b.data(), extents(b)[0], extents(b)[1]};
        return validate_top_k(aview, bview);
      }
      case TILEDB_UINT32: {
        auto bview = MatrixView<uint32_t, stdx::layout_left>{
            (uint32_t*)b.data(), extents(b)[0], extents(b)[1]};
        return validate_top_k(aview, bview);
      }
      case TILEDB_INT64: {
        auto bview = MatrixView<int64_t, stdx::layout_left>{
            (int64_t*)b.data(), extents(b)[0], extents(b)[1]};
        return validate_top_k(aview, bview);
      }
      case TILEDB_UINT64: {
        auto bview = MatrixView<uint64_t, stdx::layout_left>{
            (uint64_t*)b.data(), extents(b)[0], extents(b)[1]};
        return validate_top_k(aview, bview);
      }
      default:
        throw std::runtime_error("Unsupported attribute type");
    }
  };

  switch (a.feature_type()) {
    case TILEDB_INT32: {
      auto aview = MatrixView<int32_t, stdx::layout_left>{
          (int32_t*)a.data(), extents(a)[0], extents(a)[1]};
      return proc_b(aview);
    }
    case TILEDB_UINT32: {
      auto aview = MatrixView<uint32_t, stdx::layout_left>{
          (uint32_t*)a.data(), extents(a)[0], extents(a)[1]};
      return proc_b(aview);
    }
    case TILEDB_INT64: {
      auto aview = MatrixView<int64_t, stdx::layout_left>{
          (int64_t*)a.data(), extents(a)[0], extents(a)[1]};
      return proc_b(aview);
    }
    case TILEDB_UINT64: {
      auto aview = MatrixView<uint64_t, stdx::layout_left>{
          (uint64_t*)a.data(), extents(a)[0], extents(a)[1]};
      return proc_b(aview);
    }
    default:
      throw std::runtime_error("Unsupported attribute type");
  }
}

#endif
