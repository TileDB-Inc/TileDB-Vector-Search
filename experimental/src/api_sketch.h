#include <cmath>
#include <span>
#include <stdexcept>
#include <tiledb/tiledb>

using namespace tiledb;

namespace tiledb::vector_search {

namespace impl {

template <class T>
constexpr int get_d(const std::vector<std::span<T>>& vectors) {
  if (vectors.empty()) {
    throw std::invalid_argument("Vectors array cannot be empty.");
  }

  int d = vectors[0].size();
  for (auto& v : vectors) {
    if (v.size() != d) {
      throw std::invalid_argument("All vectors must have the same length.");
    }
  }
  return d;
}

template <class T>
ArraySchema create_kmeans_index_schema(const Context& ctx, int d, int k) {
  return ArraySchema{ctx, TILEDB_DENSE}
      .set_domain(Domain{ctx}.add_dimension(
          Dimension::create<int>(ctx, "kmeans_id", {0, k - 1})))
      .add_attribute(Attribute::create<T>(ctx, "value").set_cell_val_num(d));
}

std::string get_kmeans_index_uri(const std::string& uri) {
  return uri + "/kmeans_index";
}

template <class T>
ArraySchema create_data_schema(const Context& ctx, int d, int k) {
  return ArraySchema{ctx, TILEDB_DENSE}
      .set_domain(Domain{ctx}
                      .add_dimension(
                          Dimension::create<int>(ctx, "kmeans_id", {0, k - 1}))
                      .add_dimension(Dimension::create<int>(
                          ctx, "object_id", {0, 999999999})))
      .add_attribute(Attribute::create<T>(ctx, "vector").set_cell_val_num(d));
}

std::string get_data_uri(const std::string& uri) {
  return uri + "/data";
}

}  // namespace impl

/**
 * Provides access to a vector search index backed by a TileDB array.
 */
// template <class T>
class VectorArray {
  //   static_assert(
  //       std::is_same_v<T, float>,
  //       "VectorArray is currently only supported for floats.");
  using T = float;

 public:
  static void create(
      const Context& ctx,
      const std::string& uri,
      const std::vector<std::span<T>>& vectors) {
    int d = impl::get_d(vectors);
    int k = std::sqrt(d);

    Array::create(
        impl::get_kmeans_index_uri(uri),
        impl::create_kmeans_index_schema<T>(ctx, d, k));

    Array::create(
        impl::get_data_uri(uri), impl::create_data_schema<T>(ctx, d, k));

    throw std::runtime_error("Implement the rest");
  }

  VectorArray(
      const Context& ctx,
      const std::string& uri,
      tiledb_query_type_t query_type);

  /**
   * Returns the number of dimensions of the vectors of this vector array.
   */
  int get_d() {
    throw std::runtime_error("Implement me");
  }

  /**
   * Queries the vectors in the array that are most similar to the given vector.
   * Returns a list of the vectors alongside their ID.
   */
  std::vector<std::pair<std::vector<T>, int>> query(
      std::span<T> vector, int top_k) {
    if (top_k < 0) {
      throw std::invalid_argument("top_k cannot be negative");
    }

    if (top_k == 0) {
      return {};
    }

    throw std::runtime_error("Implement me");
  }
};
}  // namespace tiledb::vector_search
