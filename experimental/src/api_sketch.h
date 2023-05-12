#include <cmath>
#include <span>
#include <stdexcept>
#include <tiledb/tiledb>
#include <tiledb/tiledb_experimental>

using namespace tiledb;

namespace tiledb::vector_search {

/**
 * Specifies the type of index to use to speed up similarity searches.
 *
 * Flat: No index is used. All vectors are checked.
 * KMeans: The vectors are clustered into using the k-means algorithm, and only
 * the vectors in the cluster with the closest centroid are checked.
 */
enum IndexType { Flat, KMeans };

namespace impl {

const std::string& kmeans_id_attribute_name = "__kmeans_id";

const std::string& vector_attribute_name = "__vector";

const std::string& data_array_name = "data";

const std::string& kmeans_index_array_name = "kmeans_index";

const int format_version = 1;

template <class T>
int get_d(const std::vector<std::span<T>>& vectors) {
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
          Dimension::create<int>(ctx, kmeans_id_attribute_name, {0, k - 1})))
      .add_attribute(
          Attribute::create<T>(ctx, vector_attribute_name).set_cell_val_num(d));
}

std::string get_kmeans_index_uri(const std::string& uri) {
  return uri + "/" + kmeans_index_array_name;
}

template <class T>
ArraySchema create_data_schema(const Context& ctx, int d, int k) {
  return ArraySchema{ctx, TILEDB_DENSE}
      .set_domain(
          Domain{ctx}
              .add_dimension(Dimension::create<int>(
                  ctx, kmeans_id_attribute_name, {0, k - 1}))
              .add_dimension(Dimension::create<int>(
                  ctx, "object_id", {0, std::numeric_limits<int>::max()})))
      .add_attribute(Attribute::create<T>(ctx, "vector").set_cell_val_num(d));
}

std::string get_data_uri(const std::string& uri) {
  return uri + "/" + data_array_name;
}

const std::string& index_type_to_string(IndexType index_type) {
  switch (index_type) {
    case IndexType::Flat:
      return "flat";
    case IndexType::KMeans:
      return "k-means";
    default:
      throw std::invalid_argument("Invalid index type value.");
  }
}

std::string append_path(const std::string& uri, const std::string& path) {
  return uri + (uri.ends_with("/") ? "" : "/") + path;
}

void set_group_metadata(Group& group, const std::string& index_type) {
  group.put_metadata(
      "vector_search_format_version", TILEDB_INT32, 1, &format_version);
  group.put_metadata(
      "vector_search_index_type",
      TILEDB_STRING_ASCII,
      index_type.length(),
      index_type.data());
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
  /**
   * Creates a VectorArray.
   *
   * @param ctx The TileDB context to use.
   * @param uri The URI to create the vector array to.
   * @param vectors The vectors to insert into the array.
   * @param index_type The type of index to use. Defaults to flat.
   */
  static void create(
      const Context& ctx,
      const std::string& uri,
      const std::vector<std::span<T>>& vectors,
      IndexType index_type = IndexType::Flat) {
    int d = impl::get_d(vectors);
    int k = std::sqrt(d);

    // Convert the index type to string before creating the group, to give an
    // opportunity to throw if the enum has an invalid value.
    auto index_type_str{impl::index_type_to_string(index_type)};

    create_group(ctx, uri);
    Group group{ctx, uri, TILEDB_READ};
    impl::set_group_metadata(group, index_type_str);

    if (index_type == IndexType::KMeans) {
      std::string index_uri =
          impl::append_path(uri, impl::kmeans_index_array_name);
      Array::create(index_uri, impl::create_kmeans_index_schema<T>(ctx, d, k));
      group.add_member(index_uri, false, impl::data_array_name);
    }

    Array::create(
        impl::get_data_uri(uri), impl::create_data_schema<T>(ctx, d, k));

    group.close();

    throw std::runtime_error("Implement the rest");
  }

  VectorArray(
      const Context& ctx,
      const std::string& uri,
      tiledb_query_type_t query_type);

  /**
   * Returns the number of dimensions of the vectors of this vector array.
   */
  int get_d() const {
    throw std::runtime_error("Implement me");
  }

  /**
   * Queries the vectors in the array that are most similar to the given vector.
   * Returns a list of the vectors alongside their ID.
   */
  std::vector<std::pair<std::vector<T>, int>> query(
      std::span<T> vector, int top_k) const {
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
