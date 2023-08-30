#include "index.h"

#include <tiledb/tiledb>
#include <tiledb/tiledb_experimental>

namespace {

// This class is the private implementation for Index.
template <typename IndexAlgorithm>
class IndexBase : Index {
public:
  auto ctx() -> tiledb::Context& {
    return ctx_;
  }

private:
  tiledb::Context ctx_;
}; // IndexBase

// Type-specialized IVFIndex
// This class will be a thing wrapper over
// 1. the array handles
// 2. dispached to the appropriate qv_query (and similar) functions
template <typename IndexVectorType>
class IVFIndex : IndexBase<IVFIndex> {
  // Constructor: creates all type-specialized ColMajorMatrix (eg) objects
  //              based on the IndexVectorType.
  IVFIndex(tiledb::Context& ctx, tiledb::Group group, StringMap config);
};

struct IndexInfo {
  std::string index_type;
  std::string db_type;
};

auto index_info(tiledb::Group& g) -> IndexInfo;

}; // namespace

// Load index from URI
// This function loads the index metadata and creates the appropriate
// type-specialized IndexBase object based on the index element type.
Index::Index(
  URI uri,
  std::optional<StringMap> config = std::nullopt)
  {
  // Load the group metadata
  tiledb::Group group(ctx_, uri);
  auto meta = load_index_type(group);

  if (meta.index_type == "IVF") {
    impl_ = [](auto dbtype) -> auto {
      if (dbtype == "Float32") {
        return std::make_unique<IVFIndex<float>>(ctx_, group_, config);
      } else if (dbtype == "Float64") {
        return std::make_unique<IVFIndex<double>>(ctx_, group_, config);
      } else {
        throw std::runtime_error("Unsupported DB type: " + dbtype);
      }
    }(meta.db_type);
  } else {
    throw std::runtime_error("Unsupported index type: " + meta.index_type);
  }
}

Index::query(
  QueryVectorArray vectors,
  size_t top_k,
  QueryResults& results
) {

  // Execute query on the type-specialized IndexBase object
  // and return the results as a pair of FeatureVectorArray and IdVector.

  // TBD: need to flesh out the transition back to FeatureVectorArray
  features, idvector = impl_->query(vectors, top_k, results);
  return std::make_pair(
    FeatureVectorArray(features),
    IdVector(ids));
}