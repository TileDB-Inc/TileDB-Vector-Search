// Base class for all indexes

/*
- [x] we need a name for the type-erased object wrapping a Matrix: FeatureVectorArray
*/

#include <optional>
#include <span>

namespace {

using URI = std::string; // TODO

template <typename From, typename To>
struct MapLike{};

using StringMap = MapLike<std::string, std::string>;

class IndexOptions{};

class TrainingParameters{};

class FeatureVectorArray {
  // .data() -> void*
  // .ndim() -> size_t
  // .size() -> size_t
  // .extents() -> std::vector<size_t>
  // TBD .stride
};

}; // namespace

class IndexBase; // pimpl

class Index {
public:

  // Load from URI
  Index(
    URI index_uri,
    std::optional<StringMap> config = std::nullopt
  );

  // Create from input vectors
  Index(
    URI index_uri,
    FeatureVectorArray vectors,
    IndexOptions options,
    std::optional<StringMap> config = std::nullopt
  );

  // Create from input URI
  Index(
    URI index_uri,
    URI vectors_uri,
    IndexOpts options,
    std::optional<StringMap> config = std::nullopt
  );

  ~Index();

  // Query this index with given vectors
  auto query(
    QueryVectorArray vectors,
    size_t top_k,
    QueryResults& results
  ) -> std::pair(FeatureVectorArray, IdVector);

  // Insert, optionally with IDs and training parameters
  void insert(
    FeatureVectorArray vectors,
    std::optional<IdVector> ids = std::nullopt,
    std::optional<TrainingParameters> params = std::nullopt
  );

  void remove(
    std::span<IdType> ids
  );

private:
  tiledb::Context ctx_;
  std::unique_ptr<IndexBase> impl_;
};