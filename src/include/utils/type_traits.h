
#include <tiledb/tiledb>

auto shit = tiledb::impl::type_to_tiledb<int>::name;

namespace tiledb::impl {

template <>
struct type_to_tiledb<long> {
  using type = float;
  static const tiledb_datatype_t tiledb_type = TILEDB_INT64;
  static constexpr const char* name = "INT64";
};


} // namespace tiledb::impl
