
// Purely temporary file

#warning Deprecated legacy concepts

template <typename T>
concept has_load_member = requires(T&& t) { t.load(); };

template <class T>
constexpr bool is_loadable_v = has_load_member<T>;

namespace legacy_concepts {
template <class T>
bool load(T&& t) {
  return false;
}

template <has_load_member T>
bool load(T&& t) {
  return t.load();
}

template <class T>
size_t num_loads(T&& t) {
  return 1;
}

template <has_load_member T>
size_t num_loads(T&& t) {
  return t.num_loads();
}
}  // namespace legacy_concepts

template <typename T>
concept has_col_offset = requires(T&& t) { t.col_offset(); };

template <typename T>
concept has_num_col_parts = requires(T&& t) { t.num_col_parts(); };
