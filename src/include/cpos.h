

template <class T>
concept semi_integral = std::integral<T> && !std::same_as<T, bool>;


// ----------------------------------------------------------------------------
// Customization point objects (CPOs) -- implemented as "niebloids"
// ----------------------------------------------------------------------------


namespace _dimension {
void dimension(auto&) = delete;
void dimension(const auto&) = delete;

template <class T>
concept _member_dimension = requires(T t) {
  { t.dimension() } -> semi_integral;
};

template <class T>
concept _member_size = requires(T t) {
  { t.size() } -> semi_integral;
};

template <class T>
concept _member_num_rows = requires(T t) {
  { t.num_rows() } -> semi_integral;
};

struct _fn {
  template <_member_dimension T>
  auto constexpr operator()(T&& t) const noexcept {
    return t.dimension();
  }

  template <class V>
    requires _member_size<V> && (!_member_num_rows<V>) && std::is_arithmetic_v<std::ranges::range_value_t<V>>
  auto constexpr operator()(const V& v) const noexcept {
    return v.size();
  }

  template <class V>
    requires _member_num_rows<V> && std::is_arithmetic_v<std::ranges::range_value_t<V>>
  auto constexpr operator()(const V& v) const noexcept {
    return v.num_rows();
  }
};
}  // namespace _dimension

inline namespace _cpo {
inline constexpr auto dimension = _dimension::_fn{};
}  // namespace _cpo


namespace _num_partitions {
void num_partitions(auto&) = delete;
void num_partitions(const auto&) = delete;

template <class T>
concept _member_num_partitions = requires(T t) {
  { t.num_partitions() };
};

struct _fn {
  template <_member_num_partitions T>
  auto constexpr operator()(T&& t) const noexcept {
    return t.num_partitions();
  }
};
}  // namespace _num_partitions

inline namespace _cpo {
inline constexpr auto num_partitions = _num_partitions::_fn{};
}  // namespace _cpo

namespace _num_vectors {
void num_vectors(auto&) = delete;
void num_vectors(const auto&) = delete;

template <class T>
concept _member_num_vectors = requires(T t) {
  { t.num_vectors() };
};

struct _fn {
  template <_member_num_vectors T>
  auto constexpr operator()(T&& t) const noexcept {
    return t.num_vectors();
  }
};
}  // namespace _num_vectors

inline namespace _cpo {
inline constexpr auto num_vectors = _num_vectors::_fn{};
}  // namespace _cpo
