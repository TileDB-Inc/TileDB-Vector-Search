

#ifndef TILEDB_TEST_UTILS_H
#define TILEDB_TEST_UTILS_H

#include <ranges>

template <std::ranges::range R>
void randomize(R& r, std::tuple<int, int> range = {0, 128}) {
  std::random_device rd;
  // std::mt19937 gen(rd());
  std::mt19937 gen(2514908090);

  if constexpr (std::is_floating_point_v<std::ranges::range_value_t<R>>) {
    std::uniform_real_distribution<std::ranges::range_value_t<R>> dist(
        std::get<0>(range), std::get<1>(range));
    for (auto& x : r) {
      x = dist(gen);
    }
    return;
  } else {
    std::uniform_int_distribution<std::ranges::range_value_t<R>> dist(
        std::get<0>(range), std::get<1>(range));
    for (auto& x : r) {
      x = dist(gen);
    }
    return;
  }
}

#endif  // TILEDB_TEST_UTILS_H