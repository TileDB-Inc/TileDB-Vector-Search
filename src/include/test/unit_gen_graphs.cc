

#include <catch2/catch_all.hpp>

#include "gen_graphs.h"


TEST_CASE("gen_graphs: test test", "[gen_graphs]") {
  REQUIRE(true);
}

TEST_CASE("gen_graphs: grid", "[gen_graphs]") {
  auto g = gen_grid(3, 3);

}

