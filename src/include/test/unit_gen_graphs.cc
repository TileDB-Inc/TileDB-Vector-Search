/**
 * @file  unit_gen_graphs.cc
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
 * Program to test performance of different min-heap implementations.
 *
 */

#include <catch2/catch_all.hpp>

#include "gen_graphs.h"
#include "utils/print_types.h"

TEST_CASE("gen_graphs: test test", "[gen_graphs]") {
  REQUIRE(true);
}

TEST_CASE("gen_graphs: unigrid", "[gen_graphs]") {
  auto&& [vecs, edges] = gen_uni_grid(5, 7);

  dump_coordinates("coords.txt", vecs);
  dump_edgelist("edges.txt", edges);
}

TEST_CASE("gen_graphs: bigrid", "[gen_graphs]") {
  auto&& [vecs, edges] = gen_bi_grid(5, 7);

  dump_coordinates("coords.txt", vecs);
  dump_edgelist("edges.txt", edges);
}
