/**
* @file   time_insert.cc
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


#define ALLHEAPS 1
#include "../fixed_min_queues.h"
#include "../timer.h"
#include <functional>
#include <numeric>
#include <random>
#include <set>
#include <vector>

template <class Heap>
void do_time(const std::string& msg, Heap& heap, const std::vector<size_t>& v) {
  life_timer _ { msg };

  auto size_v = v.size();
  for (unsigned i = 0; i < size_v; ++i) {
    heap.insert(v[i]);
  }
}

template <class Heap>
void do_time_pair(const std::string& msg, Heap& heap, const std::vector<size_t>& v) {
  life_timer _ { msg };

  auto size_v = v.size();
  for (unsigned i = 0; i < size_v; ++i) {
    heap.insert({ v[i], i });
  }
}

template <class Heap>
void do_time_indirect(const std::string& msg, Heap& heap, const std::vector<size_t>& v) {
  life_timer _ { msg };

  auto size_v = v.size();
  for (unsigned i = 0; i < size_v; ++i) {
    heap.insert(i);
  }
}

void do_time_nth_element(const std::string& msg, std::vector<size_t>& v, size_t n) {
  life_timer _ { msg };

  std::nth_element(begin(v), begin(v) + n, end(v));
}

int main() {

  unsigned            n = 100'000'000;
  std::vector<size_t> v(n);
  std::vector<float>  scores(n);
  std::iota(begin(v), end(v), 17);
  std::iota(begin(scores), end(scores), 17);

  // Use a random device as the seed for the random number generator
  std::random_device rd;
  std::mt19937       rng(rd());


  for (auto i : { 1, 10, 50, 100 }) {
    {

      fixed_min_set_heap_1<size_t> heap1(i);
      fixed_min_set_heap_2<size_t> heap2(i);
      fixed_min_set_heap_3<size_t> heap3(i);
      //fixed_min_set_heap_3<size_t> heap4(i, std::greater<size_t>());
      fixed_min_set_set<size_t> set(i);

      using Comparator = std::function<bool(unsigned, unsigned)>;

      auto heap5 = fixed_min_set_heap_1<unsigned, Comparator>(i, [&](unsigned a, unsigned b) { return scores[a] < scores[b]; });
      auto heap6 = fixed_min_set_heap_2<unsigned, Comparator>(i, [&](unsigned a, unsigned b) { return scores[a] < scores[b]; });
      auto heap7 = fixed_min_set_heap_3<unsigned, Comparator>(i, [&](unsigned a, unsigned b) { return scores[a] < scores[b]; });

      //      do_time("heap4", heap4);
      //      do_time("set", set);

      {
        life_timer _ { "warm cache" };
        for (size_t i = 0; i < v.size(); ++i) {
          v[i] = i + 1;
        }
      }

      {
        life_timer _ { "calibration" };
        for (size_t i = 0; i < v.size(); ++i) {
          v[i] = i - 1;
        }
      }

      std::sort(begin(v), end(v), std::less<>());
      do_time("heap1 ascending", heap1, v);

      std::sort(begin(v), end(v), std::greater<>());
      do_time("heap1 descending", heap1, v);

      std::shuffle(begin(v), end(v), rng);
      do_time("heap1 random", heap1, v);

      std::shuffle(begin(v), end(v), rng);
      do_time("heap2", heap2, v);


      std::sort(begin(v), end(v), std::less<>());
      do_time("heap3 ascending", heap3, v);

      std::sort(begin(v), end(v), std::greater<>());
      do_time("heap3 descending", heap3, v);

      std::shuffle(begin(v), end(v), rng);
      do_time("heap3 random", heap3, v);

      std::shuffle(begin(v), end(v), rng);
      std::shuffle(begin(scores), end(scores), rng);
      do_time_indirect("indirect heap5 (1)", heap5, v);

      std::shuffle(begin(v), end(v), rng);
      std::shuffle(begin(scores), end(scores), rng);
      do_time_indirect("indirect heap6 (2)", heap6, v);

      std::shuffle(begin(v), end(v), rng);
      std::shuffle(begin(scores), end(scores), rng);
      do_time_indirect("indirect heap7 (3)", heap7, v);

      std::sort(begin(v), end(v), std::less<>());
      do_time_nth_element("nth descending", v, i);

      std::sort(begin(v), end(v), std::greater<>());
      do_time_nth_element("nth ascending", v, i);

      std::shuffle(begin(v), end(v), rng);
      do_time_nth_element("nth element random", v, i);
    }
    if (false) {
      fixed_min_set_heap_1<std::pair<float, size_t>> heap1(i);
      fixed_min_set_heap_2<std::pair<float, size_t>> heap2(i);
      fixed_min_set_heap_3<std::pair<float, size_t>> heap3(i);
      //fixed_min_set_heap_3<std::pair<float, size_t>> heap4(i, std::greater<>());
      fixed_min_set_set<std::pair<float, size_t>> set(i);


      std::shuffle(begin(v), end(v), rng);
      do_time_pair("heap1", heap1, v);

      std::shuffle(begin(v), end(v), rng);
      do_time_pair("heap2", heap2, v);

      std::shuffle(begin(v), end(v), rng);
      do_time_pair("heap3", heap3, v);
      //      do_time_pair("heap4", heap4);
      //      do_time_pair("set", set);
    }
    std::cout << "\n";
  }
}
