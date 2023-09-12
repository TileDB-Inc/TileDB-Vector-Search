/**
* @file   adj_list.h
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
 */

#ifndef TDB_ADJ_LIST_H
#define TDB_ADJ_LIST_H

#include <vector>
#include <list>
#include <initializer_list>


template <class I = size_t>
class index_adj_list : public std::vector<std::list<I>> {

  using Base = std::vector<std::list<I>>;

 public:
  using value_type = I;
  using index_type = size_t;

  index_adj_list(size_t num_vertices) : Base(num_vertices) {}

#if 0
  template <class EdgeList>
  index_adj_list(EdgeList&& edge_list) {
    for (auto& [src, dst] : edge_list) {
      Base::operator[](src).push_back(dst);
    }
  }
#endif

  template <class AdjList>
  index_adj_list(AdjList&& l) : Base(size(l)) {
    for (size_t i = 0; i < size(l); ++i) {
      for (auto& dst : l[i]) {
        add_edge(i, dst);
      }
    }
  }

  auto add_edge(I src, I dst) { Base::operator[](src).push_back(dst); }

  auto& out_edges(I i) { return Base::operator[](i); }

  auto& out_degree(I i) { return Base::operator[](i).size(); }

  auto& num_vertices() { return Base::size(); }

};

template <class I = size_t>
auto num_vertices(index_adj_list<I>& g) { return g.num_vertices(); }

template <class I = size_t>
auto& out_edges(index_adj_list<I>& g, I i) { return g.out_edges(i); }

template <class I = size_t>
auto& out_degree(index_adj_list<I>& g, I i) { return g.out_degree(i); }


#endif //TDB_ADJ_LIST_H