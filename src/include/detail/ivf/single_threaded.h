/**
* @file   ivf/qv.h
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
* Implementation of queries for the "qv" orderings, i.e., for which the loop
* over the queries is on the outer loop and the loop over the vectors is on the
* inner loop.  Since the vectors in the inner loop are partitioned, we can
* operate on them blockwise, which ameliorates the locality issues that
* arise when doing this order with a flat index, say.
*
* There are two implementations here: infinite RAM and finite RAM.  The
* infinite RAM case loads the entire partitioned database into memory, and then
* searches in the partitions as indicated by the nearest centroids to the
* queries.  The infinite RAM case does not perform any out-of-core operations.
* The finite RAM case only loads the partitions into memory that are necessary
* for the search. The user can specify an upper bound on the amount of RAM to
* be used for holding the queries being searched.  The searches are ordered so
* that the partitions can be loaded into memory in the order they are layed out
* in the array.
*
* In general there is probably no reason to ever use the infinite RAM case
* other than for benchmarking, as it requires machines with very large amounts
* of RAM.
*/

#ifndef TILEDB_IVF_SINGLE_THREADED_H
#define TILEDB_IVF_SINGLE_THREADED_H


#include "algorithm.h"
#include "concepts.h"
#include "detail/linalg/tdb_matrix.h"
#include "detail/linalg/tdb_partitioned_matrix.h"
#include "flat_query.h"
#include "linalg.h"

namespace detail::ivf {




}

#endif // TILEDB_IVF_SINGLE_THREADED_H