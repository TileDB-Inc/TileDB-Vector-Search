/**
 * @file   flat_query.h
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
 * This file contains the query functions for the TileDB vector similarity
 * demo program.
 *
 * The functions have the same API -- they take a database, a query, a ground
 * truth, and a top-k result set. The functions differ in how they iterate over
 * the database and query vectors. They are parallelized over their outer loops,
 * using `std::async`. They time different parts of the query and print the
 * results to `std::cout`. Each query verifies its results against the ground
 * truth and reports any errors. Note that the top k might not be unique (i.e.
 * there might be more than one vector with the same distance) so that the
 * computed top k might not match the ground truth top k for some entries.  It
 * should be obvious on inspection of the error output whether or not reported
 * errors are due to real differences or just to non-uniqueness of the top k.
 *
 * I have started to parallelize the functions using `stdx::for_each`.
 *
 * Note that although the functions are templated on the database and query
 * type, they expect a "vector of spans" interface.  This works well with
 * the current `std::for_each` and is a reasonable way to think about the
 * sets of vectors.  However, having `mdspan` is more lightweight, but does
 * not support the `std::for_each` interface because it does not have
 * iterators.  I have not yet decided which is the best representation.
 *
 * These algorithms have not been blocked yet.
 */

#ifndef TDB_FLAT_QUERY_H
#define TDB_FLAT_QUERY_H

#include "detail/flat/gemm.h"
#include "detail/flat/qv.h"
#include "detail/flat/vq.h"

#endif  // TDB_FLAT_QUERY_H
