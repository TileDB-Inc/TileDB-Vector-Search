/**
 * @file   linalg.h
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
 * Header-only library of some basic linear algebra data structures and
 * operations. Uses C++23 reference implementation of mdspan from Sandia
 * National Laboratories.
 *
 * The data structures here are lightweight wrappers around span (1-D) or
 * mdspan (2-D).  Their primary purpose is to provide classes that own
 * their storage while also presenting the span and mdspan interfaces.
 *
 * The tdbMatrix class is derived from Matrix, but gets its initial data
 * from a given TileDB array.
 *
 */

#ifndef TDB_LINALG_H
#define TDB_LINALG_H

#include "detail/linalg/choose_blas.h"
#include "detail/linalg/linalg_defs.h"
#include "detail/linalg/matrix.h"
#include "detail/linalg/tdb_helpers.h"
#include "detail/linalg/tdb_io.h"
#include "detail/linalg/tdb_matrix.h"
#include "detail/linalg/tdb_partitioned_matrix.h"
#include "detail/linalg/vector.h"

#endif  // TDB_LINALG_H
