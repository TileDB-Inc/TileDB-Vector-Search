/**
 * @file   array_types.h
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
 * Some hard-coded data types based on our current schema for query search.
 * Will be expanded and generalized.
 *
 */

#ifndef TDB_ARRAY_TYPES_H
#define TDB_ARRAY_TYPES_H

#include <cstdint>

/**
 * @todo Use size_t instead of uint64_t?
 */

#if 1

using db_type = uint8_t;
#else
using db_type = float;
#endif

using q_type = db_type;
using groundtruth_type = int32_t;

using centroids_type = float;
using shuffled_db_type = db_type;

// @todo Should be the same as groundtruth_type
using shuffled_ids_type = uint64_t;

// @todo Are these the same?
using indices_type = uint64_t;
using parts_type = uint64_t;

#endif  // TDB_ARRAY_TYPES_H
