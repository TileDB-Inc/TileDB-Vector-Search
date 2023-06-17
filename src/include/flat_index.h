/**
 * @file   flat_index.h
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
 * Header-only library of class that implements a flat index.
 *
 * The basic use case is:
 * - Create an instance of the index
 * - Call train() to build the index (nullop for flat index)
 * - OR Call load() to load the index from TileDB arrays
 * - Call add() to add vectors to the index (alt. add with ids)
 * - Call search() to query the index, returning the ids of the nearest vectors,
 *   and optionally the distances.
 * - Compute the recall of the search results.
 *
 * - Call save() to save the index to disk
 * - Call reset() to clear the index
 *
 * In general, we will neither be able to have the entire index in memory,
 * nor the original vectors.  Thus, we either construct the index from
 * URIs (and the index uses out-of-core data structures), or we construct
 * the requisite data structures and pass them to the index, again using
 * out-of-core data structures as necessary.
 */

#ifndef TILEDB_IVF_INDEX_H
#define TILEDB_IVF_INDEX_H

#endif TILEDB_IVF_INDEX_H
