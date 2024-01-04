/**
 * @file   query_common.h
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
 *
 */

#ifndef TILEDB_QUERY_COMMON_H
#define TILEDB_QUERY_COMMON_H

#include <string>
#include "linalg.h"

// clang-format off

  auto centroids = ColMajorMatrix<float> {
      {
          {8, 6, 7},
          {5, 3, 0},
          {9, 1, 2},
          {3, 4, 5},
          {6, 7, 8},
          {9, 0, 1},
          {2, 3, 4},
          {5, 6, 7},
          {8, 9, 0},
          {1, 2, 3},
          {4, 5, 6},
          {7, 8, 9},
          {3.14, 1.59, 2.65},
          {35, 89, 793},
          {2, 384, 6.26},
          {4, 33, 8},
          {32.7, 9.502, 8},
          {84, 1, 97},
          {3, 1, 4},
          {1, 5, 9},
          {9, 0, 3,},
          {5, 7, 6},
      }
  };
  auto query = ColMajorMatrix<float> {
      {
          {3, 4, 5},
          {2, 300, 8},
          {3, 1, 3.5},
          {3, 1, 3},
          {4, 5, 6},
      }
  };

  /**
   * Taken from [0:5,9:12] of sift_base
   */
  auto sift_base = ColMajorMatrix<float>
      {
          { 21.,  13.,  17.},
          { 13.,  60.,  10.},
          { 18.,  15.,   6.},
          { 11.,   4.,  47.},
          { 14.,   5.,  11.},
          {  6.,   1.,   6.},
          {  4.,   1.,   1.},
          { 14.,   9.,  20.},
          { 39.,  11.,  49.},
          { 54.,  72.,  86.},
          { 52., 114.,  36.},
          { 10.,  30.,  33.},
          {  8.,   2.,   5.},
          { 14.,   1.,   6.},
          {  5.,   9.,   2.},
          {  2.,  25.,   0.},
          { 23.,   2.,   9.},
          { 76.,  29.,  62.},
          { 65., 114.,  53.},
          { 10.,  17.,  29.},
          { 11.,   2.,  10.},
          { 23.,  12.,  19.},
          {  3.,  11.,   4.},
          {  0.,   0.,   0.},
          {  6.,   2.,   6.},
          { 10.,  33.,   9.},
          { 17.,  56.,   9.},
          {  5.,  11.,   7.},
          {  7.,   2.,   7.},
          { 21.,  35.,  30.},
          { 20.,  10.,  12.},
          { 13.,   2.,  10.},
      };

  auto sift_query = ColMajorMatrix<float>
      {
          { 0.,  7., 50.},
          {11.,  4., 43.},
          {77.,  5.,  9.},
          {24., 11.,  1.},
          { 3.,  2.,  0.}
      };

// clang-format on


#endif  // TILEDB_QUERY_COMMON_H
