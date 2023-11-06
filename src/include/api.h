/**
 * @file   api.h
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
 * Nascent C++ API.  Type-erased classes are provided here as an interface
 * between the C++ vector search library and the Python bindings.
 *
 * Type erasure is accomplished with the following pattern.
 *   - The type-erased class (the outer class) provides the API that is invoked
 * by Python
 *   - It defines an abstract base class that is not a template and a derived
 * implementation class that is a template.
 *   - Constructors for the outer class use information from how they are
 * constructed (perhaps from reading the schema of an array) to determine the
 * type of the implementation class. The unique_ptr member of the outer class is
 * constructed with the derived implementation class.
 *   - The member functions comprising the outer class API invoke the
 * corresponding member functions of the base class object stored in the
 * unique_ptr (which in turn invoke members of the concrete class stored by the
 * implementation class).
 *
 */

#ifndef TDB_API_H
#define TDB_API_H

#include <memory>
#include <vector>

#include "concepts.h"
#include "cpos.h"
#include "detail/linalg/tdb_vector.h"
#include "index/flat_l2_index.h"
#include "index/index_defs.h"
#include "index/ivf_flat_index.h"

#include "utils/print_types.h"

#include <tiledb/tiledb>
#include "detail/linalg/tdb_helpers.h"
#include "detail/linalg/tdb_matrix.h"
#include "detail/linalg/tdb_vector.h"


//------------------------------------------------------------------------------
// FeatureVector
//------------------------------------------------------------------------------

/**
 * @brief Outer class defining the API for feature vectors.
 */

//------------------------------------------------------------------------------
// FeatureVectorArray
//------------------------------------------------------------------------------

/*******************************************************************************
 * IndexFlatPQ
 ******************************************************************************/

/*******************************************************************************
 * IndexIVFPQ
 ******************************************************************************/
// OMG -- this one is even weirder

/*******************************************************************************
 * IndexVamana
 ******************************************************************************/

/*******************************************************************************
 * Testing functions
 ******************************************************************************/


#endif
