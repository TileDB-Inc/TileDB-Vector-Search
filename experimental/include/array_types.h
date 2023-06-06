
#ifndef TDB_ARRAY_TYPES_H
#define TDB_ARRAY_TYPES_H

#include <cstdint>

/**
 * In general for our similarity computations, large sets of vectors are stored
 * using 8 bits per vector element.  Occasionally, smaller datsets will be
 * stored as float.  Ideally, we would support either type (likely using some
 * form of type erasure).  But for now, we just use a sledgehammer.
 *
 * @todo Use size_t instead of uint64_t?
 */
#if 1
using db_type = uint8_t;
using shuffled_db_type = uint8_t;
using shuffled_ids_type = uint64_t;
using indices_type = uint64_t;
using parts_type = uint64_t;
using centroids_type = float;
using q_type = uint8_t;
using groundtruth_type = int32_t;

using original_ids_type = uint64_t;
#else

using db_type = float;
using shuffled_db_type = float;
using shuffled_ids_type = uint64_t;
using indices_type = uint64_t;
using parts_type = uint64_t;
using centroids_type = float;
using q_type = float;
using groundtruth_type = int32_t;

#endif

#endif  // TDB_ARRAY_TYPES_H