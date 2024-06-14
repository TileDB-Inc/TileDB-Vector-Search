/**
 * @file   tdb_helpers.h
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
 * Helper functions for certain TileDB operations.
 *
 */

#ifndef TILEDB_HELPERS_H
#define TILEDB_HELPERS_H

#include <tiledb/array.h>
#include <tiledb/tiledb>
#include "detail/time/temporal_policy.h"
#include "stats.h"

namespace tiledb_helpers {

/**
 * @brief Opens a TileDB array and displays stats to stderr.
 *
 * Stats are only collected if the TILEDB_VS_ENABLE_STATS symbol is
 * defined, and a variable named enable_stats is set to true.
 * The stats are written to a FILE* specified by the variable named stats_file.
 *
 * @param function_name The name of the function calling this. You can use the
 * tdb_func__ macro.
 * @param ctx The TileDB context to use.
 * @param uri The URI of the array to open.
 * @param query_type The mode to open the array.
 */
inline std::unique_ptr<tiledb::Array> open_array(
    const std::string& function_name,
    const tiledb::Context& ctx,
    const std::string& uri,
    tiledb_query_type_t query_type,
    TemporalPolicy temporal_policy = {}) {
  StatsCollectionScope stats_scope(uri, function_name, "open_array");
  return std::make_unique<tiledb::Array>(
      ctx, uri, query_type, temporal_policy.to_tiledb_temporal_policy());
}
/**
 * @brief Submits a TileDB query and displays stats to stderr.
 *
 * Stats are only collected if the TILEDB_VS_ENABLE_STATS symbol is
 * defined, and a variable named enable_stats is set to true.
 * The stats are written to a FILE* specified by the variable named stats_file.
 *
 * @param function_name The name of the function calling this. You can use the
 * tdb_func__ macro.
 * @param query The query to submit.
 */
inline void submit_query(
    const std::string& function_name,
    const std::string& uri,
    tiledb::Query& query) {
  StatsCollectionScope stats_scope(uri, function_name, "submit_query");
  query.submit();
}

// Adds an object to a group. Automatically infers whether to use a relative
// path or absolute path. NOTE(paris): We use absolute paths for tileDB URIs
// because of a bug tracked in SC39197, once that is fixed everything can use
// relative paths.
inline void add_to_group(
    tiledb::Group& group, const std::string& uri, const std::string& name) {
  if (uri.find("tiledb://") == 0) {
    group.add_member(uri, false, name);
  } else {
    group.add_member(name, true, name);
  }
}

}  // namespace tiledb_helpers

#endif
