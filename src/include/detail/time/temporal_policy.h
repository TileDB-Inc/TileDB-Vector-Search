/**
 * @file   temporal_policy.h
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2024 TileDB, Inc.
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
 * Provides the same functionality as tiledb::TemporalPolicy, but:
 * - Does not break the copy assignment operator if it is a member variable of a
 * class.
 * - Exposes timestamp_start and timestamp_end as public members.
 *
 */

#ifndef TDB_TIME_TEMPORAL_POLICY_H
#define TDB_TIME_TEMPORAL_POLICY_H

/** Marker class to enforce a TimeTravel TemporalPolicy. */
class TimeTravelMarker {};
static constexpr TimeTravelMarker TimeTravel{};

/** Marker class to enforce a TimestampStartEnd TemporalPolicy. */
class TimestampStartEndMarker {};
static constexpr TimestampStartEndMarker TimestampStartEnd{};

class TemporalPolicy {
 public:
  TemporalPolicy()
      : timestamp_start_(0)
      , timestamp_end_(UINT64_MAX){};

  TemporalPolicy(const TimeTravelMarker&, uint64_t timestamp)
      : timestamp_start_(0)
      , timestamp_end_(timestamp){};

  TemporalPolicy(
      const TimestampStartEndMarker&,
      uint64_t timestamp_start,
      uint64_t timestamp_end)
      : timestamp_start_(timestamp_start)
      , timestamp_end_(timestamp_end){};

  inline uint64_t timestamp_start() const {
    return timestamp_start_;
  }

  inline uint64_t timestamp_end() const {
    return timestamp_end_;
  }

  inline tiledb::TemporalPolicy to_tiledb_temporal_policy() const {
    return tiledb::TemporalPolicy(
        tiledb::TimestampStartEnd, timestamp_start_, timestamp_end_);
  }

  std::string dump() const {
    return std::string("(timestamp_start: ") +
           std::to_string(timestamp_start_) +
           ", timestamp_end: " + std::to_string(timestamp_end_) + ")";
  }

 private:
  uint64_t timestamp_start_;
  uint64_t timestamp_end_;
};

#endif
