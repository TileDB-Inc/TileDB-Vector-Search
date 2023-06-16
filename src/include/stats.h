/**
 * @file   stats.h
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
 * Very simple code for gathering and reporting execution environment.
 */

#ifndef TDB_STATS_H
#define TDB_STATS_H

#include <unistd.h>
#include <iostream>
#include <random>

#include <tiledb/tiledb>

#include "array_types.h"
#include "config.h"
#include "utils/utils.h"

#include "nlohmann/json.hpp"

using json = nlohmann::json;

extern bool global_debug;

auto config_log(const std::string& program_name) {
  std::string uuid_;
  char host_[16];
  std::string date_;
  std::size_t uuid_size_ = 24;

  auto seed = std::random_device();
  auto gen = std::mt19937(seed());
  auto dis = std::uniform_int_distribution<int8_t>(97, 122);
  uuid_.resize(uuid_size_);
  std::generate(uuid_.begin(), uuid_.end(), [&] { return dis(gen); });

  if (int e = gethostname(host_, sizeof(host_))) {
    std::cerr << "truncated host name\n";
    strncpy(host_, "ghost", 15);
  }
  {
    std::stringstream ss;
    std::time_t currentTime = std::time(nullptr);
    std::string dateString = std::ctime(&currentTime);
    dateString.erase(dateString.find('\n'));
    ss << dateString;
    date_ = ss.str();
  }

  auto&& [major, minor, patch] = tiledb::version();

  json config = {
      {"uuid", uuid_},
      {"host", host_},
      {"Program", program_name},
      {"Build_date", CURRENT_DATETIME},
      {"Run_date", date_},
      {"cmake_source_dir", CMAKE_SOURCE_DIR},
      {"Build", BUILD_TYPE},
      {"Compiler",
       {{"CXX_COMPILER", IVF_HACK_CXX_COMPILER},
        {"CXX_COMPILER_ID", CXX_COMPILER_ID},
        {"CXX_VERSION", CXX_VERSION},
        {"CMAKE_CXX_FLAGS", CMAKE_CXX_FLAGS},
        {"CMAKE_CXX_FLAGS_DEBUG", CMAKE_CXX_FLAGS_DEBUG},
        {"CMAKE_CXX_FLAGS_RELEASE", CMAKE_CXX_FLAGS_RELEASE},
        {"CMAKE_CXX_FLAGS_RELWITHDEBINFO", CMAKE_CXX_FLAGS_RELWITHDEBINFO}}},
      {"GIT_REPO",
       {{"GIT_REPO_NAME", GIT_REPO_NAME},
        {"GIT_REPO_URL", GIT_REPO_URL},
        {"GIT_BRANCH", GIT_BRANCH},
        {"GIT_COMMIT_HASH", GIT_COMMIT_HASH},
        {"GIT_COMMIT_DATE", GIT_COMMIT_DATE},
        {"GIT_COMMIT_TIME", GIT_COMMIT_TIME}}},
      {"tiledb_version",
       {{"major", major}, {"minor", minor}, {"patch", patch}}}};

  return config;
}

template <typename Args>
auto args_log(const Args& args) {
  json arg_log;

  for (auto&& arg : args) {
    std::stringstream buf;
    buf << std::get<1>(arg);
    arg_log.push_back({std::get<0>(arg), buf.str()});
  }
  return arg_log;
}


/**
 * @brief Compute the recall of a top-k matrix against a groundtruth matrix.
 *
 * @tparam Topk The type of the `top_k` matrix
 * @param groundtruth_uri URI of the groundtruth matrix
 * @param top_k The previously computed matrix of top-k nearest neighbors
 * @param nqueries How many queries were used to compute the top-k matrix
 * @param k_nn
 * @return double
 */
template <class Topk>
double compute_recall(const std::string& groundtruth_uri, const Topk& top_k) {
  tiledb::Context ctx;
  return compute_recall(ctx, groundtruth_uri, top_k);
}

template <class Topk>
double compute_recall(const tiledb::Context ctx, const std::string& groundtruth_uri, const Topk& top_k) {
  size_t nqueries = top_k.num_cols();
  auto groundtruth =
      tdbColMajorMatrix<groundtruth_type>(ctx, groundtruth_uri, nqueries);
  return compute_recall(groundtruth, top_k);
}

template <class Matrix, class Topk>
double compute_recall(const Matrix& groundtruth, const Topk& top_k) {

  if (global_debug) {
    std::cout << std::endl;

    debug_matrix(groundtruth, "groundtruth");
    debug_slice(groundtruth, "groundtruth");

    std::cout << std::endl;
    debug_matrix(top_k, "top_k");
    debug_slice(top_k, "top_k");

    std::cout << std::endl;
  }

  assert(groundtruth.num_cols() == top_k.num_cols());

  size_t k_nn = top_k.num_rows();
  size_t total_intersected{0};
  size_t total_groundtruth = top_k.num_cols() * top_k.num_rows();
  for (size_t i = 0; i < top_k.num_cols(); ++i) {
    std::sort(begin(top_k[i]), end(top_k[i]));
    std::sort(begin(groundtruth[i]), begin(groundtruth[i]) + k_nn);
    debug_matrix(top_k, "top_k");
    debug_slice(top_k, "top_k");
    total_intersected += std::set_intersection(
        begin(top_k[i]),
        end(top_k[i]),
        begin(groundtruth[i]),
        end(groundtruth[i]),
        counter{});
  }
  auto recall = ((double)total_intersected) / ((double)total_groundtruth);

  if (global_verbose) {
    std::cout << "# total intersected = " << total_intersected << " of "
              << total_groundtruth << " = "
              << "R@" << k_nn << " of " << recall << std::endl;
  }

  return recall;
}

#endif  // TDB_STATS_H
