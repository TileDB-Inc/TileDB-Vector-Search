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

#include "config.h"

#include "nlohmann/json.hpp"

using json = nlohmann::json;

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





#endif  // TDB_STATS_H
