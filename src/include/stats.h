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
 * Very simple code for gathering and reporting execution environment, using
 * the nlohmann/json library.  This is currently not being used, in favor
 * of the singletons in logging.h.  This code is left here for reference.
 *
 * @todo Optionally generate json code from the singleton loggers.
 * @todo Make the config information accessible to the rest of the code in
 * a more convenient way.
 */

#ifndef TDB_STATS_H
#define TDB_STATS_H

#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <tiledb/tiledb>
#include "config.h"
#include "utils/logging_count.h"
#include "utils/logging_memory.h"
#include "utils/logging_scoped_time.h"
#include "utils/logging_time.h"

#ifdef __GNUC__
// Disable the specific warning for the expression that causes the warning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include "nlohmann/json.hpp"
using json = nlohmann::json;

// Make stats support opt-in to avoid requiring to define an enable_stats
// variable on all projects.
#ifdef TILEDB_VS_ENABLE_STATS
extern bool enable_stats;
extern std::vector<json> core_stats;
#endif

class StatsCollectionScope final {
 public:
  explicit StatsCollectionScope(
      const std::string& uri,
      const std::string& function,
      const std::string& operation_type) {
#ifdef TILEDB_VS_ENABLE_STATS
    if (!enable_stats)
      return;
    tiledb::Stats::reset();
    uri_ = uri;
    function_ = function;
    operation_type_ = operation_type;
#else
    std::ignore = std::make_tuple(uri, function, operation_type);
#endif
  }

  ~StatsCollectionScope() {
#ifdef TILEDB_VS_ENABLE_STATS
    if (!enable_stats)
      return;
    std::string stats_str;
    tiledb::Stats::raw_dump(&stats_str);
    core_stats.push_back(
        {{"uri", uri_},
         {"function", function_},
         {"operation_type", operation_type_},
         {"stats", json::parse(stats_str)}});
#endif
  }

#ifdef TILEDB_VS_ENABLE_STATS
 private:
  std::string uri_, function_, operation_type_;
#endif
};

[[maybe_unused]] static std::string logging_string() {
  std::ostringstream output;

  output << "Scoped Timers" << std::endl;
  output << _scoped_timing_data.dump();
  output << std::endl;

  output << "Manual Timers" << std::endl;
  output << _timing_data.dump();
  output << std::endl;

  output << "Memory data" << std::endl;
  output << _memory_data.dump();
  output << std::endl;

  output << "Count data" << std::endl;
  output << _count_data.dump();
  output << std::endl;

  return output.str();
}

static auto dump_logs = [](const std::string& filename,
                           const std::string& algorithm,
                           size_t nqueries,
                           size_t nprobe,
                           size_t k_nn,
                           size_t nthreads,
                           double recall) {
  auto c = !filename.empty() ? std::ofstream(filename) : std::ofstream();
  std::ostream& output{(filename == "-") ? std::cout : c};

  output << "Repo: " << GIT_REPO_NAME << " @ " << GIT_BRANCH << " / "
         << GIT_COMMIT_HASH << std::endl;
  output << "CMake source directory: " << CMAKE_SOURCE_DIR << std::endl;
  output << "CMake build type: " << BUILD_TYPE << std::endl;
  output << "Compiler: " << IVF_HACK_CXX_COMPILER << std::endl;
  output << "Compiler ID: " << CXX_COMPILER_ID << std::endl;
  output << "Compiler version : " << CXX_VERSION << std::endl;
  output << "C++ flags: " << CMAKE_CXX_FLAGS << std::endl;
  output << "C++ Debug flags: " << CMAKE_CXX_FLAGS_DEBUG << std::endl;
  output << "C++ Release flags: " << CMAKE_CXX_FLAGS_RELEASE << std::endl;
  output << "C++ ReleaseWithDebugInfo flags: " << CMAKE_CXX_FLAGS_RELWITHDEBINFO
         << std::endl;
  output << std::endl;

  output << std::setw(9) << "Algorithm";
  output << std::setw(8) << "Queries";
  output << std::setw(8) << "nprobe";
  output << std::setw(8) << "k_nn";
  output << std::setw(8) << "thrds";
  output << std::setw(8) << "recall";
  output << std::endl;

  output << std::setw(9) << algorithm;
  output << std::setw(8) << nqueries;
  output << std::setw(8) << nprobe;
  output << std::setw(8) << k_nn;
  output << std::setw(8) << nthreads;
  output << std::fixed << std::setprecision(3);
  output << std::setw(8) << recall;
  output << std::endl;
  output << std::endl;

  output << logging_string();
};

[[maybe_unused]] static auto build_config() {
  // This is failing today, but could perhaps be added back in the future.
  // char host_[16];
  // if (int e = gethostname(host_, sizeof(host_))) {
  //   std::cerr << "truncated host name\n";
  //   strncpy(host_, "ghost", 15);
  // }

  auto&& [major, minor, patch] = tiledb::version();

  json config = {
      {"CURRENT_DATETIME", CURRENT_DATETIME},
      {"CMAKE_SOURCE_DIR", CMAKE_SOURCE_DIR},
      {"BUILD_TYPE", BUILD_TYPE},
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

/**
 * Log all of the argument data created by the command line parser (docopt).
 * @tparam Args
 * @param args
 * @return
 */
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
