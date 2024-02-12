/**
 * @file   execution_policy.h
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
 */
#ifndef EXECUTION_POLICY_H
#define EXECUTION_POLICY_H

#include <algorithm>
#include <execution>
#include <numeric>
#include <version>

namespace stdx {
#ifdef __cpp_lib_execution
namespace execution {

// using sequenced_policy = std::execution::sequenced_policy;
// using parallel_policy = std::execution::parallel_policy;
// using parallel_unsequenced_policy =
// std::execution::parallel_unsequenced_policy; using unsequenced_policy =
// std::execution::unsequenced_policy;

constexpr auto seq = std::execution::seq;
constexpr auto par = std::execution::par;
constexpr auto par_unseq = std::execution::par_unseq;
constexpr auto unseq = std::execution::unseq;
}  // namespace execution

template <class ExecutionPolicy, class ForwardIt>
auto for_each(ExecutionPolicy&& policy, ForwardIt first, ForwardIt last) {
  return std::for_each(policy, first, last);
}

template <class ExecutionPolicy, class ForwardIt, class OutputIt>
auto transform(
    ExecutionPolicy&& policy,
    ForwardIt first,
    ForwardIt last,
    OutputIt d_first) {
  return std::transform(policy, first, last, d_first);
}

template <class ExecutionPolicy, class ForwardIt, class OutputIt>
auto copy(
    ExecutionPolicy&& policy,
    ForwardIt first,
    ForwardIt last,
    OutputIt d_first) {
  return std::copy(policy, first, last, d_first);
}

template <class ExecutionPolicy, class ForwardIt, class T>
void fill(
    ExecutionPolicy&& policy, ForwardIt first, ForwardIt last, const T& value) {
  std::fill(policy, first, last, value);
}

template <class ExecutionPolicy, class ForwardIt, class T>
void iota(
    ExecutionPolicy&& policy, ForwardIt first, ForwardIt last, const T& value) {
  std::iota(policy, first, last, value);
}

template <class ExecutionPolicy, class ForwardIt, class OutputIt>
auto inclusive_scan(
    ExecutionPolicy&& policy,
    ForwardIt first,
    ForwardIt last,
    OutputIt d_first) {
  return std::inclusive_scan(policy, first, last, d_first);
}

template <class ExecutionPolicy, class ForwardIt, class OutputIt>
auto exclusive_scan(
    ExecutionPolicy&& policy,
    ForwardIt first,
    ForwardIt last,
    OutputIt d_first) {
  return std::exclusive_scan(policy, first, last, d_first);
}

template <class ExecutionPolicy, class ForwardIt>
void sort(ExecutionPolicy&& policy, ForwardIt first, ForwardIt last) {
  std::sort(policy, first, last);
}

template <class ExecutionPolicy, class ForwardIt, class BinaryOp>
void sort(
    ExecutionPolicy&& policy, ForwardIt first, ForwardIt last, BinaryOp comp) {
  std::sort(policy, first, last, comp);
}

template <class ExecutionPolicy, class ForwardIt>
auto minmax_element(ExecutionPolicy&& policy, ForwardIt first, ForwardIt last) {
  return std::minmax_element(policy, first, last);
}

template <class ExecutionPolicy, class ForwardIt, class BinaryOp>
auto minmax_element(
    ExecutionPolicy&& policy, ForwardIt first, ForwardIt last, BinaryOp comp) {
  return std::minmax_element(policy, first, last, comp);
}

template <class ExecutionPolicy, class ForwardIt>
ForwardIt adjacent_find(
    ExecutionPolicy&& policy, ForwardIt first, ForwardIt last) {
  return std::adjacent_find(policy, first, last);
}

template <class ExecutionPolicy, class ForwardIt, class BinaryPredicate>
ForwardIt adjacent_find(
    ExecutionPolicy&& policy,
    ForwardIt first,
    ForwardIt last,
    BinaryPredicate p) {
  return std::adjacent_find(policy, first, last, p);
}

#else

#warning \
    "Parallel execution policies and parallel standard library not supported!"
namespace execution {
constexpr size_t seq{0xdeadbeef};
constexpr size_t par_unseq{0xdeadbeef};
constexpr size_t par{0xdeadbeef};
}  // namespace execution

template <class ExecutionPolicy, class ForwardIt>
constexpr ForwardIt adjacent_find(
    ExecutionPolicy&& policy, ForwardIt first, ForwardIt last) {
  return std::adjacent_find(first, last);
}

template <class ExecutionPolicy, class ForwardIt>
auto for_each(ExecutionPolicy&& policy, ForwardIt first, ForwardIt last) {
  return std::for_each(first, last);
}

template <class ExecutionPolicy, class ForwardIt>
auto sort(ExecutionPolicy&& policy, ForwardIt first, ForwardIt last) {
  return std::sort(first, last);
}

template <class ExecutionPolicy, class ForwardIt, class OutputIt>
auto transform(
    ExecutionPolicy&& policy,
    ForwardIt first,
    ForwardIt last,
    OutputIt d_first) {
  return std::transform(first, last, d_first);
}

template <class ExecutionPolicy, class ForwardIt, class OutputIt>
auto copy(
    ExecutionPolicy&& policy,
    ForwardIt first,
    ForwardIt last,
    OutputIt d_first) {
  return std::copy(first, last, d_first);
}

template <class ExecutionPolicy, class ForwardIt, class T>
void fill(
    ExecutionPolicy&& policy, ForwardIt first, ForwardIt last, const T& value) {
  std::fill(first, last, value);
}

template <class ExecutionPolicy, class ForwardIt, class T>
void iota(
    ExecutionPolicy&& policy, ForwardIt first, ForwardIt last, const T& value) {
  std::iota(first, last, value);
}

template <class ExecutionPolicy, class ForwardIt, class OutputIt>
auto inclusive_scan(
    ExecutionPolicy&& policy,
    ForwardIt first,
    ForwardIt last,
    OutputIt d_first) {
  return std::inclusive_scan(first, last, d_first);
}

template <class ExecutionPolicy, class ForwardIt, class OutputIt>
auto exclusive_scan(
    ExecutionPolicy&& policy,
    ForwardIt first,
    ForwardIt last,
    OutputIt d_first) {
  return std::exclusive_scan(first, last, d_first);
}

template <class ExecutionPolicy, class ForwardIt>
std::pair<ForwardIt, ForwardIt> minmax_element(ForwardIt first, ForwardIt last);
#endif
}  // namespace stdx

#endif  // EXECUTION_POLICY_H
