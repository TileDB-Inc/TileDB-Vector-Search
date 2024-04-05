## Scoring functions

This directory contains implementations of distance computations: L2 (sum of squares), inner product, and cosine.
For each type of distance computation, naive, 4x unrolled, AVX, and BLAS implementations are provided.
In addition, for each of the preceding, versions that compute the distance over a specified view of the
two vectors is also provided.

### Naive implementations

The "naive" implementations are just simple loops over two vectors, e.g.,

```
template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, float> &&
           std::same_as<typename W::value_type, float>
inline float naive_sum_of_squares(const V& a, const W& b) {
  size_t size_a = size(a);
  float sum = 0.0;
  for (size_t i = 0; i < size_a; ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}
```

All of the distance functions are templated on the type of vector they compute over
accept and the vectors are required to meet the requirements of `feature_vector`.
Because of the need to case non-`float` elements, there are four concept-based
overloads for each function, depending on the `value_type` of each vector. The overloads
are for `float`-`float`, `float`-`uint8_t`, `uint8_t`-`float`, and `uint8_t`-`uint8_t`.

An overload for `float`-`uint8_t`

```
template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, float> &&
           std::same_as<typename W::value_type, uint8_t>
inline float naive_sum_of_squares(const V& a, const W& b) {
  size_t size_a = size(a);
  float sum = 0.0;
  for (size_t i = 0; i < size_a; ++i) {
    float diff = a[i] - (float) b[i];
    sum += diff * diff;
  }
  return sum;
}
```

### Unrolled

There are unrolled versions of the distance functions, which use a very basic
unrolling to provide a moderate performance optimization.

```c++
template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, float> &&
           std::same_as<typename W::value_type, float>
inline float unroll4_sum_of_squares(const V& a, const W& b) {
  size_t size_a = size(a);
  size_t stop = 4 * (size_a / 4);
  float sum = 0.0;
  for (size_t i = 0; i < stop; i += 4) {
    float diff0 = a[i + 0] - b[i + 0];
    float diff1 = a[i + 1] - b[i + 1];
    float diff2 = a[i + 2] - b[i + 2];
    float diff3 = a[i + 3] - b[i + 3];
    sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
  }

  // Clean up
  for (size_t i = stop; i < size_a; ++i) {
    float diff0 = a[i + 0] - b[i + 0];
    sum += diff0 * diff0;
  }
  return sum;
}
```

### Distance over a view

Overloads of the distance functions are also provided to compute distance over
just a (contiguous) portion of two vectors.

```c++
template <feature_vector V, feature_vector W>
  requires std::same_as<typename V::value_type, float> &&
           std::same_as<typename W::value_type, float>
inline float naive_sum_of_squares(
    const V& a, const W& b, size_t start, size_t stop) {
  float sum = 0.0;
  for (size_t i = start; i < stop; ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}
```

### AVX

The non-view L2 distance functions have been implemented with AVX2 instructions, which
can provide a substantial performance improvement (8X to 10X) over plain C++.
Using intrinsics, the basic body of the distance function is quite straightforward:

```c++
  for (size_t i = start; i < stop; i += 8) {
    // Load 8 floats
    __m256 vec_a = _mm256_loadu_ps(a_ptr + i + 0);
    __m256 vec_b = _mm256_loadu_ps(b_ptr + i + 0);

    // Compute the difference
    __m256 diff = _mm256_sub_ps(vec_a, vec_b);

    // Square and accumulate
    vec_sum = _mm256_fmadd_ps(diff, diff, vec_sum);
  }
```

This loads 8 `float`s from vectors `a` and `b` into 256-bit registers
and computes the pairwise distance between 8 floats in (SIMD) parallel.

The 8 `float`s need to be reduced to a single `float`:

```
// 8 to 4
__m128 lo = _mm256_castps256_ps128(vec_sum);
__m128 hi = _mm256_extractf128_ps(vec_sum, 1);
__m128 combined = _mm_add_ps(lo, hi);

// 4 to 2
combined = _mm_hadd_ps(combined, combined);

// 2 to 1
combined = _mm_hadd_ps(combined, combined);

float sum = _mm_cvtss_f32(combined);
```

### Function objects

To enable different distance functions to be passed as parameters to
queries, the file `scoring.h` defines some function objects to wrap the
raw functions:

```
namespace _l2_distance {
struct sum_of_squares_distance {
  template <feature_vector V, feature_vector U>
  constexpr inline float operator()(const V& a, const U& b) const {
    return avx2_sum_of_squares(a, b);
  }
};
}  // namespace _l2_distance

using sum_of_squares_distance = _l2_distance::sum_of_squares_distance;
inline constexpr auto l2_distance = _l2_distance::sum_of_squares_distance{};
```

With these definitions, users can compute L2 distance by calling `l2_distance()`:

```
  auto dist = l2_distance(x, y);
```

Users can pass the type `sum_of_squared_distance` where it is required as a template argument, for instance.

### Functions parameterized by distance function

All of the functions in the TileDB-Vector-Search library that use a distance function have been
parameterized by the type of the distance function. Except for the pq related functions, they
also take the distance function as an argument.

For example

```c++
template <
    feature_vector_array F,
    feature_vector_array Q,
    class Distance = sum_of_squares_distance>
auto query_finite_ram(
    F& partitioned_vectors,  // not const because of load()
    const Q& query,
    auto&& active_queries,
    size_t k_nn,
    size_t upper_bound,
    size_t nthreads,
    Distance distance = Distance{});
```

In the body of the code, distance is computed simply by invoking `distance`.

Query functions are invoked simply by passing the distance function object as a parameter.

```c++
nuv_query_heap_finite_ram(inputs, query, active_queries,
                          k_nn, upper_bound, nthreads,
                          counting_l2_distance);
```

_Comments:_

- We need to get rid of the `nthreads` parameter throughout the entire code.
- We may need to add timestamping to the functions.
- IMPORTANT: We need to absolutely and positively understand what is happening with the function
  objects vis-a-vis inlining and optimization.

### Implementation status:

| Metric        | Naive | 4x unrolled | AVX | BLAS |
| ------------- | ----- | ----------- | --- | ---- |
| L2            | Y     | Y           | Y   | N    |
| Dot           | Y     | Y           | Y   | N    |
| Cosine        | N     | N           | N   | N    |
| L2 w/view     | Y     | Y           | N   | N    |
| Dot w/view    | N     | N           | N   | N    |
| Cosine w/view | N     | N           | N   | N    |

NOTE: Cosine is just dot using normalized vectors.
One approach to computing cosine similarity is
to first normalize the vectors, rather than
normalizing them on the fly.
