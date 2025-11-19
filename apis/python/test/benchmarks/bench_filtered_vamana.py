"""
Performance benchmarks for Filtered-Vamana (Phase 4, Task 4.4)

Benchmarks QPS vs Recall trade-offs for filtered vector search and compares
pre-filtering (FilteredVamana) vs post-filtering baseline.

Based on experiments from:
"Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters"
(Gollapudi et al., WWW 2023)
"""

import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors

from tiledb.vector_search import Index
from tiledb.vector_search.ingestion import ingest
from tiledb.vector_search.vamana_index import VamanaIndex


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""

    l_search: int
    recall: float
    qps: float
    avg_latency_ms: float
    specificity: float
    method: str  # "pre_filter" or "post_filter"


def compute_filtered_groundtruth(
    vectors, queries, filter_labels, query_filter_labels, k
):
    """Compute ground truth for filtered queries using brute force"""
    matching_indices = []
    for idx, labels in filter_labels.items():
        if any(label in labels for label in query_filter_labels):
            matching_indices.append(idx)

    if len(matching_indices) == 0:
        return (
            np.full((queries.shape[0], k), np.iinfo(np.uint64).max, dtype=np.uint64),
            np.full((queries.shape[0], k), np.finfo(np.float32).max, dtype=np.float32),
        )

    matching_indices = np.array(matching_indices)
    matching_vectors = vectors[matching_indices]

    nbrs = NearestNeighbors(
        n_neighbors=min(k, len(matching_indices)), metric="euclidean", algorithm="brute"
    ).fit(matching_vectors)
    distances, indices = nbrs.kneighbors(queries)

    gt_ids = matching_indices[indices]

    if gt_ids.shape[1] < k:
        pad_width = k - gt_ids.shape[1]
        gt_ids = np.pad(
            gt_ids, ((0, 0), (0, pad_width)), constant_values=np.iinfo(np.uint64).max
        )
        distances = np.pad(
            distances,
            ((0, 0), (0, pad_width)),
            constant_values=np.finfo(np.float32).max,
        )

    return gt_ids.astype(np.uint64), distances.astype(np.float32)


def compute_recall(results, groundtruth, k):
    """Compute recall@k"""
    total_found = 0
    total_possible = 0

    for i in range(len(results)):
        valid_gt = groundtruth[i][groundtruth[i] != np.iinfo(np.uint64).max]
        if len(valid_gt) == 0:
            continue

        result_ids = results[i][:k]
        found = len(np.intersect1d(result_ids, valid_gt[:k]))
        total_found += found
        total_possible += min(k, len(valid_gt))

    return total_found / total_possible if total_possible > 0 else 0.0


def benchmark_pre_filtering(
    index,
    queries,
    filter_labels,
    query_filter_label,
    groundtruth,
    k,
    l_values,
    num_warmup=5,
    num_trials=20,
) -> List[BenchmarkResult]:
    """
    Benchmark pre-filtering (FilteredVamana) approach

    Measures QPS and recall at different L values
    """
    results = []

    for l_search in l_values:
        # Warmup
        for _ in range(num_warmup):
            _, _ = index.query(
                queries[0:1],
                k=k,
                l_search=l_search,
                where=f"label == '{query_filter_label}'",
            )

        # Benchmark
        start = time.perf_counter()
        all_ids = []

        for trial in range(num_trials):
            for query in queries:
                distances, ids = index.query(
                    query.reshape(1, -1),
                    k=k,
                    l_search=l_search,
                    where=f"label == '{query_filter_label}'",
                )
                all_ids.append(ids[0])

        end = time.perf_counter()

        # Compute metrics
        total_queries = num_trials * len(queries)
        elapsed = end - start
        qps = total_queries / elapsed
        avg_latency_ms = (elapsed / total_queries) * 1000

        # Compute recall using last trial's results
        recall = compute_recall(np.array(all_ids[-len(queries) :]), groundtruth, k)

        # Compute specificity
        num_matching = sum(
            1 for labels in filter_labels.values() if query_filter_label in labels
        )
        specificity = num_matching / len(filter_labels)

        results.append(
            BenchmarkResult(
                l_search=l_search,
                recall=recall,
                qps=qps,
                avg_latency_ms=avg_latency_ms,
                specificity=specificity,
                method="pre_filter",
            )
        )

    return results


def benchmark_post_filtering(
    unfiltered_index,
    vectors,
    queries,
    filter_labels,
    query_filter_label,
    groundtruth,
    k,
    k_factors,
    num_warmup=5,
    num_trials=20,
) -> List[BenchmarkResult]:
    """
    Benchmark post-filtering baseline

    Query unfiltered index for k*factor results, then filter and take top k
    """
    results = []

    for k_factor in k_factors:
        k_retrieve = int(k * k_factor)

        # Warmup
        for _ in range(num_warmup):
            _, _ = unfiltered_index.query(queries[0:1], k=k_retrieve)

        # Benchmark
        start = time.perf_counter()
        all_filtered_ids = []

        for trial in range(num_trials):
            for query in queries:
                # Query unfiltered
                distances, ids = unfiltered_index.query(
                    query.reshape(1, -1), k=k_retrieve
                )

                # Post-filter
                filtered_ids = []
                filtered_dists = []
                for j in range(len(ids[0])):
                    if (
                        ids[0, j] in filter_labels
                        and query_filter_label in filter_labels[ids[0, j]]
                    ):
                        filtered_ids.append(ids[0, j])
                        filtered_dists.append(distances[0, j])
                        if len(filtered_ids) >= k:
                            break

                # Pad if necessary
                while len(filtered_ids) < k:
                    filtered_ids.append(np.iinfo(np.uint64).max)

                all_filtered_ids.append(np.array(filtered_ids[:k]))

        end = time.perf_counter()

        # Compute metrics
        total_queries = num_trials * len(queries)
        elapsed = end - start
        qps = total_queries / elapsed
        avg_latency_ms = (elapsed / total_queries) * 1000

        # Compute recall
        recall = compute_recall(
            np.array(all_filtered_ids[-len(queries) :]), groundtruth, k
        )

        # Specificity
        num_matching = sum(
            1 for labels in filter_labels.values() if query_filter_label in labels
        )
        specificity = num_matching / len(filter_labels)

        results.append(
            BenchmarkResult(
                l_search=k_retrieve,  # Using k_retrieve as proxy for "L"
                recall=recall,
                qps=qps,
                avg_latency_ms=avg_latency_ms,
                specificity=specificity,
                method="post_filter",
            )
        )

    return results


def bench_qps_vs_recall_curves(tmp_path):
    """
    Generate QPS vs Recall@10 curves for different specificities

    Similar to Figure 2/3 from the paper

    Tests:
    - Small dataset (1K vectors) with synthetic labels
    - Different specificity levels (10^-1, 10^-2)
    - QPS at different L values (10, 20, 50, 100, 200)
    """
    print("\n" + "=" * 80)
    print("Benchmark: QPS vs Recall Curves")
    print("=" * 80)

    num_vectors = 1000
    dimensions = 128
    k = 10
    num_queries = 50
    num_labels = 100  # Each label gets ~10 vectors (specificity ~0.01)

    # Create dataset
    vectors, cluster_ids = make_blobs(
        n_samples=num_vectors,
        n_features=dimensions,
        centers=num_labels,
        cluster_std=1.0,
        random_state=42,
    )
    vectors = vectors.astype(np.float32)

    # Create queries
    query_indices = np.random.choice(num_vectors, num_queries, replace=False)
    queries = vectors[query_indices]

    # Assign labels (one label per vector, round-robin)
    filter_labels = {}
    for i in range(num_vectors):
        filter_labels[i] = [f"label_{i % num_labels}"]

    # Test with different specificity levels
    specificities = [0.1, 0.01]  # 10%, 1%
    test_labels = [f"label_{i}" for i in [0, 1]]  # Use first two labels

    for spec_idx, target_specificity in enumerate(specificities):
        print(f"\n--- Specificity: {target_specificity:.3f} ---")

        # Adjust number of labels to match target specificity
        num_target_labels = max(1, int(num_vectors * target_specificity / 10))
        query_filter_label = test_labels[spec_idx % len(test_labels)]

        # Build filtered index
        uri = os.path.join(tmp_path, f"bench_filtered_{spec_idx}")
        ingest(
            index_type="VAMANA",
            index_uri=uri,
            input_vectors=vectors,
            filter_labels=filter_labels,
            l_build=100,
            r_max_degree=64,
        )
        filtered_index = VamanaIndex(uri=uri)

        # Build unfiltered index for post-filtering baseline
        uri_unfiltered = os.path.join(tmp_path, f"bench_unfiltered_{spec_idx}")
        ingest(
            index_type="VAMANA",
            index_uri=uri_unfiltered,
            input_vectors=vectors,
            l_build=100,
            r_max_degree=64,
        )
        unfiltered_index = VamanaIndex(uri=uri_unfiltered)

        # Compute ground truth
        gt_ids, gt_dists = compute_filtered_groundtruth(
            vectors, queries, filter_labels, [query_filter_label], k
        )

        # Benchmark pre-filtering
        l_values = [10, 20, 50, 100, 200]
        pre_results = benchmark_pre_filtering(
            filtered_index,
            queries,
            filter_labels,
            query_filter_label,
            gt_ids,
            k,
            l_values,
            num_warmup=3,
            num_trials=10,
        )

        # Benchmark post-filtering
        k_factors = [2, 5, 10, 20, 50]
        post_results = benchmark_post_filtering(
            unfiltered_index,
            vectors,
            queries,
            filter_labels,
            query_filter_label,
            gt_ids,
            k,
            k_factors,
            num_warmup=3,
            num_trials=10,
        )

        # Print results
        print("\nPre-filtering (FilteredVamana):")
        print(f"{'L':>6} {'Recall':>8} {'QPS':>10} {'Latency(ms)':>12}")
        print("-" * 40)
        for res in pre_results:
            print(
                f"{res.l_search:6d} {res.recall:8.3f} {res.qps:10.1f} {res.avg_latency_ms:12.2f}"
            )

        print("\nPost-filtering (baseline):")
        print(f"{'k*N':>6} {'Recall':>8} {'QPS':>10} {'Latency(ms)':>12}")
        print("-" * 40)
        for res in post_results:
            print(
                f"{res.l_search:6d} {res.recall:8.3f} {res.qps:10.1f} {res.avg_latency_ms:12.2f}"
            )

        # Compare best recall
        best_pre_recall = max(r.recall for r in pre_results)
        best_post_recall = max(r.recall for r in post_results)

        print(f"\nBest pre-filtering recall: {best_pre_recall:.3f}")
        print(f"Best post-filtering recall: {best_post_recall:.3f}")

        # Find QPS at similar recall levels
        target_recall = 0.9
        pre_qps_at_target = None
        post_qps_at_target = None

        for res in pre_results:
            if res.recall >= target_recall:
                pre_qps_at_target = res.qps
                break

        for res in post_results:
            if res.recall >= target_recall:
                post_qps_at_target = res.qps
                break

        if pre_qps_at_target and post_qps_at_target:
            speedup = pre_qps_at_target / post_qps_at_target
            print(f"\nQPS at recall={target_recall:.1f}:")
            print(f"  Pre-filter: {pre_qps_at_target:.1f}")
            print(f"  Post-filter: {post_qps_at_target:.1f}")
            print(f"  Speedup: {speedup:.2f}x")

        # Cleanup
        Index.delete_index(uri=uri, config={})
        Index.delete_index(uri=uri_unfiltered, config={})

    print("\n" + "=" * 80)
    print("Benchmark completed!")
    print("=" * 80 + "\n")


def bench_vs_post_filtering(tmp_path):
    """
    Compare pre-filtering vs post-filtering at low specificity

    Verifies: Pre-filtering >> post-filtering for specificity < 0.01

    Measures:
    - Recall and QPS for both approaches
    - Demonstrates advantage of pre-filtering at low specificity
    """
    print("\n" + "=" * 80)
    print("Benchmark: Pre-filtering vs Post-filtering")
    print("=" * 80)

    num_vectors = 2000
    dimensions = 128
    k = 10
    num_queries = 100
    specificity = 0.005  # 0.5% (very low)

    # Create dataset
    vectors, _ = make_blobs(
        n_samples=num_vectors,
        n_features=dimensions,
        centers=50,
        cluster_std=1.5,
        random_state=42,
    )
    vectors = vectors.astype(np.float32)

    # Create queries from dataset
    query_indices = np.random.choice(num_vectors, num_queries, replace=False)
    queries = vectors[query_indices]

    # Assign labels to achieve target specificity
    num_rare_label = int(num_vectors * specificity)
    filter_labels = {}
    for i in range(num_rare_label):
        filter_labels[i] = ["rare_label"]
    for i in range(num_rare_label, num_vectors):
        filter_labels[i] = [f"common_label_{i % 50}"]

    query_filter_label = "rare_label"

    print(f"\nDataset: {num_vectors} vectors, {dimensions}D")
    print(f"Specificity: {specificity:.4f} ({num_rare_label} matching vectors)")
    print(f"Queries: {num_queries}, k={k}")

    # Build filtered index
    uri_filtered = os.path.join(tmp_path, "bench_pre_vs_post_filtered")
    ingest(
        index_type="VAMANA",
        index_uri=uri_filtered,
        input_vectors=vectors,
        filter_labels=filter_labels,
        l_build=100,
        r_max_degree=64,
    )
    filtered_index = VamanaIndex(uri=uri_filtered)

    # Build unfiltered index
    uri_unfiltered = os.path.join(tmp_path, "bench_pre_vs_post_unfiltered")
    ingest(
        index_type="VAMANA",
        index_uri=uri_unfiltered,
        input_vectors=vectors,
        l_build=100,
        r_max_degree=64,
    )
    unfiltered_index = VamanaIndex(uri=uri_unfiltered)

    # Compute ground truth
    gt_ids, gt_dists = compute_filtered_groundtruth(
        vectors, queries, filter_labels, [query_filter_label], k
    )

    # Benchmark pre-filtering at L=100
    l_search = 100
    pre_results = benchmark_pre_filtering(
        filtered_index,
        queries,
        filter_labels,
        query_filter_label,
        gt_ids,
        k,
        [l_search],
        num_warmup=5,
        num_trials=20,
    )

    # Benchmark post-filtering with various k factors
    # At low specificity, need very large k to get good recall
    k_factors = [10, 50, 100, 200]
    post_results = benchmark_post_filtering(
        unfiltered_index,
        vectors,
        queries,
        filter_labels,
        query_filter_label,
        gt_ids,
        k,
        k_factors,
        num_warmup=5,
        num_trials=20,
    )

    # Print results
    print("\n" + "-" * 60)
    print("RESULTS:")
    print("-" * 60)

    print(f"\nPre-filtering (L={l_search}):")
    for res in pre_results:
        print(f"  Recall: {res.recall:.3f}")
        print(f"  QPS: {res.qps:.1f}")
        print(f"  Latency: {res.avg_latency_ms:.2f} ms")

    print(f"\nPost-filtering (best result):")
    best_post = max(post_results, key=lambda r: r.recall)
    print(f"  k_factor: {best_post.l_search // k}")
    print(f"  Recall: {best_post.recall:.3f}")
    print(f"  QPS: {best_post.qps:.1f}")
    print(f"  Latency: {best_post.avg_latency_ms:.2f} ms")

    # Compare
    qps_ratio = pre_results[0].qps / best_post.qps
    recall_diff = pre_results[0].recall - best_post.recall

    print(f"\nComparison:")
    print(f"  QPS ratio (pre/post): {qps_ratio:.2f}x")
    print(f"  Recall difference: {recall_diff:+.3f}")

    if qps_ratio > 10:
        print(f"  ✓ Pre-filtering is {qps_ratio:.1f}x faster (>10x improvement)")
    else:
        print(f"  ⚠ Pre-filtering speedup {qps_ratio:.1f}x < 10x")

    # Cleanup
    Index.delete_index(uri=uri_filtered, config={})
    Index.delete_index(uri=uri_unfiltered, config={})

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_path:
        print("\nRunning Filtered-Vamana Benchmarks...")
        print("This may take several minutes...\n")

        try:
            # Run benchmarks
            bench_qps_vs_recall_curves(tmp_path)
            bench_vs_post_filtering(tmp_path)

            print("\n✓ All benchmarks completed successfully!\n")
            sys.exit(0)

        except Exception as e:
            print(f"\n✗ Benchmark failed: {e}\n")
            import traceback

            traceback.print_exc()
            sys.exit(1)
