# `VamanaIndex`

The `VamanaIndex` class implements the approach to indexing and querying presented in a series of papers about Microsoft's DiskANN vector search library. The papers are:

```
  Subramanya, Suhas Jayaram, and Rohan Kadekodi. DiskANN: Fast Accurate Billion-Point Nearest Neighbor Search on a Single Node.

  Singh, Aditi, et al. FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search. arXiv:2105.09613, arXiv, 20 May 2021, http://arxiv.org/abs/2105.09613.

  Gollapudi, Siddharth, et al. “Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters.” Proceedings of the ACM Web Conference 2023, ACM, 2023, pp. 3406–16, https://doi.org/10.1145/3543507.3583552.
```

The three papers present an evolution of DiskANN. The first is the original paper, the second adds searching with updates, and the last adds filtering.

DiskANN uses a graph-based indexing scheme along with a highly efficient out-of-core process (assuming low-latency, high-bandwidth, SSD storage) for searching graphs that are larger than main memory. The idea with a graph-based index is to create a graph where nodes in the graph represent the vectors to be searched. The graph is constructed (during ingestion) to have bounded vertex degree and small network diameter (a "small world" graph as originally developed by Watts and Strogatz).

There are two fundamental operations in a vector search system: creating the index ("ingestion"), and querying the index (search). The process for ingestion and for search both rely on a "greedy search" algorithm. Ingestion further relies on a "robust prune" algorithm.

We note that the description of the ingestion process changed between the second and the third paper. It originally was based on construction a random nearest neighbor graph and then iterating over it with greedy search and robust prune to obtain, essentially, a "small worlds" graph. The third paper implements ingestion by simply invoking node insertion for all of the input vectors.

## Greedy Search

Searching for the nearest neighbors of a given query is accomplished using a bounded best-first search. A frontier with a given number (`L`) of nodes is advanced by exploring the unvisited neighbors of the node closest to the query. Nodes are added to the frontier if they are closer to the query vector than the furthest node in the frontier. When there are no nodes that are closer than the nodes in the frontier, the algorithm terminates.

We experimented with several algorithms for implementing greedy search (aka best-first search): The "greedy search" algorithm as presented in the DiskANN papers and a number of best-first implementations of our own design.

### DiskANN Vamana Greedy Search

```
GreedySearch(Graph G, node source, query q, result_size k, frontier_size L) {
  𝓛 <- {s} and 𝓥 ← ∅   // 𝓛 is the frontier set, 𝓥 is the set of visited nodes
  while ( 𝓛 \ 𝓥 ≠ ∅ ) { // 𝓛 \ 𝓥 is the active frontier: the frontier minus visited nodes
    p* ← node in active frontier closest to q
    𝓛 ← 𝓛 ∪ { unvisited neighbors of p* }
    𝓥 ← 𝓥 ∪ p* // Put p* in visited set
    Trim 𝓛 to keep only L closest nodes to q
  }
}
```

In our initial implementation we used a a `k_min_heap` to represent 𝓛, an `std::unordered_set` to represent 𝓥, and a`k_min_heap` to represent 𝓛 \ 𝓥. By setting the max size of the `k_min_heap` for 𝓛 to be the frontier size, we can maintain that on the fly without a separate "trim" operation.

```
template </* SearchPath SP, */ class Distance = sum_of_squares_distance>
auto greedy_search(auto&& graph, auto&& db, id_type source, auto&& query, size_t k_nn, size_t L) {
  std::unordered_set<id_type> visited_vertices;      // 𝓥
  auto result = k_min_heap<score_type, id_type>{L};  // 𝓛
  auto q1 = k_min_heap<score_type, id_type>{L};      // 𝓛 \ 𝓥
  auto q2 = k_min_heap<score_type, id_type>{L};

  while (!q1.empty()) {
    // Remove smallest element from q1
    visited_vertices.insert(p_star);
    // Copy unvisited vertices from result to q2
    // Visit neighbors of p_star, inserting unvisited neighbors into q2
    q1.swap(q2);
    q2.clear();
  }
}
```

We can observe a few things about this (somewhat naive) implementation. First, there isn't any need to keep a queue of the active frontier. Only a single node is pulled from active frontier (and moved from 𝓛 to 𝓥). Subsequent smallest nodes can also be pulled from 𝓛 -- as long as they are not in the visited set.

```
template <class Graph, feature_vector_array A, feature_vector V, class Distance = sum_of_squares_distance>
auto best_first_O4(const Graph& graph, const A& db, id_type source, const V& query, size_t k_nn, size_t Lmax) {
  std::unordered_set<id_type> visited_vertices;      // 𝓥
  auto pq = k_min_heap<score_type, id_type>{Lmax};   // 𝓛
  std::vector<uint8_t> vertex_state_property_map(graph.num_vertices(), 0); // bitmap

  id_type p_star = source;
  do {
    visited_vertices.insert(p_star);
    for (auto&& [_, neighbor_id] : graph[p_star]) { // For each neighbor of p*
      // if unvisited, attempt to insert into pq
      // if insertion succeeds and evicts another node, mark it as evicted
      p_star ← smallest unvisited node in pq
    }
  } while(p_star != max_int);
}
```

In profiling previsou implementation, the lookups of node state in the bitmap consumed a substantial fraction of time. This was improved by using a bitmap rather than a `std::unordered_set`. However, there is a cost in terms of memory usage with that approach.
