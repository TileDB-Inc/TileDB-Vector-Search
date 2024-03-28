# PQ Encoding

Product quantization (pq) is a method for compressing vectors in a vector database. The basic idea is to divide the vectors into some number of subvectors and then to use the nearest centroid to each subvector to represent the original vector. Since there are a fixed number of actual centroids, we only need to store the index of the centroid.
For example, if we have 256 centroids, we can represent each centroid with an 8-bit index. If we subdivide a vector into, say, eight subvectors, we can represent each subvector with 8bits, for a total of 64bits for the entire vector. If the original vector comprised 128 floats, this is a compression of (128 \* 32 ) / 64 -- a factor of 64.

In addition to space savings, distance computation can also be greatly accelerated. If we are using the "L2" distance -- which is really the sum of squares -- the distance between two vectors is the sum of the distance between each of its subvectors. Since each subvector is represented by the index of the nearest centroid, we can store those distances in a lookup table and access them with a simple lookup.

The tutorial at https://www.pinecone.io/learn/series/faiss/product-quantization/ explains the general idea of product quantization in more detail.

To realize PQ in TileDB-Vector-Search, we implement the following:

- The `flatpq_index` constructor defines the number of subvectors and bits per subvector (equivalent to the number of centroids).
- `train()` runs a kmeans algorithm to compute the centroids for each subspace. Each subspace has its own set of centroids. For each subspace, a 2D table is created holding the distance between each pair of centroids in the subspace.
- `add()` takes a set of input vectors and runs `qv_partition` on each subspace to obtain the codes.
- `symmetri_query()` first encodes the query vectors and then calls `flat::qv_query_heap` on the encoded queries, using the symmetric pq distance function defined in the class. They symmetric distance function uses the lookup tables for computing distances.
- `asymmetric_query()` calls `flat::qv_query_heap` directly on the query vectors, using the asymmetric pq distance function defined in the class. The asymmetric distance function decodes the encoded vector and computes the distance in the normal fashion.
- `write_index` saves the index information. This needs to be brought up to speed by defining index group and index metadata classes.
- The "loading" constructor reads an index -- it also needs to be brought up to speed.
