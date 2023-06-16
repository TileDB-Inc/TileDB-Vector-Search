## Glossary of Terms

This glossary is intended to describe the terms used in this project to facilitate communication.  Right now these are organized alphabetically, but we might want to put different terms into different subsections.  However, since this is only for internal consumption, additional organization is probably not necessary.

Please feel free to suggest better terminology.  Whatever terms we choose, it is extremely important that we all use the same terminogy.

Also included below is suggested naming of similarity search notions to C++ abstractions.  Initially these will just be datatypes and named template parameters.  Later we can create actual concepts.

* **approximate nearest neighbor (ANN)** Search algorithms not guaranteed to find the exact nearest neighbor in a similarity search, but instead returns a point that is expected to be close to the true nearest neighbor.

* **embedding** A mapping of features from a data item in a dataset to a finite-dimensional vector.  cf https://www.pinecone.io/learn/vector-embeddings-for-developers/ .  Embeddings can be generated from ML models and represent the outputs of the final stage of the model (just before computation of the loss function).  The motivation for embeddings is to reduce the dimensionality of the original items in such a way that original items that are similar to each other (according to designated features) have corresponding vector embeddings that are also close to each other.

* **feature space:** A mathematical representation of a set of objects or data points, where each object is described by a set of *features* (or variables) that are important for a particular task or analysis.

* **flat:** Index corresponding to brute force search for similarity between query vectors and database vectors. A flat index is implicit; it is a direct indexing into the vector database. A search using a flat index consists of computing scores between all query vectors and all database vectors. 
 
* **index:** A data structure for accelerating the lookup of vectors in a vector database that are closest to a given query vector.

* **inverted file index:** An index that looks up a set of partitions (or clusters) in the vector database, given query vector.  Each cluster is presumed to contain the closest vectors to the query vector.  Flat search is performed over each of the clusters that are found.   

* **k-means:**  A clustering algorithm that computes `k` clusters of vectors such that the vectors in each cluster are closer to the mean of the cluster (the centroid) than to any other centroids.  The algorithm begins with a set of `k` seed vectors as the initial centroids from which it computes initial cluster.  It then iteratively updates the centroids based on the current clustering and based on the new centroids, computes new clustering.  The algorithm terminates either when two successive updates to the centroids are sufficiently small, or when a maximum number of iterations are achieved.

* **L2** A similarity metric based on the Euclidean distance between two vectors.  Computed as $$\sqrt{\sum_i (x_i - y_i)^2}$$ 

* **nearest neighbor:**  The vector in a vector database that is closer than any other vector in the database to a given query vector. 

* **product quantization (PQ)** A lossy algorithm for reducing the dimensionality of the vectors in a vector database.

* **query** A process (algorithm) for finding particular data in a database.  In our case, the process consists of presenting a single vector (the *query vector*) or a set of query vectors and finding the members of the vector database that are closest in terms of some giving similarity metric (typically L2 norm, cosine / dot product, or Jaccard).  Naming of query methods often consistes of the index type, the metric type, and the quantization type.  E.g., `IVFFlat` uses an inverted file index for first lookup of clusters and then a flat search over the vectors in the clusters.

* **similarity:** A metric of closeness between two vectors.

* **similarity search:**

* **vector** A one-dimension container of elements that can be indexed to retrieve an element.  The index values are assumed to be in the range of `[0, N)`.  `N` is the *dimension* (or *size* or *length*) of the vector.  This should not be confused with `std::vector` and we should explicitly refer to `std::vector` when that is what we actually mean.

* **vector database** A collection of vectors representing embeddings from a given corpus of data.  These are the complete set of vectors that we want to query.  We can talk about the vector database being stored in a TileDB array or being held in memory.  

## Reflecting Terminology in C++ Abstractions

### Enums

Faiss naming of our highest-priority algorithms: 
*  indexFlatL2
*  indexIVFFlat
*  indexIVFPQ

Milvus naming:
* Flat
* IVF_Flat
* IVF_PQ


For the C++ implementation, our naming:
* `flat` -- can use L2, cosine, jaccard comparisons; does all-all; stores flat vectors
* `ivf_flat` -- uses inverted file index and flat vector storage (and exact search)
* `ivf_pq` -- uses inverted file index and pq vector storage

### Data types

### Functions

High-level pseudo description of interfaces (Python but good C++ should be similar)

Faiss:
```
xb = ... # database
xq = ... # query
index = faiss.IndexFlatL2(d)
index.add(xb)
D, I = index.search(xq, k)  # Returns Data and Indices for nearest neighbors
```

Or

```
xb = ... # database
xq = ... # query
nlist = 50  # how many cells
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)nlist = 50  # how many cells
index.train(xb)  # or could be separate, smaller, training set
index.add(xb)
D, I = index.search(xq, k)
```

Milvus:

```
status, results = conn.search(collection_name, query_records, top_k, params)
```


### C++

```
// One dimensional storage. A lightweight wrapper around span, but that owns its storage 
template <class T>
class Vector;

// Two dimensional storage. A lightweight wrapper around span, but that owns its storage 
template <class T>
class Matrix;

// Feature vector type.  Vectors can be used for more than feature vectors
template <class T>
using feature_vector_type = Vector<T>;   

// Collection of feature vectors
template <class T>
using feature_vector_list = Matrix<T>;   

// TBD
template <class T, class I>
class index_class;

// Do similar
template <class T, class I>
std::tuple<feature_vector_list<T>, Matrix<I>> flat_query(const feature_vector_list<T>& collection, 
                                               const feature_vector_list<T>& query_records,
                                               size_t k); 

std::tuple<feature_vector_list<T>, Matrix<I>> ivf_query(const index_class_type& index,
                                               const feature_vector_list<T>& collection, 
                                               const feature_vector_list<T>& query_records,
                                               size_t nprobe, size_t k); 
```

