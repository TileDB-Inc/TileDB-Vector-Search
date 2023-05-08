### Terminology

Faiss naming of algorithms:
*  indexFlatL2
*  indexIVFFlat
*  indexIVFPQ


For the C++ implementation, our naming:
* flat -- can use L2, cosine, jaccard comparisons; does all-all; stores flat vectors
* ivf_flat -- uses inverted file index and flat vector storage
* ivf_pq -- uses inverted file index and pq vector storage
