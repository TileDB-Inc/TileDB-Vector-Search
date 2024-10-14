---
title: Storage Format Spec
description: "Learn about the vector search storage format specification for different indexing algorithms."
---

The underlying storage model used for indexing vectors in TileDB-Vector-Search is heavily dependent on the indexing algorithm used. However, there are also high level structures that are used across algorithms.

## Cross algorithm storage format

All data and metadata required for a TileDB-Vector-Search index are stored inside a TileDB group (`index_uri`). All the listed, named arrays below are stored under this URI.

### Index metadata

Metadata values required for configuring the different properties of an index are stored in the `index_uri` group metadata. There are some metadata values that are required for all algorithm implementations as well as per-algorithm specific metadata values. Below is a table of all the metadata values that are recorded for all algorithms.

| Name                   | Description                                                                                                                                                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `dataset_type`         | The asset type for disambiguation in TileDB cloud. Value: `vector_search`                                                                                                                                                      |
| `index_type`           | The index algorithm used for this index. Can be one of the following values: `FLAT`, `IVF_FLAT`, `VAMANA`, `IVF_PQ`                                                                                                            |
| `storage_version`      | The storage version used for the index. The storage version is used to make sure that indexing algorithms can update their storage logic without affecting previously created indexes and maintaining backwards compatibility. |
| `dtype`                | The data type of the vector values.                                                                                                                                                                                            |
| `ingestion_timestamps` | An ordered list of timestamps that correspond to different calls of ingestion and update consolidation through the lifetime of the index.                                                                                      |
| `base_sizes`           | An ordered list of number of vectors in the base index at the different ingestion timestamps.                                                                                                                                  |
| `has_updates`          | Boolean value denoting if there are updates recorded in the updates array.                                                                                                                                                     |

### Object metadata

This is a 1D sparse array with `external_id` as dimension and attributes the user defined metadata attributes for the respective vectors.

#### Basic schema parameters

| **Parameter** | **Value** |
| :------------ | :-------- |
| Array type    | Sparse    |
| Rank          | 1D        |
| Cell order    | Row-major |
| Tile order    | Row-major |

#### Dimensions

| Dimension Name | TileDB Datatype |
| :------------- | :-------------- |
| `external_id`  | `uint64_t`      |

### Updates

TileDB-Vector-Search offers support for updates for all different index algorithms by recording updates outside the main indexing storage structure and periodically consolidating them. This implementation is using the `updates` array, a sparse 1D array with dimension the `external_ids` of the vectors and 1 variable length attribute encoding the vector itself or an empty value if the vector is deleted.

#### Basic schema parameters

| **Parameter** | **Value** |
| :------------ | :-------- |
| Array type    | Sparse    |
| Rank          | 1D        |
| Cell order    | Row-major |
| Tile order    | Row-major |

#### Dimensions

| Dimension Name | TileDB Datatype |
| :------------- | :-------------- |
| `external_id`  | `uint64_t`      |

#### Attributes

| Attribute Name | TileDB Datatype  | Description                                                             |
| :------------- | :--------------- | :---------------------------------------------------------------------- |
| `vector`       | variable `dtype` | Contains the vector value. Empty values correspond to vector deletions. |

## Algorithm specific storage format

### FLAT

#### `shuffled_vectors`

This is a 2D dense array that holds all the vectors with no specific ordering.

#### Basic schema parameters

| **Parameter** | **Value** |
| :------------ | :-------- |
| Array type    | Dense     |
| Rank          | 2D        |
| Cell order    | Col-major |
| Tile order    | Col-major |

#### Dimensions

| Dimension Name | TileDB Datatype | Domain            | Description                                               |
| :------------- | :-------------- | :---------------- | :-------------------------------------------------------- |
| `rows`         | `int32_t`       | `[0, dimensions]` | Corresponds to the vector dimensions.                     |
| `cols`         | `int32_t`       | `[0, MAX_INT32]`  | Corresponds to the vector position in the set of vectors. |

#### Attributes

| Attribute Name | TileDB Datatype | Description                                          |
| :------------- | :-------------- | :--------------------------------------------------- |
| `values`       | `dtype`         | Contains the vector value at the specific dimension. |

#### `shuffled_ids`

This is a 1D dense array that maps vector positions in the `shuffled_vectors` array to `external_ids` of each vector.

#### Basic schema parameters

| **Parameter** | **Value** |
| :------------ | :-------- |
| Array type    | Dense     |
| Rank          | 1D        |
| Cell order    | Col-major |
| Tile order    | Col-major |

#### Dimensions

| Dimension Name | TileDB Datatype | Domain           | Description                                               |
| :------------- | :-------------- | :--------------- | :-------------------------------------------------------- |
| `rows`         | `int32_t`       | `[0, MAX_INT32]` | Corresponds to the vector position in `shuffled_vectors`. |

#### Attributes

| Attribute Name | TileDB Datatype | Description                          |
| :------------- | :-------------- | :----------------------------------- |
| `values`       | `uint64_t`      | Contains the vector's `external_id`. |

### IVF_FLAT

#### Metadata

| Name                | Description                                                                         |
| ------------------- | ----------------------------------------------------------------------------------- |
| `partition_history` | An ordered list of the number of partitions used at different ingestion timestamps. |

#### `partition_centroids`

This is a 2D dense array storing the k-means centroids for the different vector partitions.

#### Basic schema parameters

| **Parameter** | **Value** |
| :------------ | :-------- |
| Array type    | Dense     |
| Rank          | 2D        |
| Cell order    | Col-major |
| Tile order    | Col-major |

#### Dimensions

| Dimension Name | TileDB Datatype | Domain            | Description                             |
| :------------- | :-------------- | :---------------- | :-------------------------------------- |
| `rows`         | `int32_t`       | `[0, dimensions]` | Corresponds to the centroid dimensions. |
| `cols`         | `int32_t`       | `[0, MAX_INT32]`  | Corresponds to the centroid id.         |

#### Attributes

| Attribute Name | TileDB Datatype | Description                                            |
| :------------- | :-------------- | :----------------------------------------------------- |
| `centroids`    | `dtype`         | Contains the centroid value at the specific dimension. |

#### `partition_indexes`

This is a 1D dense array recording the start-end index of each partition of vectors in the `shuffled_vectors` array.

#### Basic schema parameters

| **Parameter** | **Value** |
| :------------ | :-------- |
| Array type    | Dense     |
| Rank          | 1D        |
| Cell order    | Col-major |
| Tile order    | Col-major |

#### Dimensions

| Dimension Name | TileDB Datatype | Domain           | Description                      |
| :------------- | :-------------- | :--------------- | :------------------------------- |
| `rows`         | `int32_t`       | `[0, MAX_INT32]` | Corresponds to the partition id. |

#### Attributes

| Attribute Name | TileDB Datatype | Description                                                                      |
| :------------- | :-------------- | :------------------------------------------------------------------------------- |
| `values`       | `uint64_t`      | Contains to the position of the partition split in the `shuffled_vectors` array. |

#### `shuffled_vectors`

This is a 2D dense array that holds all the vectors. Each vector partition is stored in a consecutive index range of this array.

#### Basic schema parameters

| **Parameter** | **Value** |
| :------------ | :-------- |
| Array type    | Dense     |
| Rank          | 2D        |
| Cell order    | Col-major |
| Tile order    | Col-major |

#### Dimensions

| Dimension Name | TileDB Datatype | Domain            | Description                                               |
| :------------- | :-------------- | :---------------- | :-------------------------------------------------------- |
| `rows`         | `int32_t`       | `[0, dimensions]` | Corresponds to the vector dimensions.                     |
| `cols`         | `int32_t`       | `[0, MAX_INT32]`  | Corresponds to the vector position in the set of vectors. |

#### Attributes

| Attribute Name | TileDB Datatype | Description                                          |
| :------------- | :-------------- | :--------------------------------------------------- |
| `values`       | `dtype`         | Contains the vector value at the specific dimension. |

#### `shuffled_ids`

This is a 1D dense array that maps vector indices in the `shuffled_vectors` array to `external_ids` of each vector.

#### Basic schema parameters

| **Parameter** | **Value** |
| :------------ | :-------- |
| Array type    | Dense     |
| Rank          | 1D        |
| Cell order    | Col-major |
| Tile order    | Col-major |

#### Dimensions

| Dimension Name | TileDB Datatype | Domain           | Description                                               |
| :------------- | :-------------- | :--------------- | :-------------------------------------------------------- |
| `rows`         | `int32_t`       | `[0, MAX_INT32]` | Corresponds to the vector position in `shuffled_vectors`. |

#### Attributes

| Attribute Name | TileDB Datatype | Description                        |
| :------------- | :-------------- | :--------------------------------- |
| `values`       | `uint64_t`      | Contains the vector `external_id`. |

### VAMANA

#### Metadata

| Name           | Description                                                    |
| -------------- | -------------------------------------------------------------- |
| `l_build`      | The `l_build` parameter used when constructing the graph.      |
| `r_max_degree` | The `r_max_degree` parameter used when constructing the graph. |

#### `shuffled_vectors`

This is a 2D dense array that holds all the vectors. Each vector partition is stored in a consecutive index range of this array.

#### Basic schema parameters

| **Parameter** | **Value** |
| :------------ | :-------- |
| Array type    | Dense     |
| Rank          | 2D        |
| Cell order    | Col-major |
| Tile order    | Col-major |

#### Dimensions

| Dimension Name | TileDB Datatype | Domain            | Description                                               |
| :------------- | :-------------- | :---------------- | :-------------------------------------------------------- |
| `rows`         | `int32_t`       | `[0, dimensions]` | Corresponds to the vector dimensions.                     |
| `cols`         | `int32_t`       | `[0, MAX_INT32]`  | Corresponds to the vector position in the set of vectors. |

#### Attributes

| Attribute Name | TileDB Datatype | Description                                          |
| :------------- | :-------------- | :--------------------------------------------------- |
| `values`       | `dtype`         | Contains the vector value at the specific dimension. |

#### `shuffled_ids`

This is a 1D dense array that maps vector indices in the `shuffled_vectors` array to `external_ids` of each vector.

#### Basic schema parameters

| **Parameter** | **Value** |
| :------------ | :-------- |
| Array type    | Dense     |
| Rank          | 1D        |
| Cell order    | Col-major |
| Tile order    | Col-major |

#### Dimensions

| Dimension Name | TileDB Datatype | Domain           | Description                                               |
| :------------- | :-------------- | :--------------- | :-------------------------------------------------------- |
| `rows`         | `int32_t`       | `[0, MAX_INT32]` | Corresponds to the vector position in `shuffled_vectors`. |

#### Attributes

| Attribute Name | TileDB Datatype | Description                        |
| :------------- | :-------------- | :--------------------------------- |
| `values`       | `uint64_t`      | Contains the vector `external_id`. |

#### `adjacency_row_index_array_name`

This is a 1D dense array that holds the edges for each node in the compressed sparse row (CSR) format graph. Each value indicates where the neighbors (edges) for each successive node start in `adjacency_ids` and `adjacency_scores`. For example, we might have [0, 2, 8, 13] which indicates that the neighbors for node 0 start at index 0, the neighbors for node 1 start at index 2, and the neighbors for node 2 start at index 8. The final value is the end of the array, so the neighbors for node 2 end at index 13. With that information, we can look in `adjacency_ids` to determine the destination node. The source node can be inferred by the index of the Adjacency Row Indices array. Once you know the source or destination node index, you can look at that index in `shuffled_vectors` or `shuffled_ids` to get the vector or external ID for that node.

#### Basic schema parameters

| **Parameter** | **Value** |
| :------------ | :-------- |
| Array type    | Dense     |
| Rank          | 1D        |
| Cell order    | Col-major |
| Tile order    | Col-major |

#### Dimensions

| Dimension Name | TileDB Datatype | Domain           | Description                                                                  |
| :------------- | :-------------- | :--------------- | :--------------------------------------------------------------------------- |
| `rows`         | `int32_t`       | `[0, MAX_INT32]` | Corresponds to the vector position in `shuffled_vectors` and `shuffled_ids`. |

#### Attributes

| Attribute Name | TileDB Datatype | Description                                                                                 |
| :------------- | :-------------- | :------------------------------------------------------------------------------------------ |
| `values`       | `uint64_t`      | Contains the start and stop indexes in `adjacency_ids` and `adjacency_scores` for the node. |

#### `adjacency_ids`

This is a 1D dense array that holds the indexes of the destination vector for each edge in the compressed sparse row (CSR) format graph. Each value is an index into the `shuffled_vectors` and `shuffled_ids` arrays. This only holds the destination nodes of the graph, the source node is in `adjacency_row_index_array_name`, which itself points to `adjacency_ids`.

#### Basic schema parameters

| **Parameter** | **Value** |
| :------------ | :-------- |
| Array type    | Dense     |
| Rank          | 1D        |
| Cell order    | Col-major |
| Tile order    | Col-major |

#### Dimensions

| Dimension Name | TileDB Datatype | Domain           | Description                                                                  |
| :------------- | :-------------- | :--------------- | :--------------------------------------------------------------------------- |
| `rows`         | `int32_t`       | `[0, MAX_INT32]` | Corresponds to the vector position in `shuffled_vectors` and `shuffled_ids`. |

#### Attributes

| Attribute Name | TileDB Datatype | Description                                                              |
| :------------- | :-------------- | :----------------------------------------------------------------------- |
| `values`       | `uint64_t`      | Contains the index of the destination vector for this edge in the graph. |

#### `adjacency_scores`

This is a 1D dense array that holds the distance of the edge in `adjacency_ids` in the compressed sparse row (CSR) format graph. This follows the same pattern as `adjacency_ids`, but holds the edge distance instead of the destination node.

#### Basic schema parameters

| **Parameter** | **Value** |
| :------------ | :-------- |
| Array type    | Dense     |
| Rank          | 1D        |
| Cell order    | Col-major |
| Tile order    | Col-major |

#### Dimensions

| Dimension Name | TileDB Datatype | Domain           | Description                                            |
| :------------- | :-------------- | :--------------- | :----------------------------------------------------- |
| `rows`         | `int32_t`       | `[0, MAX_INT32]` | Corresponds to the vector position in `adjacency_ids`. |

#### Attributes

| Attribute Name | TileDB Datatype | Description                                           |
| :------------- | :-------------- | :---------------------------------------------------- |
| `values`       | `float`         | Contains the distance between neighbors in the graph. |
