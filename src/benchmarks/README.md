| Abbrv     | Vector Set      | Dimension | \# Vectors | dtype   |
| --------- | --------------- | --------- | ---------- | ------- |
| siftsmall | siftsmall_base  | 128       | 10,000     | float32 |
| siftsmall | siftsmall_learn | 128       | 25,000     | float32 |
| sift      | sift_base       | 128       | 1,000,000  | float32 |
| 1M        | bigann1M_base   | 128       | 1M         | uint8   |
| 10M       | bigann10M_base  | 128       | 10M        | uint8   |
| 100M      | bigann100M_base | 128       | 100M       | uint8   |
| 1B        | bigann1B_base   | 128       | 1B         | uint8   |

| Abbrv     | Index Set | \# Vectors | Index dtype | IDs     | \# IDs    | ID dtype |
| --------- | --------- | ---------- | ----------- | ------- | --------- | -------- |
| siftsmall |
| sift      | index     | 2,001      | uint64      | ids     | 1,000,000 | uint64   |
| 1M        | index.tdb | 1,000      | uint64      | ids.tdb | 1M        | uint64   |

| Abbrv     | Query            | \#Queries | dtype   | Groundtruth | Groundtruth dtype |
| --------- | ---------------- | --------- | ------- | ----------- | ----------------- |
| siftsmall | siftsmall_query  | 100       | float32 |
| siftsmall | siftsmall_query  | 100       | float32 |
| sift      | sift_query       | 10,000    | float32 |
| 1M        | query_public_10k | 10,000    | uint8   |
| 10M       | query_public_10k | 10,000    | uint8   |
| 100M      | query_public_10k | 10,000    | uint8   |
| 1B        | query_public_10k | 10,000    | uint8   |

| Vector set  | Download                                                                       | descriptor | dimension | nb base vectors | nb query vectors | nb learn vectors | file format |
| ----------- | ------------------------------------------------------------------------------ | ---------- | --------- | --------------- | ---------------- | ---------------- | ----------- |
| ANN_SIFT10K | siftsmall.tar.gz (5.1MB)                                                       | SIFT (1)   | 128       | 10,000          | 100              | 25,000           | fvecs       |
| ANN_SIFT1M  | sift.tar.gz (161MB)                                                            | SIFT (1)   | 128       | 1,000,000       | 10,000           | 100,000          | fvecs       |
| ANN_GIST1M  | gist.tar.gz (2.6GB)                                                            | GIST (2)   | 960       | 1,000,000       | 1,000            | 500,000          | fvecs       |
| ANN_SIFT1B  | Base set (92 GB) Learning set (9.1 GB) Query set (964 KB) Groundtruth (512 MB) | SIFT (3)   | 128       | 1,000,000,000   | 10,000           | 100,000,000      | bvecs       |

.bvecs, .fvecs and .ivecs vector file formats:

The vectors are stored in raw little endian.
Each vector takes 4+d\*4 bytes for .fvecs and .ivecs formats, and 4+d bytes for .bvecs formats,
where d is the dimensionality of the vector, as shown below.

| field      | field type                       | description           |
| ---------- | -------------------------------- | --------------------- |
| d          | int                              | the vector dimension  |
| components | (unsigned char\|float \| int)\*d | the vector components |
