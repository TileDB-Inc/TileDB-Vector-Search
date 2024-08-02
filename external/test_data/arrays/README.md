### Arrays

This directory holds TileDB Array's created from various data sources. To create the siftsmall array, run the following command:

```cpp
auto siftsmall_inputs = read_bin_local<siftsmall_feature_type>(ctx, siftsmall_inputs_file);
if (vfs.is_dir(sift_inputs_uri)) {
    vfs.remove_dir(sift_inputs_uri);
}
create_matrix(ctx, siftsmall_inputs, sift_inputs_uri, TILEDB_FILTER_ZSTD);
write_matrix(ctx, siftsmall_inputs, sift_inputs_uri, 0, false);

auto siftsmall_query = read_bin_local<siftsmall_feature_type>(ctx, siftsmall_query_file);
if (vfs.is_dir(sift_query_uri)) {
        vfs.remove_dir(sift_query_uri);
}
create_matrix(ctx, siftsmall_query, sift_query_uri, TILEDB_FILTER_ZSTD);
write_matrix(ctx, siftsmall_query, sift_query_uri, 0, false);

auto siftsmall_groundtruth = read_bin_local<siftsmall_groundtruth_type>(ctx, siftsmall_groundtruth_file);
if (vfs.is_dir(sift_groundtruth_uri)) {
  vfs.remove_dir(sift_groundtruth_uri);
}
create_matrix(ctx, siftsmall_groundtruth, sift_groundtruth_uri, TILEDB_FILTER_ZSTD);
write_matrix(ctx, siftsmall_groundtruth, sift_groundtruth_uri, 0, false);
```
