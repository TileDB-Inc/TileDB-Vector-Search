//
// Created by Andrew Lumsdaine on 4/17/23.
//

#include <algorithm>
#include <cmath>
// #include <execution>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <docopt.h>
#include <tiledb/tiledb>

#include "defs.h"
#include "query.h"
#include "sift_db.h"
#include "timer.h"

bool verbose = false;
bool debug = false;

static constexpr const char USAGE[] =
        R"(ingest: ingest feature vectors into TileDB.
  Usage:
      tdb (-h | --help)
      tdb --in DBFILE  --out URI [-f | -i | -b] [-d | -v]

  Options:
      -h, --help            show this screen
      --in DBFILE           database file with feature vectors
      --out URI             database URI with feature vectors
      -f, --float           use float32 (the file is a .fvecs file) [default]
      -i, --int             use int32 (the file is a .ivecs file)
      -b, --byte            use uint8 (the file is a .bvecs file)
      -d, --debug           run in debug mode [default: false]
      -v, --verbose         run in verbose mode [default: false]
)";

#if 0
#include <tiledb/tiledb>

int main(int argc, char *argv[]) {
  std::vector<std::string> strings(argv + 1, argv + argc);
  auto args = docopt::docopt(USAGE, strings, true);

  if (args["--help"].asBool()) {
    std::cout << USAGE << std::endl;
    return 0;
  }

  debug = args["--debug"].asBool();
  verbose = args["--verbose"].asBool();

  if (!args["--in"]) {
    std::cerr << "Must specify an input file" << std::endl;
    return 1;
  }
  if (!args["--out"]) {
    std::cerr << "Must specify an output file" << std::endl;
    return 1;
  }

  std::string dbfile_name = args["--in"].asString();
  std::string uri_name = args["--out"].asString();
  // Define the array schema
  tiledb::Context ctx;
  tiledb::ArraySchema schema(ctx, TILEDB_DENSE);
  schema.set_domain(tiledb::Domain(
          ctx, {tiledb::Dimension::create<int>(ctx, "x", {{0, 99}}, 10),
                tiledb::Dimension::create<int>(ctx, "y", {{0, 99}}, 10)}));
  schema.add_attribute(tiledb::Attribute::create<double>(ctx, "a"));

  // Create the TileDB array
  tiledb::Array::create("my_array", schema);

  // Define a function that reads data from disk in chunks
  auto data_reader = [](size_t offset, void* buffer, size_t length) {
    std::ifstream ifs("data.bin", std::ios::binary);
    ifs.seekg(offset);
    ifs.read(static_cast<char*>(buffer), length);
    ifs.close();
  };

  // Write the data to the array in chunks
  tiledb::Array array(ctx, "my_array", TILEDB_WRITE);
  uint64_t chunk_size = 100 * sizeof(double);
  uint64_t data_size = 10000 * sizeof(double);
  for (uint64_t offset = 0; offset < data_size; offset += chunk_size) {
    std::vector<double> data(100);
    tiledb::Query query(ctx, array);
    query.set_layout(TILEDB_ROW_MAJOR)
            .set_data_buffer("a", data)
            .set_subarray({tiledb::Range(0, 9), tiledb::Range(0, 9)});
    query.set_read_user_data(&data_reader)
            .set_read_user_data_offset(offset)
            .set_read_bytes(chunk_size);
    query.submit();
  }
  array.close();

  return 0;
}
#endif
#if 0
int main(int argc, char *argv[]) {
  std::vector<std::string> strings(argv + 1, argv + argc);
  auto args = docopt::docopt(USAGE, strings, true);

  if (args["--help"].asBool()) {
    std::cout << USAGE << std::endl;
    return 0;
  }

  debug = args["--debug"].asBool();
  verbose = args["--verbose"].asBool();

  if (!args["--in"]) {
    std::cerr << "Must specify an input file" << std::endl;
    return 1;
  }
  if (!args["--out"]) {
    std::cerr << "Must specify an output file" << std::endl;
    return 1;
  }

  std::string dbfile_name = args["--in"].asString();
  std::string uri_name = args["--out"].asString();

  size_t chunk_size{1024*1024};
  long long row_count{50'000};
  long long col_count{128};

  // Create TileDB context
  tiledb::Context ctx;

  // Define the array schema
  tiledb::ArraySchema schema(ctx, TILEDB_DENSE);
  tiledb::Domain domain(ctx);
  domain.add_dimension(tiledb::Dimension::create<int64_t>(ctx, "rows", {{0, row_count-1}}, chunk_size))
          .add_dimension(tiledb::Dimension::create<int64_t>(ctx, "cols", {{0, col_count-1}}, chunk_size));
  schema.set_domain(domain);
  schema.add_attribute(tiledb::Attribute::create<double>(ctx, "values"));
  schema.set_cell_order(TILEDB_ROW_MAJOR);
  schema.set_tile_order(TILEDB_ROW_MAJOR);

  // Create the TileDB array
  std::string array_name = "my_array";
  tiledb::Array::create(array_name, schema);

  // Open the array for writing
  tiledb::Array array(ctx, array_name, TILEDB_WRITE);

  // Open the file for reading
  std::string filename = "my_file.txt";
  std::ifstream infile(filename, std::ios::binary);

  // Define a buffer for the data to be written
  std::vector<double> values(chunk_size*chunk_size);

  // Iterate over the file data in chunks and write each chunk to the array
  for (int64_t row_start = 0; row_start < row_count; row_start += chunk_size) {
    int64_t row_end = std::min<int64_t>(row_start + chunk_size, row_count);
    for (int64_t col_start = 0; col_start < col_count; col_start += chunk_size) {
      int64_t col_end = std::min<int64_t>(col_start + chunk_size, col_count);

      // Read the data from the file into the buffer
      for (int64_t row = row_start; row < row_end; ++row) {
        std::vector<double> data_chunk(chunk_size);
        infile.read(reinterpret_cast<char*>(data_chunk.data()), chunk_size*sizeof(double));
        if (!infile.good() && !infile.eof()) {
          std::cerr << "Error reading data from file" << std::endl;
          return -1;
        }
        int64_t col_range_start = col_start - row_start;
        int64_t col_range_end = std::min<int64_t>(col_end - row_start, chunk_size);
        std::copy_n(data_chunk.begin(), col_range_end - col_range_start, values.begin() + row*chunk_size + col_range_start);
      }

      // Write the buffer to the array
      std::vector<int64_t> subarray = {row_start, row_end-1, col_start, col_end-1};
      array.write(subarray, values);

      // If we've reached the end of the file, break out of the loop
      if (infile.eof()) {
        break;
      }
    }
    if (infile.eof()) {
      break;
    }
  }

  // Close the array and file
  array.close();
  infile.close();

  return 0;
}



int main(int argc, char *argv[]) {
  std::vector<std::string> strings(argv + 1, argv + argc);
  auto args = docopt::docopt(USAGE, strings, true);

  if (args["--help"].asBool()) {
    std::cout << USAGE << std::endl;
    return 0;
  }

  debug = args["--debug"].asBool();
  verbose = args["--verbose"].asBool();

  if (!args["--in"]) {
    std::cerr << "Must specify an input file" << std::endl;
    return 1;
  }
  if (!args["--out"]) {
    std::cerr << "Must specify an output file" << std::endl;
    return 1;
  }

  std::string dbfile_name = args["--in"].asString();
  std::string uri_name = args["--out"].asString();

  size_t chunk_size = 1024*1024;
  long long file_size = 1024*1024*8;

    // Define the TileDB context and group
    tiledb::Context ctx;
    // tiledb::Group group(ctx, "my_group");

    // Define the TileDB array schema
    tiledb::Domain domain(ctx);
    domain.add_dimension(tiledb::Dimension::create<int64_t>(ctx, "dim1", {{0, file_size - 1}}, chunk_size));
    tiledb::ArraySchema schema(ctx, TILEDB_SPARSE);
    schema.set_domain(domain);
    schema.add_attribute(tiledb::Attribute::create<double>(ctx, "attr1"));
    schema.set_cell_order(TILEDB_ROW_MAJOR);
    schema.set_tile_order(TILEDB_ROW_MAJOR);

    // Create the TileDB array
    tiledb::Array::create("my_array", schema);

    // Open the array for writing
    tiledb::Array array(ctx, "my_array", TILEDB_WRITE);

    // Open the file for reading
    std::ifstream file("my_file.txt");
    if (!file.is_open()) {
      std::cerr << "Failed to open file!" << std::endl;
      return 1;
    }

    // Define a buffer for the data to be written
    std::vector<int64_t> dim1(chunk_size);
    std::vector<double> attr1(chunk_size);
    tiledb::Query query(ctx, array, TILEDB_WRITE);
    query.set_layout(TILEDB_UNORDERED);
    query.set_buffer("dim1", dim1);
    query.set_buffer("attr1", attr1);

    // Read the file data in chunks and write each chunk to the array
    while (!file.eof()) {
      file.read(reinterpret_cast<char *>(dim1.data()), chunk_size * sizeof(int64_t));
      const size_t bytes_read = file.gcount() * sizeof(int64_t);
      const size_t num_elements = bytes_read / sizeof(int64_t);

      for (size_t i = 0; i < num_elements; ++i) {
        attr1[i] = parse_data(dim1[i]);// Parse the data from the file format to the TileDB format
      }

      query.set_subarray(dim1);
      query.set_data_buffer("attr1", attr1);
      query.submit();
    }

    // Finalize the query and close the array and file
    query.finalize();
    file.close();
    array.close();
    return 0;

}

#endif