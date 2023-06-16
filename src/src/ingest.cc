

#include <docopt.h>
#include <string>
#include "linalg.h"

bool global_debug{false};
std::string global_region;

static constexpr const char USAGE[] =
    R"(ingest: create TileDB array from input data
  Usage:
      tdb (-h | --help)
      tdb --input URI --output URI [--type TYPE] [--subset N] [--nthreads N]
[-d | -v]

  Options:
      -h, --help            show this screen
      --input URI           database file with feature vectors
      --output URI          new database file, subset of input
      --type TYPE           type of input data [default: float32]
      --subset N            how many vectors to take from input (0 = all) [default: 0]
      --nthreads N          number of threads to use (0 = all) [default: 0]
)";

int main(int argc, char* argv[]) {
  std::vector<std::string> strings(argv + 1, argv + argc);
  auto args = docopt::docopt(USAGE, strings, true);

  auto input_uri = args["--input"].asString();
  auto output_uri = args["--output"].asString();
  size_t subset = args["--subset"].asLong();
  size_t nthreads = args["--nthreads"].asLong();
  auto type = args["--type"] ? args["--type"].asString() : "float32";

  if (type == "char" || type == "byte" || type == "uint8" || type == "int8") {
    auto in = tdbColMajorMatrix<uint8_t>(input_uri, subset);
    write_matrix<uint8_t>(in, output_uri);
  } else if (type == "int" || type == "uint32" || type == "int32") {
    auto in = tdbColMajorMatrix<uint32_t>(input_uri, subset);
    write_matrix<uint32_t>(in, output_uri);
  } else if (type == "float" || type == "float32") {
    auto in = tdbColMajorMatrix<float>(input_uri, subset);
    write_matrix<float>(in, output_uri);
  } else {
    std::cerr << "Unknown type: " << type << std::endl;
    return 1;
  }
}
