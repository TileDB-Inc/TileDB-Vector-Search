//
// Created by Andrew Lumsdaine on 4/12/23.
//

#ifndef TDB_SIFT_DB_H
#define TDB_SIFT_DB_H


#include <cassert>
#include <fcntl.h>
#include <filesystem>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string>
#include <span>
#include <vector>
#include <unistd.h>

/**
 * See http://corpus-texmex.irisa.fr for file format
 */

template <class T>
class sift_db : public std::vector<std::span<T>> {

  using Base = std::vector<std::span<T>>;

  std::vector<T> data_;

public:
  sift_db(const std::string& bin_file, size_t subset = 0) {

    if (!std::filesystem::exists(bin_file)) {
      throw std::runtime_error("file " + bin_file + " does not exist");
    }
    auto file_size = std::filesystem::file_size(bin_file);

    auto fd = open(bin_file.c_str(), O_RDONLY);
    if (fd == -1) {
      throw std::runtime_error("could not open " + bin_file);
    }

    uint32_t dimension{0};
    auto num_read = read(fd, &dimension, 4);
    lseek(fd, 0, SEEK_SET);

    auto max_vectors = file_size / (4 + dimension * sizeof(T));
    if (subset > max_vectors) {
      throw std::runtime_error("specified subset is too large " + std::to_string(subset) + " > " + std::to_string(max_vectors));
    }
    
    auto num_vectors = subset == 0 ? max_vectors : subset;

    struct stat s;
    fstat(fd, &s);
    size_t mapped_size = s.st_size;
    assert(s.st_size == file_size);

    T *mapped_ptr = reinterpret_cast<T*>(mmap(0, mapped_size, PROT_READ, MAP_FILE | MAP_PRIVATE, fd, 0));
    if ((long)mapped_ptr == -1) {
      throw std::runtime_error("mmap failed");
    }

    // @todo use unique_ptr for overwrite
    data_.resize(num_vectors * dimension);

    auto data_ptr = data_.data();
    auto sift_ptr = mapped_ptr;


    // Perform strided read 
    for (size_t k = 0; k < num_vectors; ++k) {

      // Check for consistency of dimensions
      decltype(dimension) dim = *reinterpret_cast<int*>(sift_ptr++);
      if (dim != dimension) {
        throw std::runtime_error("dimension mismatch: " + std::to_string(dim) + " != " + std::to_string(dimension));
      }
      std::copy(sift_ptr, sift_ptr + dimension, data_ptr);
      this->emplace_back(std::span<T>{data_ptr, dimension});
      data_ptr += dimension;
      sift_ptr += dimension;
    }

    munmap(mapped_ptr, mapped_size);
    close(fd);
  }
};



#endif//TDB_SIFT_DB_H
