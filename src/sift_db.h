//
// Created by Andrew Lumsdaine on 4/12/23.
//

#ifndef TDB_SIFT_DB_H
#define TDB_SIFT_DB_H


#include <fcntl.h>
#include <filesystem>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string>
#include <span>
#include <vector>
#include <unistd.h>

/*
We use three different file formats:
  • The vector files are stored in .bvecs or .fvecs format,
  • The groundtruth file in is .ivecs format.

.bvecs, .fvecs and .ivecs vector file formats:

The vectors are stored in raw little endian.
Each vector takes 4+d*4 bytes for .fvecs and .ivecs formats, and 4+d bytes for .bvecs formats,
where d is the dimensionality of the vector, as shown below.

 field 	 field type 	 description
d	int	the vector dimension
components	(unsigned char|float | int)*d	the vector components


The only difference between .bvecs, .fvecs and .ivecs files is the base type for the
vector components, which is unsigned char, float or int, respectively.

In the Input/Output section below, we provide two functions to read such files in matlab.

Details and Download

MD5 sums are available here.

Vector set	Download	descriptor	dimension	   nb base
   vectors	nb query
vectors	  nb learn
  vectors	file
format
ANN_SIFT10K	 siftsmall.tar.gz   (5.1MB) 	 SIFT (1)	   128 	       10,000	   100	     25,000	fvecs
ANN_SIFT1M	 sift.tar.gz   (161MB)	 SIFT (1)	   128	    1,000,000	10,000	    100,000	fvecs
ANN_GIST1M	 gist.tar.gz   (2.6GB)	 GIST (2) 	   960	    1,000,000	 1,000	    500,000	fvecs
ANN_SIFT1B	 Base set   (92 GB)
 Learning set
   (9.1 GB)
 Query set
   (964 KB)
 Groundtruth
   (512 MB)	 SIFT (3) 	   128	1,000,000,000	10,000	100,000,000	bvecs

(1) SIFT descriptors, Mikolajczyk implementation of Hessian-affine detector
(2) GIST descriptors, INRIA C implementation
(3) SIFT descriptors Lowe's implementation (DoG)

 file size = (4+d*4) * N bytes
*/

// siftsmall_base.fvecs		siftsmall_groundtruth.ivecs	siftsmall_learn.fvecs		siftsmall_query.fvecs


template <class T>
class sift_db : public std::vector<std::span<T>> {

  using Base = std::vector<std::span<T>>;

  std::vector<T> data_;

public:
  sift_db(const std::string& bin_file, size_t dimension) {

    if (!std::filesystem::exists(bin_file)) {
      throw std::runtime_error("file " + bin_file + " does not exist");
    }
    auto file_size = std::filesystem::file_size(bin_file);
    auto num_vectors = file_size / (4 + dimension * sizeof(T));

    auto fd = open(bin_file.c_str(), O_RDONLY);

    struct stat s;
    fstat(fd, &s);
    size_t mapped_size = s.st_size;
    assert(s.st_size == file_size);

    T *mapped_ptr = (T*)mmap(0, mapped_size, PROT_READ, MAP_FILE | MAP_PRIVATE, fd, 0);
    if ((long)mapped_ptr == -1) {
      throw std::runtime_error("mmap failed");
    }
    data_.resize(num_vectors * dimension);

    auto data_ptr = data_.data();
    auto sift_ptr = mapped_ptr;

    for (size_t k = 0; k < num_vectors; ++k) {
      auto dim = *reinterpret_cast<int*>(sift_ptr++);
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
