#include <pybind11/pybind11.h>

#include "ivf_index.h"

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(tiledbvspy, m) {
  m.def("add", &add);

  m.def("ivf_flat",
        [](const Matrix<float32_t>& data,
           unsigned nclusters,
           InitType init_type,
           unsigned nrepeats,
           unsigned max_iter,
           double tol,
           size_t seed,
           KMeansAlgorithm algorithm,
           size_t nthreads) {
          return ivf_flat(data,
                          nclusters,
                          init_type,
                          nrepeats,
                          max_iter,
                          tol,
                          seed,
                          algorithm,
                          nthreads);
        });
}