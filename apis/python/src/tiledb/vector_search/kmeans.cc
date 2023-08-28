#include <tiledb/tiledb>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "linalg.h"
#include "ivf_index.h"
#include "ivf_query.h"
#include "flat_query.h"

namespace py = pybind11;
using Ctx = tiledb::Context;

namespace {

template <typename T, typename shuffled_ids_type = uint64_t>
static void declare_kmeans(py::module& m, const std::string& suffix) {
  m.def(("kmeans_fit_" + suffix).c_str(),
        [](size_t n_clusters,
           std::string init,
           size_t max_iter,
           bool verbose,
           size_t n_init,
           double tol,
           size_t n_threads,
           const ColMajorMatrix<T>& sample_vectors) {
             kmeans_index<T> idx(sample_vectors.num_rows(), n_clusters, max_iter, tol, n_threads);
             // TODO: support verbose
             std::ignore = verbose;
             if (init == "kmeans++") {
                idx.kmeans_pp(sample_vectors);
             } else if (init == "random") {
                idx.kmeans_random_init(sample_vectors);
             } else {
                throw std::invalid_argument("Invalid init method");
             }
             return std::move(idx.get_centroids());
  });

  m.def(("kmeans_predict_" + suffix).c_str(),
		[](const ColMajorMatrix<T>& centroids,
		   const ColMajorMatrix<T>& sample_vectors) {
			 return kmeans_index<T>::predict(centroids, sample_vectors);
  });
}

} // anonymous namespace


void init_kmeans(py::module_& m) {
  declare_kmeans<float>(m, "f32");
  declare_kmeans<uint8_t>(m, "u8");
}
