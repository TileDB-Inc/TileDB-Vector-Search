#include <tiledb/tiledb>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "index/ivf_flat_index.h"
#include "linalg.h"

namespace py = pybind11;
using Ctx = tiledb::Context;

namespace {

template <typename T, typename shuffled_ids_type = uint64_t>
static void declare_kmeans(py::module& m, const std::string& suffix) {
  m.def(
      ("kmeans_fit_" + suffix).c_str(),
      [](size_t n_clusters,
         std::string init,
         size_t max_iter,
         bool verbose,
         size_t n_init,
         const ColMajorMatrix<T>& sample_vectors,
         std::optional<double> tol,
         std::optional<unsigned int> seed,
         std::optional<size_t> nthreads) {
        // TODO: support verbose and n_init
        std::ignore = verbose;
        std::ignore = n_init;
        kmeans_init init_val;
        if (init == "k-means++") {
          init_val = kmeans_init::kmeanspp;
        } else if (init == "random") {
          init_val = kmeans_init::random;
        } else {
          throw std::invalid_argument("Invalid init method");
        }
        ivf_flat_index<T> idx(
            /*sample_vectors.num_rows(),*/ n_clusters,
            max_iter,
            tol.value_or(0.0001));
        idx.train(sample_vectors, init_val);
        return std::move(idx.get_centroids());
      });

  m.def(
      ("kmeans_predict_" + suffix).c_str(),
      [](const ColMajorMatrix<T>& centroids,
         const ColMajorMatrix<T>& sample_vectors) {
        return ivf_flat_index<T>::predict(centroids, sample_vectors);
      });
}

}  // anonymous namespace

void init_kmeans(py::module_& m) {
  declare_kmeans<float>(m, "f32");
}
