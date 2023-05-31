#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "linalg.h"
#include "ivf_index.h"
#include "ivf_query.h"

namespace py = pybind11;

bool global_debug = true;
//std::string global_region = "us-east-1";
std::string global_region = "us-west-2";

namespace {


template <typename T>
static void declareVector(py::module& mod, std::string const& suffix) {
  using TVector = Vector<T>;
  using PyTVector = py::class_<TVector>;
  //using PyTVector = py::class_<TVector, std::shared_ptr<TVector>>;

  PyTVector cls(mod, ("Vector" + suffix).c_str(), py::buffer_protocol());

  cls.def(py::init<T>());
  cls.def("size", &TVector::num_rows);
  cls.def("__getitem__", [](TVector& self, size_t i) { return self[i]; });
  cls.def("__setitem__", [](TVector& self, size_t i) { return self[i]; });
  cls.def_buffer([](TVector &m) -> py::buffer_info {
        return py::buffer_info(
            m.data(),                               /* Pointer to buffer */
            sizeof(T),                          /* Size of one scalar */
            py::format_descriptor<T>::format(), /* Python struct-style format descriptor */
            1,                                      /* Number of dimensions */
            { m.num_rows() },                 /* Buffer dimensions */
            { sizeof(T) }
        );
    });

}

template <typename T>
static void declareColMajorMatrix(py::module& mod, std::string const& suffix) {
  using value_type = T;
  using TMatrix = ColMajorMatrix<T>;
  using PyTMatrix = py::class_<TMatrix>;

  std::cout << "ColMajorMatrix" << suffix << std::endl;
  PyTMatrix cls(mod, ("ColMajorMatrix" + suffix).c_str(), py::buffer_protocol());

  cls.def(py::init<size_t, size_t>());
  cls.def("size", &TMatrix::num_rows);
  cls.def_property_readonly("dtype", [](TMatrix& self) -> py::dtype {
      return py::dtype(py::format_descriptor<T>::format());
  });
  cls.def("__getitem__", [](TMatrix& self, std::pair<size_t, size_t> v) {
    // TODO: check bounds
    return self(v.first, v.second); });
  cls.def("__setitem__", [](TMatrix& self, std::pair<size_t, size_t> v, T val) {
    // TODO: check bounds
    self(v.first, v.second) = val;
  });
  cls.def_buffer([](TMatrix &m) -> py::buffer_info {
        return py::buffer_info(
            m.data(),                               /* Pointer to buffer */
            sizeof(T),                          /* Size of one scalar */
            py::format_descriptor<T>::format(), /* Python struct-style format descriptor */
            2,                                      /* Number of dimensions */
            { m.num_rows(), m.num_cols() },                 /* Buffer dimensions */
            { sizeof(T), sizeof(T) * m.num_rows() }
        );
    });

}


// Declarations for typed subclasses of ColMajorMatrix
template <typename P>
static void declareColMajorMatrixSubclass(py::module& mod,
    std::string const& name,
    std::string const& suffix) {
  using T = typename P::value_type;
  using TMatrix = P;
  using PyTMatrix = py::class_<TMatrix, ColMajorMatrix<T>>;

  // TODO auto-namify
  PyTMatrix cls(mod, (name + suffix).c_str(), py::buffer_protocol());
  cls.def(py::init<std::string, size_t>());
}

}

PYBIND11_MODULE(tiledbvspy, m) {
  /* Vector */
  declareVector<float>(m, "_f32");

  /* === Matrix === */

  // template specializations
  //declareTdbMatrix<float>(m, "_f32");

  declareColMajorMatrix<float>(m, "_f32");
  declareColMajorMatrix<double>(m, "_f64");
  declareColMajorMatrix<int32_t>(m, "_i32");
  declareColMajorMatrix<int64_t>(m, "_i64");
  declareColMajorMatrix<size_t>(m, "_szt");

  declareColMajorMatrixSubclass<tdbColMajorMatrix<size_t>>(
      m, "tdbColMajorMatrix", "_szt");
  declareColMajorMatrixSubclass<tdbColMajorMatrix<float>>(
      m, "tdbColMajorMatrix", "_f32");
  declareColMajorMatrixSubclass<tdbColMajorMatrix<int32_t>>(
      m, "tdbColMajorMatrix", "_i32");
  declareColMajorMatrixSubclass<tdbColMajorMatrix<int64_t>>(
      m, "tdbColMajorMatrix", "_i64");



  /* Query API */

  // KMeansAlgorithm enum
  py::enum_<KMeansAlgorithm>(m, "KMeansAlgorithm")
    .value("lloyd", KMeansAlgorithm::lloyd)
    .value("elkan", KMeansAlgorithm::elkan);

  m.def("query_vq_f32",
        [](const ColMajorMatrix<float>& data,
           const ColMajorMatrix<float>& query_vectors,
           int k,
           bool nth,
           size_t nthreads) {
          auto r = vq_query_heap(data, query_vectors, k, nthreads);
          return r;
        });

  m.def("validate_top_k",
      [](const ColMajorMatrix<size_t>& top_k,
         const ColMajorMatrix<int32_t>& ground_truth) -> bool {
        return validate_top_k(top_k, ground_truth);
      });

  m.def("kmeans_query",
      [](const std::string& part_uri,
         const ColMajorMatrix<uint8_t>& centroids,
         const ColMajorMatrix<uint8_t>& query_vectors,
         const std::vector<uint64_t>& indices,
         const std::string& id_uri,
         size_t nprobe,
         size_t k_nn,
         bool nth,
         size_t nthreads) {
        auto r = kmeans_query(
            part_uri,
            centroids,
            query_vectors,
            indices,
            id_uri,
            nprobe,
            k_nn,
            nth,
            nthreads);
         });

}