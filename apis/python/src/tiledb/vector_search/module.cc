#include <pybind11/pybind11.h>

#include "linalg.h"

namespace py = pybind11;

bool global_debug = false;
std::string global_region = "us-east=1";

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
static void declareMatrix(py::module& mod, std::string const& suffix) {
  using TMatrix = Matrix<T>;
  using PyTMatrix = py::class_<TMatrix, std::shared_ptr<TMatrix>>;

  PyTMatrix cls(mod, ("Matrix" + suffix).c_str(), py::buffer_protocol());

  cls.def(py::init<size_t, size_t, Kokkos::layout_right>());
  cls.def("size", &TMatrix::num_rows);
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
            { sizeof(T) * m.num_cols(), sizeof(T) }
        );
    });

}

template <typename T>
static void declareTdbMatrix(py::module& mod, std::string const& suffix) {
  using TMatrix = tdbMatrix<T>;
  using PyTMatrix = py::class_<TMatrix, std::shared_ptr<TMatrix>>;

  PyTMatrix cls(mod, ("tdbMatrix" + suffix).c_str(), py::buffer_protocol());

  cls.def(py::init<std::string>());
  cls.def("size", &TMatrix::num_rows);
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
            { sizeof(T) * m.num_cols(), sizeof(T) }
        );
    });

}

}

PYBIND11_MODULE(tiledbvspy, m) {
/*
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
*/

  declareVector<float>(m, "_f32");

  // Test helpers
  m.def("get_v", []() {
    auto a = std::make_shared<Vector<float>>(7);
    auto v = a->data();
    std::iota(v, v + 7, 1);

    return a;
  });

/* === Matrix === */

  declareMatrix<float>(m, "_f32");
  declareTdbMatrix<float>(m, "_f32");

  // Test helpers
  m.def("get_m", []() {
    auto a = std::make_shared<Matrix<float>>(3,3);
    auto v = a->data();
    std::iota(v, v + 9, 1);
    py::print(a->num_rows(), a->num_cols());

    return a;
  });

  m.def("get_tdb", []() -> std::shared_ptr<tdbMatrix<float>> {
    auto a = std::make_shared<tdbMatrix<float, Kokkos::layout_right>>("test1");
    return a;
    //return std::make_shared<Matrix<float>>(reinterpret_cast<Matrix<float>*>(a.get()));
  });
}