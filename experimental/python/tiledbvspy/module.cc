#include <pybind11/pybind11.h>

#include "ivf_index.h"

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

namespace {

template <typename T>
static void declareVector(py::module& mod, std::string const& suffix) {
  using TVector = Vector<T>;
  using PyTVector = py::class_<TVector, std::shared_ptr<TVector>>;

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

  declareVector<float32_t>(m, "_f32");

  m.def("get_v", []() {
    auto a = std::make_shared<Vector<float32_t>>(7);
    auto v = a->data();
    std::iota(v, v + 7, 1);

    return a;
  });
}