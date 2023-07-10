#include <tiledb/tiledb>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "linalg.h"
#include "ivf_index.h"
#include "ivf_query.h"

namespace py = pybind11;
using Ctx = tiledb::Context;

bool global_debug = true;
double global_time_of_interest;

bool enable_stats = false;
std::vector<json> core_stats;

PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint32_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint64_t>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<double>);
#if !defined(__GNUC__)
  PYBIND11_MAKE_OPAQUE(std::vector<size_t>);
#endif

namespace {


template <typename T>
static void declareVector(py::module& mod, std::string const& suffix) {
  using TVector = Vector<T>;
  using PyTVector = py::class_<TVector>;

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
  using TMatrix = ColMajorMatrix<T>;
  using PyTMatrix = py::class_<TMatrix>;

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

template <typename T>
static void declare_pyarray_to_matrix(py::module& m, const std::string& suffix) {
  m.def(("pyarray_copyto_matrix" + suffix).c_str(),
      [](py::array_t<T, py::array::f_style> arr) -> ColMajorMatrix<T> {
        py::buffer_info info = arr.request();
        if (info.ndim != 2)
          throw std::runtime_error("Number of dimensions must be two");
        if (info.format != py::format_descriptor<T>::format())
          throw std::runtime_error("Mismatched buffer format!");

        auto data = std::unique_ptr<T[]>{new T[info.shape[0] * info.shape[1]]};
        std::memcpy(data.get(), info.ptr, info.shape[0] * info.shape[1] * sizeof(T));
        auto r = ColMajorMatrix<T>(std::move(data), info.shape[0], info.shape[1]);
        return r;
        });
}

template <typename T, typename Id_Type = uint64_t>
static void declare_qv_query_heap_infinite_ram(py::module& m, const std::string& suffix) {
  m.def(("qv_query_heap_infinite_ram_" + suffix).c_str(),
      [](const ColMajorMatrix<T>& parts,
         const ColMajorMatrix<float>& centroids,
         const ColMajorMatrix<float>& query_vectors,
         std::vector<Id_Type> indices,
         std::vector<Id_Type> ids,
         size_t nprobe,
         size_t k_nn,
         bool nth,
         size_t nthreads) -> ColMajorMatrix<size_t> { // TODO change return type

        auto r = detail::ivf::qv_query_heap_infinite_ram(
            parts,
            centroids,
            query_vectors,
            indices,
            ids,
            nprobe,
            k_nn,
            nth,
            nthreads);
        return r;
        }, py::keep_alive<1,2>());
}

template <typename T, typename Id_Type = uint64_t>
static void declare_qv_query_heap_finite_ram(py::module& m, const std::string& suffix) {
  m.def(("qv_query_heap_finite_ram_" + suffix).c_str(),
      [](tiledb::Context& ctx,
         const std::string& parts_uri,
         const ColMajorMatrix<float>& centroids,
         const ColMajorMatrix<float>& query_vectors,
         std::vector<Id_Type> indices,
         const std::string& ids_uri,
         size_t nprobe,
         size_t k_nn,
         size_t upper_bound,
         bool nth,
         size_t nthreads) -> ColMajorMatrix<size_t> { // TODO change return type

        auto r = detail::ivf::qv_query_heap_finite_ram<T, Id_Type>(
            ctx,
            parts_uri,
            centroids,
            query_vectors,
            indices,
            ids_uri,
            nprobe,
            k_nn,
            upper_bound,
            nth,
            nthreads);
        return r;
        }, py::keep_alive<1,2>());
}

template <typename T, typename Id_Type = uint64_t>
static void declare_nuv_query_heap_infinite_ram(py::module& m, const std::string& suffix) {
  m.def(("nuv_query_heap_infinite_ram_" + suffix).c_str(),
      [](const ColMajorMatrix<T>& parts,
         const ColMajorMatrix<float>& centroids,
         const ColMajorMatrix<float>& query_vectors,
         std::vector<Id_Type> indices,
         std::vector<Id_Type> ids,
         size_t nprobe,
         size_t k_nn,
         bool nth,
         size_t nthreads) -> ColMajorMatrix<size_t> { // TODO change return type

        auto r = detail::ivf::nuv_query_heap_infinite_ram(
            parts,
            centroids,
            query_vectors,
            indices,
            ids,
            nprobe,
            k_nn,
            nth,
            nthreads);
        return r;
        }, py::keep_alive<1,2>());
}

template <typename T, typename Id_Type = uint64_t>
static void declare_nuv_query_heap_finite_ram(py::module& m, const std::string& suffix) {
  m.def(("nuv_query_heap_finite_ram_" + suffix).c_str(),
      [](tiledb::Context& ctx,
         const std::string& parts_uri,
         const ColMajorMatrix<float>& centroids,
         const ColMajorMatrix<float>& query_vectors,
         std::vector<Id_Type> indices,
         const std::string& ids_uri,
         size_t nprobe,
         size_t k_nn,
         size_t upper_bound,
         bool nth,
         size_t nthreads) -> ColMajorMatrix<size_t> { // TODO change return type

        auto r = detail::ivf::nuv_query_heap_finite_ram<T, Id_Type>(
            ctx,
            parts_uri,
            centroids,
            query_vectors,
            indices,
            ids_uri,
            nprobe,
            k_nn,
            upper_bound,
            nth,
            nthreads);
        return r;
        }, py::keep_alive<1,2>());
}

template <typename T>
static void declare_ivf_index(py::module& m, const std::string& suffix) {
  m.def(("ivf_index_" + suffix).c_str(),
      [](tiledb::Context& ctx,
        const ColMajorMatrix<T>& db,
        const std::string& centroids_uri,
        const std::string& parts_uri,
        const std::string& index_uri,
        const std::string& id_uri,
        size_t start_pos,
        size_t end_pos,
        size_t nthreads) -> int {
            return detail::ivf::ivf_index<T, uint64_t, float>(
                ctx,
                db,
                centroids_uri,
                parts_uri,
                index_uri,
                id_uri,
                start_pos,
                end_pos,
                nthreads);
        }, py::keep_alive<1,2>());
}

template <typename T>
static void declare_ivf_index_tdb(py::module& m, const std::string& suffix) {
  m.def(("ivf_index_tdb_" + suffix).c_str(),
      [](tiledb::Context& ctx,
        const std::string& db_uri,
        const std::string& centroids_uri,
        const std::string& parts_uri,
        const std::string& index_uri,
        const std::string& id_uri,
        size_t start_pos,
        size_t end_pos,
        size_t nthreads) -> int {
            return detail::ivf::ivf_index<T, uint64_t, float>(
                ctx,
                db_uri,
                centroids_uri,
                parts_uri,
                index_uri,
                id_uri,
                start_pos,
                end_pos,
                nthreads);
        }, py::keep_alive<1,2>());
}

template <class T=float, class U=size_t>
static void declareFixedMinPairHeap(py::module& mod) {
  using PyFixedMinPairHeap = py::class_<fixed_min_pair_heap<T, U>>;
  PyFixedMinPairHeap cls(mod, "FixedMinPairHeap", py::buffer_protocol());

  cls.def(py::init<unsigned>());
  cls.def("insert", &fixed_min_pair_heap<T, U>::insert);
  cls.def("__len__", [](const fixed_min_pair_heap<T, U> &v) { return v.size(); });
  cls.def("__getitem__", [](fixed_min_pair_heap<T, U>& v, size_t i) { return v[i]; });
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

  cls.def(py::init<const Ctx&, std::string, size_t>(),  py::keep_alive<1,2>());

  if constexpr (std::is_same<P, tdbColMajorMatrix<T>>::value) {
    cls.def("load", &TMatrix::load);
  }
}

template <typename P>
static void declarePartitionedMatrix(py::module& mod,
    std::string const& name,
    std::string const& suffix) {

  using T = typename P::value_type;
  using TMatrix = P;
  using PyTMatrix = py::class_<TMatrix, ColMajorMatrix<T>>;

  PyTMatrix cls(mod, (name + "_" + suffix).c_str(), py::buffer_protocol());
  cls.def(py::init<const tiledb::Context&,
                   const std::string&,      // db_uri
                   std::vector<uint64_t>&,  // partition array indices
                   std::vector<uint64_t>&,  // partition list to load
                   const std::string&>(),   // id_uri
                  py::keep_alive<1,2>());
  cls.def("load", &TMatrix::load);
}

template <typename T>
void declareStdVector(py::module& m, const std::string& suffix) {
  auto name = std::string("StdVector_") + suffix;
  py::class_<std::vector<T>>(m, name.c_str(), py::buffer_protocol())
    .def(py::init<>())
    .def(py::init([suffix](py::array_t<T> b) -> std::vector<T> {
        py::buffer_info info = b.request();
        if (info.ndim != 1)
            throw std::runtime_error("Incompatible buffer dimension!");
        std::vector<T> v(info.shape[0]);
        std::memcpy(v.data(), info.ptr, info.shape[0] * sizeof(T));
        return v;
    }))
    .def("clear", &std::vector<T>::clear)
    .def("pop_back", &std::vector<T>::pop_back)
    .def("__len__", [](const std::vector<T> &v) { return v.size(); })
    .def_buffer([](std::vector<T> &v) -> py::buffer_info {
        return py::buffer_info(
            v.data(),                               /* Pointer to buffer */
            sizeof(T),                          /* Size of one scalar */
            py::format_descriptor<T>::format(), /* Python struct-style format descriptor */
            1,                                      /* Number of dimensions */
            { v.size() },                 /* Buffer dimensions */
            { sizeof(T) });
    });
}

template <typename T, typename indices_type = size_t>
void declarePartitionIvfIndex(py::module& m, const std::string& suffix) {
  m.def(("partition_ivf_index_" + suffix).c_str(),
        [](ColMajorMatrix<T>& centroids,
           ColMajorMatrix<float>& query,
           size_t nprobe,
           size_t nthreads) {
          return detail::ivf::partition_ivf_index(centroids, query, nprobe, nthreads);
           }
        );
}

template <typename query_type, typename shuffled_ids_type = uint64_t>
static void declare_dist_qv(py::module& m, const std::string& suffix) {
  m.def(("dist_qv_" + suffix).c_str(),
      [](tiledb::Context& ctx,
        const std::string& part_uri,
        std::vector<int>& active_partitions,
        ColMajorMatrix<query_type>& query,
        std::vector<std::vector<int>>& active_queries,
        std::vector<shuffled_ids_type>& indices,
        const std::string& id_uri,
        size_t k_nn
        /* size_t nthreads TODO: optional arg w/ fallback to C++ default arg */
        ) { /* TODO return type */
            return detail::ivf::dist_qv_finite_ram_part<query_type, shuffled_ids_type>(
                ctx,
                part_uri,
                active_partitions,
                query,
                active_queries,
                indices,
                id_uri,
                k_nn);
        }, py::keep_alive<1,2>());
}

} // anonymous namespace


PYBIND11_MODULE(_tiledbvspy, m) {

  py::class_<tiledb::Context> (m, "Ctx", py::module_local())
    .def(py::init([](std::optional<py::dict> maybe_config) {
      tiledb::Config cfg;
      if (maybe_config.has_value()) {
        for (auto item : maybe_config.value()) {
            cfg.set(item.first.cast<std::string>(), item.second.cast<std::string>());
        }
      }
      return tiledb::Context(cfg);
    }
  ));

  /* === Vector === */

  // Must have matching PYBIND11_MAKE_OPAQUE declaration at top of file
  declareStdVector<float>(m, "f32");
  declareStdVector<double>(m, "f64");
  declareStdVector<uint8_t>(m, "u8");
  declareStdVector<uint32_t>(m, "u32");
  declareStdVector<uint64_t>(m, "u64");
  if constexpr (!std::is_same<uint64_t, unsigned long>::value) {
    declareStdVector<size_t>(m, "szt");
  }

  m.def("read_vector_u32", &read_vector<uint32_t>, "Read a vector from TileDB");
  m.def("read_vector_u64", &read_vector<uint64_t>, "Read a vector from TileDB");

  m.def("_create_vector_u64", []() {
    auto v = std::vector<uint64_t>(10);
    // fill vector with range 1:10 using std::iota
    std::iota(v.begin(), v.begin() + 10, 0);
    return v;
  });

  /* === Matrix === */

  // template specializations
  declareColMajorMatrix<uint8_t>(m, "_u8");
  declareColMajorMatrix<float>(m, "_f32");
  declareColMajorMatrix<double>(m, "_f64");
  declareColMajorMatrix<int32_t>(m, "_i32");
  declareColMajorMatrix<int64_t>(m, "_i64");
  declareColMajorMatrix<uint32_t>(m, "_u32");
  declareColMajorMatrix<uint64_t>(m, "_u64");
  if constexpr (!std::is_same<uint64_t, unsigned long>::value) {
    // Required for a return type, but these types are equivalent on linux :/
    declareColMajorMatrix<unsigned long>(m, "_ul");
  }

  declareColMajorMatrixSubclass<tdbColMajorMatrix<uint8_t>>(
      m, "tdbColMajorMatrix", "_u8");
  declareColMajorMatrixSubclass<tdbColMajorMatrix<uint64_t>>(
      m, "tdbColMajorMatrix", "_u64");
  declareColMajorMatrixSubclass<tdbColMajorMatrix<float>>(
      m, "tdbColMajorMatrix", "_f32");
  declareColMajorMatrixSubclass<tdbColMajorMatrix<int32_t>>(
      m, "tdbColMajorMatrix", "_i32");
  declareColMajorMatrixSubclass<tdbColMajorMatrix<int64_t>>(
      m, "tdbColMajorMatrix", "_i64");

  // Converters from pyarray to matrix
  declare_pyarray_to_matrix<uint8_t>(m, "_u8");
  declare_pyarray_to_matrix<uint64_t>(m, "_u64");
  declare_pyarray_to_matrix<float>(m, "_f32");
  declare_pyarray_to_matrix<double>(m, "_f64");

  /* Query API */

  m.def("query_vq_f32",
        [](ColMajorMatrix<float>& data,
           ColMajorMatrix<float>& query_vectors,
           int k,
           bool nth,
           size_t nthreads) -> ColMajorMatrix<size_t> {
          auto r = detail::flat::vq_query_nth(data, query_vectors, k, true, nthreads);
          return r;
        });

  m.def("query_vq_u8",
        [](tdbColMajorMatrix<uint8_t>& data,
           ColMajorMatrix<float>& query_vectors,
           int k,
           bool nth,
           size_t nthreads) -> ColMajorMatrix<size_t> {
          auto r = detail::flat::vq_query_nth(data, query_vectors, k, true, nthreads);
          return r;
        });

  m.def("validate_top_k_u64",
      [](const ColMajorMatrix<uint64_t>& top_k,
         const ColMajorMatrix<int32_t>& ground_truth) -> bool {
        return validate_top_k(top_k, ground_truth);
      });

  m.def("stats_enable", []() {
    enable_stats = true;
    tiledb::Stats::enable();
  });

  m.def("stats_disable", []() {
    enable_stats = false;
    tiledb::Stats::disable();
  });

  m.def("stats_reset", []() {
    core_stats.clear();
  });

  m.def("stats_dump", []() {
    return json{core_stats}.dump();
  });

  declare_qv_query_heap_infinite_ram<uint8_t>(m, "u8");
  declare_qv_query_heap_infinite_ram<float>(m, "f32");
  declare_qv_query_heap_finite_ram<uint8_t>(m, "u8");
  declare_qv_query_heap_finite_ram<float>(m, "f32");
  declare_nuv_query_heap_infinite_ram<uint8_t>(m, "u8");
  declare_nuv_query_heap_infinite_ram<float>(m, "f32");
  declare_nuv_query_heap_finite_ram<uint8_t>(m, "u8");
  declare_nuv_query_heap_finite_ram<float>(m, "f32");

  declare_ivf_index<uint8_t>(m, "u8");
  declare_ivf_index<float>(m, "f32");
  declare_ivf_index_tdb<uint8_t>(m, "u8");
  declare_ivf_index_tdb<float>(m, "f32");

  declarePartitionIvfIndex<uint8_t>(m, "u8");
  declarePartitionIvfIndex<float>(m, "f32");

  declarePartitionedMatrix<tdbColMajorPartitionedMatrix<uint8_t, uint64_t, uint64_t, uint64_t > >(m, "tdbPartitionedMatrix", "u8");
  declarePartitionedMatrix<tdbColMajorPartitionedMatrix<float, uint64_t, uint64_t, uint64_t> >(m, "tdbPartitionedMatrix", "f32");

  declare_dist_qv<uint8_t>(m, "u8");
  declare_dist_qv<float>(m, "f32");
  declareFixedMinPairHeap(m);
}
