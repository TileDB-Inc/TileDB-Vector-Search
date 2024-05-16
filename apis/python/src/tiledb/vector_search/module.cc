#include <tiledb/tiledb>

#include <cstdio>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// @todo Replace
#include "detail/flat/qv.h"
#include "detail/flat/vq.h"
#include "detail/ivf/dist_qv.h"
#include "detail/ivf/index.h"
#include "detail/ivf/qv.h"
#include "detail/linalg/compat.h"
#include "detail/linalg/matrix.h"
#include "detail/linalg/partitioned_matrix.h"
#include "detail/linalg/tdb_matrix.h"
#include "detail/linalg/tdb_partitioned_matrix.h"
#include "detail/time/temporal_policy.h"

namespace py = pybind11;
using Ctx = tiledb::Context;

bool enable_stats = false;
std::vector<json> core_stats;

PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>);
PYBIND11_MAKE_OPAQUE(std::vector<int8_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint32_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint64_t>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::list<std::vector<uint8_t>>);
PYBIND11_MAKE_OPAQUE(std::list<std::vector<int8_t>>);
PYBIND11_MAKE_OPAQUE(std::list<std::vector<uint32_t>>);
PYBIND11_MAKE_OPAQUE(std::list<std::vector<uint64_t>>);
PYBIND11_MAKE_OPAQUE(std::list<std::vector<float>>);
PYBIND11_MAKE_OPAQUE(std::list<std::vector<double>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::list<uint8_t>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::list<int8_t>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::list<uint32_t>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::list<uint64_t>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::list<float>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::list<double>>);
#if !(defined(__GNUC__) || defined(_MSC_VER))
PYBIND11_MAKE_OPAQUE(std::vector<size_t>);
#endif

void init_type_erased_module(py::module&);

namespace {

template <class T>
static void declareVector(py::module& mod, std::string const& suffix) {
  using TVector = Vector<T>;
  using PyTVector = py::class_<TVector>;

  PyTVector cls(mod, ("Vector" + suffix).c_str(), py::buffer_protocol());

  cls.def(py::init<T>());
  cls.def("size", &TVector::num_rows);
  cls.def("__getitem__", [](TVector& self, size_t i) { return self[i]; });
  cls.def("__setitem__", [](TVector& self, size_t i) { return self[i]; });
  cls.def_buffer([](TVector& m) -> py::buffer_info {
    return py::buffer_info(
        m.data(),                           /* Pointer to buffer */
        sizeof(T),                          /* Size of one scalar */
        py::format_descriptor<T>::format(), /* Python struct-style format
                                               descriptor */
        1,                                  /* Number of dimensions */
        {m.num_rows()},                     /* Buffer dimensions */
        {sizeof(T)});
  });
}

template <class T>
static void declareColMajorMatrix(py::module& mod, std::string const& suffix) {
  using TMatrix = ColMajorMatrix<T>;
  using PyTMatrix = py::class_<TMatrix>;

  PyTMatrix cls(
      mod, ("ColMajorMatrix" + suffix).c_str(), py::buffer_protocol());

  cls.def(py::init<size_t, size_t>());
  cls.def("size", &TMatrix::num_rows);
  cls.def("num_rows", &TMatrix::num_rows);
  cls.def("num_cols", &TMatrix::num_cols);

  cls.def_property_readonly("dtype", [](TMatrix& self) -> py::dtype {
    return py::dtype(py::format_descriptor<T>::format());
  });
  cls.def("__getitem__", [](TMatrix& self, std::pair<size_t, size_t> v) {
    // TODO: check bounds
    return self(v.first, v.second);
  });
  cls.def("__setitem__", [](TMatrix& self, std::pair<size_t, size_t> v, T val) {
    // TODO: check bounds
    self(v.first, v.second) = val;
  });
  cls.def_buffer([](TMatrix& m) -> py::buffer_info {
    return py::buffer_info(
        m.data(),                           /* Pointer to buffer */
        sizeof(T),                          /* Size of one scalar */
        py::format_descriptor<T>::format(), /* Python struct-style format
                                               descriptor */
        2,                                  /* Number of dimensions */
        {m.num_rows(), m.num_cols()},       /* Buffer dimensions */
        {sizeof(T), sizeof(T) * m.num_rows()});
  });
}

template <class T>
static void declare_debug_matrix(py::module& m, const std::string& suffix) {
  m.def(
      ("debug_matrix" + suffix).c_str(),
      [](ColMajorMatrix<T>& mat, const std::string& msg = "module.cc: ") {
        debug_matrix(mat, msg);
      });
  // py::keep_alive<1, 2>());
}

template <class T>
static void declare_pyarray_to_matrix(
    py::module& m, const std::string& suffix) {
  m.def(
      ("pyarray_copyto_matrix" + suffix).c_str(),
      [](py::array_t<T, py::array::f_style> arr) -> ColMajorMatrix<T> {
        py::buffer_info info = arr.request();
        if (info.ndim != 2)
          throw std::runtime_error("Number of dimensions must be two");
        if (info.format != py::format_descriptor<T>::format())
          throw std::runtime_error("Mismatched buffer format!");

        auto data = std::unique_ptr<T[]>{new T[info.shape[0] * info.shape[1]]};
        std::memcpy(
            data.get(), info.ptr, info.shape[0] * info.shape[1] * sizeof(T));
        auto r =
            ColMajorMatrix<T>(std::move(data), info.shape[0], info.shape[1]);
        return r;
      });
}

namespace {
template <class... TArgs>
py::tuple make_python_pair(std::tuple<TArgs...>&& arg) {
  static_assert(sizeof...(TArgs) == 2, "Must have exactly two arguments");

  return py::make_tuple<py::return_value_policy::automatic>(
      py::cast(std::get<0>(arg), py::return_value_policy::move),
      py::cast(std::get<1>(arg), py::return_value_policy::move));
}

}  // namespace

template <class T, class Id_Type = uint64_t>
static void declare_qv_query_heap_infinite_ram(
    py::module& m, const std::string& suffix) {
  m.def(
      ("qv_query_heap_infinite_ram_" + suffix).c_str(),
      [](ColMajorMatrix<T>& parts,          // Note, will be moved!
         ColMajorMatrix<float>& centroids,  // Note, will be moved!
         const ColMajorMatrix<float>& query_vectors,
         std::vector<Id_Type>& indices,
         std::vector<Id_Type>& ids,
         size_t nprobe,
         size_t k_nn,
         size_t nthreads) -> py::tuple {
        auto mat = ColMajorPartitionedMatrixWrapper<T, Id_Type, Id_Type>(
            parts, ids, indices);

        auto top_centroids = detail::ivf::ivf_top_centroids(
            centroids, query_vectors, nprobe, nthreads);
        auto r = detail::ivf::qv_query_heap_infinite_ram(
            top_centroids, mat, query_vectors, nprobe, k_nn, nthreads);

        return make_python_pair(std::move(r));
      },
      py::keep_alive<1, 2>());
}

// This hasn't been converted to new index scheme yet
template <class T, class Id_Type = uint64_t>
static void declare_qv_query_heap_finite_ram(
    py::module& m, const std::string& suffix) {
  m.def(
      ("qv_query_heap_finite_ram_" + suffix).c_str(),
      [](tiledb::Context& ctx,
         const std::string& parts_uri,
         const ColMajorMatrix<float>& centroids,
         const ColMajorMatrix<float>& query_vectors,
         const std::vector<Id_Type>& indices,
         const std::string& ids_uri,
         size_t nprobe,
         size_t k_nn,
         size_t upper_bound,
         size_t nthreads,
         uint64_t timestamp)
          -> py::tuple {  // std::tuple<ColMajorMatrix<float>,
                          // ColMajorMatrix<size_t>> { //
                          // TODO change return type
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
            nthreads,
            timestamp);

        return make_python_pair(std::move(r));
      },
      py::keep_alive<1, 2>());
}

template <class T, class Id_Type = uint64_t>
static void declare_nuv_query_heap_infinite_ram(
    py::module& m, const std::string& suffix) {
  m.def(
      ("nuv_query_heap_infinite_ram_reg_blocked_" + suffix).c_str(),
      [](ColMajorMatrix<T>& parts,
         ColMajorMatrix<float>& centroids,
         const ColMajorMatrix<float>& query_vectors,
         std::vector<Id_Type>& indices,
         std::vector<Id_Type>& ids,
         size_t nprobe,
         size_t k_nn,
         size_t nthreads)
          -> std::tuple<
              ColMajorMatrix<float>,
              ColMajorMatrix<uint64_t>> {  // TODO change return type
        auto mat = ColMajorPartitionedMatrixWrapper<T, Id_Type, Id_Type>(
            parts, ids, indices);

        auto&& [active_partitions, active_queries] =
            detail::ivf::partition_ivf_flat_index<Id_Type>(
                centroids, query_vectors, nprobe, nthreads);

        auto r = detail::ivf::nuv_query_heap_infinite_ram(
            mat,
            active_partitions,
            query_vectors,
            active_queries,
            k_nn,
            nthreads);
        return r;
      },
      py::keep_alive<1, 2>());
}

template <class T, class Id_Type = uint64_t>
static void declare_nuv_query_heap_finite_ram(
    py::module& m, const std::string& suffix) {
  m.def(
      ("nuv_query_heap_finite_ram_reg_blocked_" + suffix).c_str(),
      [](tiledb::Context& ctx,
         const std::string& parts_uri,
         const ColMajorMatrix<float>& centroids,
         const ColMajorMatrix<float>& query_vectors,
         std::vector<Id_Type>& indices,
         const std::string& ids_uri,
         size_t nprobe,
         size_t k_nn,
         size_t upper_bound,
         size_t nthreads,
         uint64_t timestamp)
          -> std::tuple<
              ColMajorMatrix<float>,
              ColMajorMatrix<uint64_t>> {  // TODO change return type
        auto&& [active_partitions, active_queries] =
            detail::ivf::partition_ivf_flat_index<Id_Type>(
                centroids, query_vectors, nprobe, nthreads);

        auto temporal_policy{
            (timestamp == 0) ? TemporalPolicy() :
                               TemporalPolicy(TimeTravel, timestamp)};

        auto mat = tdbColMajorPartitionedMatrix<T, Id_Type, Id_Type>(
            ctx,
            parts_uri,
            indices,
            ids_uri,
            active_partitions,
            upper_bound,
            temporal_policy);

        auto r = detail::ivf::nuv_query_heap_finite_ram_reg_blocked(
            mat, query_vectors, active_queries, k_nn, upper_bound, nthreads);

        return r;
      },
      py::keep_alive<1, 2>());
}

/** Calls the principal ivf_index in index.h -- does not create a C++
 * `ivf_index` object */
template <class T>
static void declare_ivf_index(py::module& m, const std::string& suffix) {
  m.def(
      ("ivf_index_" + suffix).c_str(),
      [](tiledb::Context& ctx,
         const ColMajorMatrix<T>& db,
         const std::vector<uint64_t>& external_ids,
         const std::vector<uint64_t>& deleted_ids,
         const std::string& centroids_uri,
         const std::string& parts_uri,
         const std::string& index_array_uri,
         const std::string& id_uri,
         size_t start_pos,
         size_t end_pos,
         size_t nthreads,
         uint64_t timestamp) -> int {
        return detail::ivf::ivf_index<T, uint64_t, float>(
            ctx,
            db,
            external_ids,
            deleted_ids,
            centroids_uri,
            parts_uri,
            index_array_uri,
            id_uri,
            start_pos,
            end_pos,
            nthreads,
            timestamp);
      },
      py::keep_alive<1, 2>());
}

/** Calls the second ivf_index function in index.h -- does not create a C++
 * `ivf_index` object */
template <class T>
static void declare_ivf_index_tdb(py::module& m, const std::string& suffix) {
  m.def(
      ("ivf_index_tdb_" + suffix).c_str(),
      [](tiledb::Context& ctx,
         const std::string& db_uri,
         const std::string& external_ids_uri,
         const std::vector<uint64_t>& deleted_ids,
         const std::string& centroids_uri,
         const std::string& parts_uri,
         const std::string& index_array_uri,
         const std::string& id_uri,
         size_t start_pos,
         size_t end_pos,
         size_t nthreads,
         uint64_t timestamp) -> int {
        return detail::ivf::ivf_index<T, uint64_t, float>(
            ctx,
            db_uri,
            external_ids_uri,
            deleted_ids,
            centroids_uri,
            parts_uri,
            index_array_uri,
            id_uri,
            start_pos,
            end_pos,
            nthreads,
            timestamp);
      },
      py::keep_alive<1, 2>());
}

template <class T = float, class U = uint64_t>
static void declareFixedMinPairHeap(py::module& mod) {
  using PyFixedMinPairHeap = py::class_<fixed_min_pair_heap<T, U>>;
  PyFixedMinPairHeap cls(mod, "FixedMinPairHeap", py::buffer_protocol());

  cls.def(py::init<unsigned>());
  cls.def(
      "insert",
      [](fixed_min_pair_heap<T, U>& heap, const T& x, const U& y) {
        return heap.insert(x, y);
      }),
      cls.def("__len__", [](const fixed_min_pair_heap<T, U>& v) {
        return v.size();
      });
  cls.def("__getitem__", [](fixed_min_pair_heap<T, U>& v, size_t i) {
    return v[i];
  });
}

// Declarations for typed subclasses of ColMajorMatrix
template <class P>
static void declareColMajorMatrixSubclass(
    py::module& mod, std::string const& name, std::string const& suffix) {
  using T = typename P::value_type;
  using TMatrix = P;
  using PyTMatrix = py::class_<TMatrix, ColMajorMatrix<T>>;

  // TODO auto-namify
  PyTMatrix cls(mod, (name + suffix).c_str(), py::buffer_protocol());

  cls.def(
      py::init<
          const Ctx&,             // ctx
          std::string,            // uri
          size_t,                 // first_row
          std::optional<size_t>,  // last_row
          size_t,                 // first_col
          std::optional<size_t>,  // last_col
          size_t,                 // upper_bound
          uint64_t>(),            // timestamp
      py::keep_alive<1, 2>());

  if constexpr (std::is_same<P, tdbColMajorMatrix<T>>::value) {
    cls.def("load", &TMatrix::load);
  }
}

template <class T, class Id_Type, class Indices_Type, class I>
static void declarePartitionedMatrix(
    py::module& mod, std::string const& name, std::string const& suffix) {
  using TMatrix = tdbColMajorPartitionedMatrix<T, Id_Type, Indices_Type, I>;
  using PyTMatrix = py::class_<TMatrix>;

  PyTMatrix cls(mod, (name + "_" + suffix).c_str(), py::buffer_protocol());

  cls.def(
      py::init<
          const tiledb::Context&,
          const std::string&,  // sift_inputs_uri
          const std::string&,
          const std::string&,                // id_uri
          const std::vector<Indices_Type>&,  // partition list to load
          size_t>(),                         // upper_bound

      py::keep_alive<1, 2>());
  cls.def("load", &TMatrix::load);
}

template <class T>
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
      .def("__len__", [](const std::vector<T>& v) { return v.size(); })
      .def(
          "__getitem__", [](const std::vector<T>& v, size_t i) { return v[i]; })
      .def_buffer([](std::vector<T>& v) -> py::buffer_info {
        return py::buffer_info(
            v.data(),                           /* Pointer to buffer */
            sizeof(T),                          /* Size of one scalar */
            py::format_descriptor<T>::format(), /* Python struct-style format
                                                   descriptor */
            1,                                  /* Number of dimensions */
            {v.size()},                         /* Buffer dimensions */
            {sizeof(T)});
      });
}

template <class T, class indices_type = uint64_t>
void declarePartitionIvfIndex(py::module& m, const std::string& suffix) {
  m.def(
      ("partition_ivf_index_" + suffix).c_str(),
      [](ColMajorMatrix<float>& centroids,
         ColMajorMatrix<T>& query,
         size_t nprobe,
         size_t nthreads) {
        return detail::ivf::partition_ivf_flat_index<indices_type>(
            centroids, query, nprobe, nthreads);
      });
}

template <
    class T,
    class shuffled_ids_type = uint64_t,
    class indices_type = uint64_t>
static void declare_dist_qv(py::module& m, const std::string& suffix) {
  m.def(
      ("dist_qv_" + suffix).c_str(),
      [](tiledb::Context& ctx,                           // 0
         const std::string& part_uri,                    // 1
         std::vector<indices_type>& active_partitions,   // 2
         ColMajorMatrix<float>& query,                   // 3
         std::vector<std::vector<int>>& active_queries,  // 4
         std::vector<indices_type>& indices,             // 5
         const std::string& id_uri,
         size_t k_nn,
         uint64_t timestamp
         /* size_t nthreads TODO: optional arg w/ fallback to C++ default arg */
      ) { /* TODO return type */
          size_t upper_bound{0};
          auto nthreads = std::thread::hardware_concurrency();

          return detail::ivf::dist_qv_finite_ram_part<T, shuffled_ids_type>(
              ctx,
              part_uri,
              active_partitions,
              query,
              active_queries,
              indices,
              id_uri,
              k_nn,
              timestamp);
      },
      py::keep_alive<1, 2>());
  m.def(
      ("dist_qv_" + suffix).c_str(),
      [](tiledb::Context& ctx,
         const std::string& part_uri,
         std::vector<indices_type>& active_partitions,
         ColMajorMatrix<T>& query,
         py::array& active_queries_arr,  // Alternative to std::vector argument
                                         // in above API
         std::vector<shuffled_ids_type>& indices,
         const std::string& id_uri,
         size_t k_nn,
         uint64_t timestamp
         /* size_t nthreads @todo: optional arg w/ fallback to C++ default arg
          */
      ) { /* @todo: return type */
          size_t upper_bound{0};
          auto nthreads = std::thread::hardware_concurrency();
          auto temporal_policy{
              (timestamp == 0) ? TemporalPolicy() :
                                 TemporalPolicy(TimeTravel, timestamp)};

          py::buffer_info buf_info = active_queries_arr.request();
          auto shape = active_queries_arr.shape();
          size_t num_rows = shape[0];

          auto active_queries = std::vector<std::vector<indices_type>>();
          active_queries.reserve(num_rows);

          auto ptr = static_cast<py::object*>(buf_info.ptr);

          for (size_t i = 0; i < num_rows; ++i) {
            py::list sublist = py::cast<py::list>(ptr[i]);
            size_t sublist_length = py::len(sublist);
            active_queries.emplace_back();
            active_queries.back().reserve(sublist_length);
            for (size_t j = 0; j < sublist_length; ++j) {
              active_queries.back().emplace_back(
                  py::cast<indices_type>(sublist[j]));
            }
          }

          return detail::ivf::dist_qv_finite_ram_part<T, shuffled_ids_type>(
              ctx,
              part_uri,
              active_partitions,
              query,
              active_queries,
              indices,
              id_uri,
              k_nn,
              timestamp);
      },
      py::keep_alive<1, 2>());
}

template <class T, class shuffled_ids_type = uint64_t>
static void declare_vq_query_heap(py::module& m, const std::string& suffix) {
  m.def(
      ("vq_query_heap_" + suffix).c_str(),
      [](tdbColMajorMatrix<T>& data,
         ColMajorMatrix<float>& query_vectors,
         const std::vector<uint64_t>& ids,
         int k,
         size_t nthreads)
          -> std::tuple<ColMajorMatrix<float>, ColMajorMatrix<uint64_t>> {
        auto r =
            detail::flat::vq_query_heap(data, query_vectors, ids, k, nthreads);
        return r;
      });
}

template <class T, class shuffled_ids_type = uint64_t>
static void declare_vq_query_heap_pyarray(
    py::module& m, const std::string& suffix) {
  m.def(
      ("vq_query_heap_pyarray_" + suffix).c_str(),
      [](ColMajorMatrix<T>& data,
         ColMajorMatrix<float>& query_vectors,
         const std::vector<uint64_t>& ids,
         int k,
         size_t nthreads)
          -> std::tuple<ColMajorMatrix<float>, ColMajorMatrix<uint64_t>> {
        auto r =
            detail::flat::vq_query_heap(data, query_vectors, ids, k, nthreads);
        return r;
      });
}

}  // anonymous namespace

void init_kmeans(py::module&);
void init_type_erased_module(py::module&);

/**************************************************************************
 *
 * Template instantiations to create typed interface functions
 *
 **************************************************************************/

PYBIND11_MODULE(_tiledbvspy, m) {
  py::class_<tiledb::Context>(m, "Ctx", py::module_local())
      .def(py::init([](std::optional<py::dict> maybe_config) {
        tiledb::Config cfg;
        if (maybe_config.has_value()) {
          for (auto item : maybe_config.value()) {
            cfg.set(
                item.first.cast<std::string>(),
                item.second.cast<std::string>());
          }
        }
        return tiledb::Context(cfg);
      }));

  /* === Vector === */

  // Must have matching PYBIND11_MAKE_OPAQUE declaration at top of file
  declareStdVector<float>(m, "f32");
  declareStdVector<double>(m, "f64");
  declareStdVector<uint8_t>(m, "u8");
  declareStdVector<int8_t>(m, "i8");
  declareStdVector<uint32_t>(m, "u32");
  declareStdVector<uint64_t>(m, "u64");
  if constexpr (!std::is_same_v<uint64_t, size_t>) {
    declareStdVector<size_t>(m, "szt");
  }

  m.def(
      "read_vector_u32",
      [](const tiledb::Context& ctx,
         const std::string& uri,
         size_t start_pos,
         size_t end_pos,
         uint64_t timestamp) -> std::vector<uint32_t> {
        TemporalPolicy temporal_policy =
            (timestamp == 0) ? TemporalPolicy() :
                               TemporalPolicy(TimeTravel, timestamp);
        auto r = read_vector<uint32_t>(
            ctx, uri, start_pos, end_pos, temporal_policy);
        return r;
      });
  m.def(
      "read_vector_u64",
      [](const tiledb::Context& ctx,
         const std::string& uri,
         size_t start_pos,
         size_t end_pos,
         uint64_t timestamp) -> std::vector<uint64_t> {
        TemporalPolicy temporal_policy =
            (timestamp == 0) ? TemporalPolicy() :
                               TemporalPolicy(TimeTravel, timestamp);
        auto r = read_vector<uint64_t>(
            ctx, uri, start_pos, end_pos, temporal_policy);
        return r;
      });

  m.def("_create_vector_u64", []() {
    auto v = std::vector<uint64_t>(10);
    // fill vector with range 1:10 using std::iota
    std::iota(v.begin(), v.begin() + 10, 0);
    return v;
  });

  /* === Matrix === */

  declareColMajorMatrix<uint8_t>(m, "_u8");
  declareColMajorMatrix<int8_t>(m, "_i8");
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
  declareColMajorMatrixSubclass<tdbColMajorMatrix<int8_t>>(
      m, "tdbColMajorMatrix", "_i8");
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
  declare_pyarray_to_matrix<int8_t>(m, "_i8");
  declare_pyarray_to_matrix<uint64_t>(m, "_u64");
  declare_pyarray_to_matrix<float>(m, "_f32");
  declare_pyarray_to_matrix<double>(m, "_f64");

  /* === Queries === */

  m.def(
      "query_vq_f32",
      [](ColMajorMatrix<float>& data,
         ColMajorMatrix<float>& query_vectors,
         int k,
         size_t nthreads)
          -> std::tuple<ColMajorMatrix<float>, ColMajorMatrix<uint64_t>> {
        auto r = detail::flat::vq_query_heap(data, query_vectors, k, nthreads);
        return r;
      });

  m.def(
      "query_vq_u8",
      [](tdbColMajorMatrix<uint8_t>& data,
         ColMajorMatrix<float>& query_vectors,
         int k,
         size_t nthreads)
          -> std::tuple<ColMajorMatrix<float>, ColMajorMatrix<uint64_t>> {
        auto r = detail::flat::vq_query_heap(data, query_vectors, k, nthreads);
        return r;
      });

  m.def(
      "query_vq_i8",
      [](tdbColMajorMatrix<int8_t>& data,
         ColMajorMatrix<float>& query_vectors,
         int k,
         size_t nthreads)
          -> std::tuple<ColMajorMatrix<float>, ColMajorMatrix<uint64_t>> {
        auto r = detail::flat::vq_query_heap(data, query_vectors, k, nthreads);
        return r;
      });

  m.def(
      "validate_top_k_u64",
      [](const ColMajorMatrix<uint64_t>& top_k,
         const ColMajorMatrix<int32_t>& ground_truth) -> bool {
        return validate_top_k(top_k, ground_truth);
      });

  declare_vq_query_heap<uint8_t>(m, "u8");
  declare_vq_query_heap<int8_t>(m, "i8");
  declare_vq_query_heap<float>(m, "f32");
  declare_vq_query_heap_pyarray<uint8_t>(m, "u8");
  declare_vq_query_heap_pyarray<int8_t>(m, "i8");
  declare_vq_query_heap_pyarray<float>(m, "f32");

  declare_qv_query_heap_infinite_ram<uint8_t>(m, "u8");
  declare_qv_query_heap_infinite_ram<int8_t>(m, "i8");
  declare_qv_query_heap_infinite_ram<float>(m, "f32");
  declare_qv_query_heap_finite_ram<uint8_t>(m, "u8");
  declare_qv_query_heap_finite_ram<int8_t>(m, "i8");
  declare_qv_query_heap_finite_ram<float>(m, "f32");
  declare_nuv_query_heap_infinite_ram<uint8_t>(m, "u8");
  declare_nuv_query_heap_infinite_ram<int8_t>(m, "i8");
  declare_nuv_query_heap_infinite_ram<float>(m, "f32");
  declare_nuv_query_heap_finite_ram<uint8_t>(m, "u8");
  declare_nuv_query_heap_finite_ram<int8_t>(m, "i8");
  declare_nuv_query_heap_finite_ram<float>(m, "f32");

  declare_ivf_index<uint8_t>(m, "u8");
  declare_ivf_index<int8_t>(m, "i8");
  declare_ivf_index<float>(m, "f32");
  declare_ivf_index_tdb<uint8_t>(m, "u8");
  declare_ivf_index_tdb<int8_t>(m, "i8");
  declare_ivf_index_tdb<float>(m, "f32");

  declarePartitionIvfIndex<uint8_t>(m, "u8");
  declarePartitionIvfIndex<int8_t>(m, "i8");
  declarePartitionIvfIndex<float>(m, "f32");

  declarePartitionedMatrix<uint8_t, uint64_t, uint64_t, uint64_t>(
      m, "tdbPartitionedMatrix", "u8");
  declarePartitionedMatrix<int8_t, uint64_t, uint64_t, uint64_t>(
      m, "tdbPartitionedMatrix", "i8");
  declarePartitionedMatrix<float, uint64_t, uint64_t, uint64_t>(
      m, "tdbPartitionedMatrix", "f32");

  declare_dist_qv<uint8_t>(m, "u8");
  declare_dist_qv<int8_t>(m, "i8");
  declare_dist_qv<float>(m, "f32");
  declareFixedMinPairHeap(m);

  /* === Stats and Debugging === */

  m.def("stats_enable", []() {
    enable_stats = true;
    tiledb::Stats::enable();
  });

  m.def("stats_disable", []() {
    enable_stats = false;
    tiledb::Stats::disable();
  });

  m.def("stats_reset", []() { core_stats.clear(); });
  m.def("stats_dump", []() { return json{core_stats}.dump(); });

  declare_debug_matrix<uint8_t>(m, "_u8");
  declare_debug_matrix<int8_t>(m, "_i8");
  declare_debug_matrix<float>(m, "_f32");
  declare_debug_matrix<uint64_t>(m, "_u64");

  /* === Module inits === */

  init_kmeans(m);
  init_type_erased_module(m);
}
