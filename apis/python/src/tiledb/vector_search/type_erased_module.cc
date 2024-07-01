/**
 * @file   tiledb/vector_search/type_erased_module.cc
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2023 TileDB, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * @section DESCRIPTION
 */

#include <tiledb/tiledb>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "api/api_defs.h"
#include "api/feature_vector.h"
#include "api/feature_vector_array.h"
#include "api/flat_l2_index.h"
#include "api/ivf_flat_index.h"
#include "api/ivf_pq_index.h"
#include "api/vamana_index.h"
#include "detail/time/temporal_policy.h"
#include "index/index_defs.h"
#include "stats.h"

namespace py = pybind11;

namespace {
template <typename... TArgs>
py::tuple make_python_pair(std::tuple<TArgs...>&& arg) {
  static_assert(sizeof...(TArgs) == 2, "Must have exactly two arguments");

  return py::make_tuple<py::return_value_policy::automatic>(
      py::cast(std::get<0>(arg), py::return_value_policy::move),
      py::cast(std::get<1>(arg), py::return_value_policy::move));
}

std::map<std::string, std::string> kwargs_to_map(py::kwargs kwargs) {
  std::map<std::string, std::string> result;

  for (auto item : kwargs) {
    // Convert the Python objects to strings
    std::string key = py::str(item.first);
    std::string value = py::str(item.second);

    result[key] = value;
  }

  return result;
}

}  // namespace

auto datatype_to_format(tiledb_datatype_t datatype) {
  switch (datatype) {
    case TILEDB_FLOAT32:
      return py::format_descriptor<float>::format();
    case TILEDB_FLOAT64:
      return py::format_descriptor<double>::format();
    case TILEDB_INT8:
      return py::format_descriptor<int8_t>::format();
    case TILEDB_UINT8:
      return py::format_descriptor<uint8_t>::format();
    case TILEDB_INT16:
      return py::format_descriptor<int16_t>::format();
    case TILEDB_UINT16:
      return py::format_descriptor<uint16_t>::format();
    case TILEDB_INT32:
      return py::format_descriptor<int32_t>::format();
    case TILEDB_UINT32:
      return py::format_descriptor<uint32_t>::format();
    case TILEDB_INT64:
      return py::format_descriptor<int64_t>::format();
    case TILEDB_UINT64:
      return py::format_descriptor<uint64_t>::format();
    default:
      throw std::runtime_error("Unsupported datatype");
  }
}

// Define Pybind11 bindings

// PYBIND11_MODULE(_tiledbvspy2, m) {
void init_type_erased_module(py::module_& m) {
  m.def(
      "count_intersections",
      [](const FeatureVectorArray& a,
         const FeatureVectorArray& b,
         size_t k_nn) { return count_intersections(a, b, k_nn); });
#if 0
  py::class_<tiledb::Context> (m, "Ctx", py::module_local())
      .def(py::init([](std::optional<py::dict> maybe_config) {
        tiledb::Config cfg;
        if (maybe_config.has_value()) {
          for (auto item : maybe_config.value()) {
            cfg.set(item.first.cast<std::string>(), item.second.cast<std::string>());
          }
        }
        return tiledb::Context(cfg);
      }))
      ;
#endif
  py::class_<TemporalPolicy>(m, "TemporalPolicy", py::buffer_protocol())
      // From 0 to UINT64_MAX.
      .def(py::init<>())
      // From 0 to timestamp_end.
      .def(
          "__init__",
          [](TemporalPolicy& instance,
             std::optional<uint64_t> timestamp_end_input) {
            uint64_t timestamp_end = timestamp_end_input.has_value() ?
                                         timestamp_end_input.value() :
                                         UINT64_MAX;
            new (&instance) TemporalPolicy(TimeTravel, timestamp_end);
          })
      // From timestamp_start to timestamp_end.
      .def(
          "__init__",
          [](TemporalPolicy& instance,
             std::optional<uint64_t> timestamp_start_input,
             std::optional<uint64_t> timestamp_end_input) {
            uint64_t timestamp_start = timestamp_start_input.has_value() ?
                                           timestamp_start_input.value() :
                                           0;
            uint64_t timestamp_end = timestamp_end_input.has_value() ?
                                         timestamp_end_input.value() :
                                         UINT64_MAX;
            new (&instance) TemporalPolicy(
                TimestampStartEnd, timestamp_start, timestamp_end);
          })
      .def("timestamp_start", &TemporalPolicy::timestamp_start)
      .def("timestamp_end", &TemporalPolicy::timestamp_end);

  py::class_<FeatureVector>(m, "FeatureVector", py::buffer_protocol())
      .def(
          py::init<const tiledb::Context&, const std::string&>(),
          py::keep_alive<1, 2>()  // IndexIVFFlat should keep ctx alive.
          )
      .def(py::init<size_t, const std::string&>())
      .def(py::init<size_t, void*, const std::string&>())
      .def("dimensions", &FeatureVector::dimensions)
      .def("feature_type", &FeatureVector::feature_type)
      .def("feature_type_string", &FeatureVector::feature_type_string)
      .def_buffer([](FeatureVector& v) -> py::buffer_info {
        return py::buffer_info(
            v.data(),                           /* Pointer to buffer */
            datatype_to_size(v.feature_type()), /* Size of one scalar */
            datatype_to_format(
                v.feature_type()), /* Python struct-style format descriptor */
            1,                     /* Number of dimensions */
            {v.dimensions()},      /* Buffer dimension */
            {datatype_to_size(v.feature_type())}
            /* Strides (in bytes) for each index */
        );
      })
      .def(py::init([](py::array b) {
        /* Request a buffer descriptor from Python */
        py::buffer_info info = b.request();
        if (info.ndim != 1)
          throw std::runtime_error(
              "Incompatible buffer dimension! Should be 1.");

        auto dtype_str = b.dtype().str();
        tiledb_datatype_t datatype = string_to_datatype(dtype_str);
        if (info.format != datatype_to_format(datatype))
          throw std::runtime_error(
              "Incompatible format: expected array of " +
              datatype_to_string(datatype));

        size_t sz = datatype_to_size(datatype);

        auto v = FeatureVector(info.shape[0], dtype_str);

        auto data = (uint8_t*)v.data();
        std::memcpy(data, (uint8_t*)info.ptr, info.shape[0] * sz);

        return v;
      }));

  py::class_<FeatureVectorArray>(m, "FeatureVectorArray", py::buffer_protocol())
      .def(
          py::init<const tiledb::Context&, const std::string&>(),
          py::keep_alive<1, 2>()  // FeatureVectorArray should keep ctx alive.
          )
      .def(
          "__init__",
          [](FeatureVectorArray& instance,
             const tiledb::Context& ctx,
             const std::string& uri,
             const std::string& ids_uri,
             size_t num_vectors,
             std::optional<TemporalPolicy> temporal_policy) {
            new (&instance) FeatureVectorArray(
                ctx, uri, ids_uri, num_vectors, temporal_policy);
          },
          py::keep_alive<1, 2>(),  // FeatureVectorArray should keep ctx alive.
          py::arg("ctx"),
          py::arg("uri"),
          py::arg("ids_uri") = "",
          py::arg("num_vectors") = 0,
          py::arg("temporal_policy") = std::nullopt)
      .def(py::init<size_t, size_t, const std::string&, const std::string&>())
      .def("dimensions", &FeatureVectorArray::dimensions)
      .def("num_vectors", &FeatureVectorArray::num_vectors)
      .def("feature_type", &FeatureVectorArray::feature_type)
      .def("feature_type_string", &FeatureVectorArray::feature_type_string)
      .def("num_ids", &FeatureVectorArray::num_ids)
      .def("ids_type", &FeatureVectorArray::ids_type)
      .def("ids_type_string", &FeatureVectorArray::ids_type_string)
      .def_buffer([](FeatureVectorArray& v) -> py::buffer_info {
        return py::buffer_info(
            v.data(),                           /* Pointer to buffer */
            datatype_to_size(v.feature_type()), /* Size of one scalar */
            datatype_to_format(
                v.feature_type()), /* Python struct-style format descriptor */
            2,                     /* Number of dimensions */
            {v.num_vectors(),
             v.dimensions()}, /* Buffer dimensions -- row major */
            {datatype_to_size(v.feature_type()) *
                 v.dimensions(), /* Strides (in bytes) for each index */
             datatype_to_size(v.feature_type())});
      })
      .def(py::init([](py::array b) {
        /* Request a buffer descriptor from Python */
        py::buffer_info info = b.request();
        if (info.ndim != 2)
          throw std::runtime_error(
              "Incompatible buffer dimension! Should be 2.");

        auto dtype_str = b.dtype().str();
        tiledb_datatype_t datatype = string_to_datatype(dtype_str);
        if (info.format != datatype_to_format(datatype))
          throw std::runtime_error(
              "Incompatible format: expected array of " +
              datatype_to_string(datatype));

        size_t sz = datatype_to_size(datatype);

        auto v = [&]() {
          auto order = b.flags() & py::array::f_style ? TILEDB_COL_MAJOR :
                                                        TILEDB_ROW_MAJOR;
          if (order == TILEDB_COL_MAJOR) {
            return FeatureVectorArray(info.shape[0], info.shape[1], dtype_str);
          } else {
            return FeatureVectorArray(info.shape[1], info.shape[0], dtype_str);
          }
        }();

        auto data = (uint8_t*)v.data();
        std::memcpy(
            data, (uint8_t*)info.ptr, info.shape[0] * info.shape[1] * sz);

        return v;
      }));

  py::class_<IndexFlatL2>(m, "IndexFlatL2")
      .def(
          py::init<const tiledb::Context&, const std::string&>(),
          py::keep_alive<1, 2>()  // IndexFlatL2 should keep ctx alive.
          )
      .def("add", &IndexFlatL2::add)
      .def("add_with_ids", &IndexFlatL2::add_with_ids)
      .def("train", &IndexFlatL2::train)
      .def("save", &IndexFlatL2::save)
      .def("feature_type_string", &IndexFlatL2::feature_type_string)
      .def("dimensions", &IndexFlatL2::dimensions)
      .def(
          "query",
          [](IndexFlatL2& index, FeatureVectorArray& vectors, size_t top_k) {
            auto r = index.query(vectors, top_k);
            return make_python_pair(std::move(r));
          });

  py::class_<kmeans_init>(m, "kmeans_init")
      .def(py::init([](const std::string& s) {
        if (s == "kmeanspp") {
          return kmeans_init::kmeanspp;
        } else if (s == "random") {
          return kmeans_init::random;
        } else {
          throw std::runtime_error("Invalid kmeans_init value");
        }
      }));

  py::class_<IndexVamana>(m, "IndexVamana")
      .def(
          "__init__",
          [](IndexVamana& instance,
             const tiledb::Context& ctx,
             const std::string& group_uri,
             std::optional<TemporalPolicy> temporal_policy) {
            new (&instance) IndexVamana(ctx, group_uri, temporal_policy);
          },
          py::keep_alive<1, 2>(),  // IndexVamana should keep ctx alive.
          py::arg("ctx"),
          py::arg("group_uri"),
          py::arg("temporal_policy") = std::nullopt)
      .def(
          "__init__",
          [](IndexVamana& instance, py::kwargs kwargs) {
            auto args = kwargs_to_map(kwargs);
            new (&instance) IndexVamana(args);
          })
      .def(
          "train",
          [](IndexVamana& index, const FeatureVectorArray& vectors) {
            index.train(vectors);
          },
          py::arg("vectors"))
      .def(
          "add",
          [](IndexVamana& index, const FeatureVectorArray& vectors) {
            index.add(vectors);
          },
          py::arg("vectors"))
      .def(
          "query",
          [](IndexVamana& index,
             FeatureVectorArray& vectors,
             size_t top_k,
             size_t l_search) {
            auto r = index.query(vectors, top_k, l_search);
            return make_python_pair(std::move(r));
          },
          py::arg("vectors"),
          py::arg("top_k"),
          py::arg("l_search"))
      .def(
          "write_index",
          [](IndexVamana& index,
             const tiledb::Context& ctx,
             const std::string& group_uri,
             std::optional<TemporalPolicy> temporal_policy,
             const std::string& storage_version) {
            index.write_index(ctx, group_uri, temporal_policy, storage_version);
          },
          py::keep_alive<1, 2>(),  // IndexVamana should keep ctx alive.
          py::arg("ctx"),
          py::arg("group_uri"),
          py::arg("temporal_policy") = std::nullopt,
          py::arg("storage_version") = "")
      .def("feature_type_string", &IndexVamana::feature_type_string)
      .def("id_type_string", &IndexVamana::id_type_string)
      .def("dimensions", &IndexVamana::dimensions)
      .def("l_build", &IndexVamana::l_build)
      .def("r_max_degree", &IndexVamana::r_max_degree)
      .def_static(
          "clear_history",
          [](const tiledb::Context& ctx,
             const std::string& group_uri,
             uint64_t timestamp) {
            IndexVamana::clear_history(ctx, group_uri, timestamp);
          },
          py::keep_alive<1, 2>(),  // IndexVamana should keep ctx alive.
          py::arg("ctx"),
          py::arg("group_uri"),
          py::arg("timestamp"));

  py::class_<IndexIVFPQ>(m, "IndexIVFPQ")
      .def(
          "__init__",
          [](IndexIVFPQ& instance,
             const tiledb::Context& ctx,
             const std::string& group_uri,
             std::optional<TemporalPolicy> temporal_policy) {
            new (&instance) IndexIVFPQ(ctx, group_uri, temporal_policy);
          },
          py::keep_alive<1, 2>(),  // IndexIVFPQ should keep ctx alive.
          py::arg("ctx"),
          py::arg("group_uri"),
          py::arg("temporal_policy") = std::nullopt)
      .def(
          "__init__",
          [](IndexIVFPQ& instance, py::kwargs kwargs) {
            auto args = kwargs_to_map(kwargs);
            new (&instance) IndexIVFPQ(args);
          })
      .def(
          "train",
          [](IndexIVFPQ& index, const FeatureVectorArray& vectors) {
            index.train(vectors);
          },
          py::arg("vectors"))
      .def(
          "add",
          [](IndexIVFPQ& index, const FeatureVectorArray& vectors) {
            index.add(vectors);
          },
          py::arg("vectors"))
      .def(
          "query",
          [](IndexIVFPQ& index,
             QueryType queryType,
             FeatureVectorArray& vectors,
             size_t top_k,
             size_t nprobe) {
            auto r = index.query(queryType, vectors, top_k, nprobe);
            return make_python_pair(std::move(r));
          },
          py::arg("queryType"),
          py::arg("vectors"),
          py::arg("top_k"),
          py::arg("nprobe"))
      .def(
          "write_index",
          [](IndexIVFPQ& index,
             const tiledb::Context& ctx,
             const std::string& group_uri,
             std::optional<TemporalPolicy> temporal_policy,
             const std::string& storage_version) {
            index.write_index(ctx, group_uri, temporal_policy, storage_version);
          },
          py::keep_alive<1, 2>(),  // IndexIVFPQ should keep ctx alive.
          py::arg("ctx"),
          py::arg("group_uri"),
          py::arg("temporal_policy") = std::nullopt,
          py::arg("storage_version") = "")
      .def("feature_type_string", &IndexIVFPQ::feature_type_string)
      .def("id_type_string", &IndexIVFPQ::id_type_string)
      .def(
          "partitioning_index_type_string",
          &IndexIVFPQ::partitioning_index_type_string)
      .def("dimensions", &IndexIVFPQ::dimensions)
      .def_static(
          "clear_history",
          [](const tiledb::Context& ctx,
             const std::string& group_uri,
             uint64_t timestamp) {
            IndexIVFPQ::clear_history(ctx, group_uri, timestamp);
          },
          py::keep_alive<1, 2>(),  // IndexIVFPQ should keep ctx alive.
          py::arg("ctx"),
          py::arg("group_uri"),
          py::arg("timestamp"));

  py::class_<IndexIVFFlat>(m, "IndexIVFFlat")
      .def(
          py::init<const tiledb::Context&, const std::string&>(),
          py::keep_alive<1, 2>()  // IndexIVFFlat should keep ctx alive.
          )
      .def(
          "__init__",
          [](IndexIVFFlat& instance, py::kwargs kwargs) {
            auto args = kwargs_to_map(kwargs);
            new (&instance) IndexIVFFlat(args);
          })
      .def(
          "train",
          [](IndexIVFFlat& index,
             const FeatureVectorArray& vectors,
             py::str init_str) {
            kmeans_init init = kmeans_init::random;
            if (std::string(init_str) == "kmeans++") {
              init = kmeans_init::kmeanspp;
            } else if (std::string(init_str) == "random") {
              init = kmeans_init::random;
            } else {
              throw std::runtime_error("Invalid kmeans_init value");
            }
            index.train(vectors, init);
          },
          py::arg("vectors"),
          py::arg("init") = "random")
      .def(
          "add",
          [](IndexIVFFlat& index, const FeatureVectorArray& vectors) {
            index.add(vectors);
          })
      .def("add_with_ids", &IndexIVFFlat::add_with_ids)
      // .def("save", &IndexIVFFlat::save)
      .def(
          "query_infinite_ram",
          [](IndexIVFFlat& index,
             const FeatureVectorArray& query,
             size_t top_k,
             size_t nprobe) {
            auto r = index.query_infinite_ram(query, top_k, nprobe);
            return make_python_pair(std::move(r));
          })  //  , py::arg("vectors"), py::arg("top_k") = 1, py::arg("nprobe")
              //  = 10)
      .def(
          "query_finite_ram",
          [](IndexIVFFlat& index,
             const FeatureVectorArray& query,
             size_t top_k,
             size_t nprobe,
             size_t upper_bound) {
            auto r = index.query_finite_ram(query, top_k, nprobe, upper_bound);
            return make_python_pair(std::move(r));
          },
          py::arg("vectors"),
          py::arg("top_k") = 1,
          py::arg("nprobe") = 10,
          py::arg("upper_bound") = 0)
      .def("feature_type_string", &IndexIVFFlat::feature_type_string)
      .def("id_type_string", &IndexIVFFlat::id_type_string)
      .def("px_type_string", &IndexIVFFlat::px_type_string)
      .def("dimensions", &IndexIVFFlat::dimensions);

  py::enum_<QueryType>(m, "QueryType")
      .value("FiniteRAM", QueryType::FiniteRAM)
      .value("InfiniteRAM", QueryType::InfiniteRAM)
      .export_values();

  m.def("build_config_string", []() { return build_config().dump(); });
}
