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
      throw std::runtime_error(
          "[type_erased_module@datatype_to_format] Unsupported datatype");
  }
}

bool check_datatype_format(
    const std::string& dtype_format, const std::string& buffer_info_format) {
  if (dtype_format == buffer_info_format) {
    return true;
  }
  // We need to handle uint64 specifically of a numpy quirk:
  // a. dtype_format (i.e.
  // `datatype_to_format(string_to_datatype(<py::array>.dtype().str()))`) will
  // give us 'Q' (numpy.ulonglong) See:
  //  https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.ulonglong
  // b. buffer_info_format (i.e. `<py::array>.request().format`) will
  // give us 'L' (numpy.uint) because numpy.uint is an alias for numpy.uint64 on
  // Darwin arm64. See:
  //  https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.uint
  if (dtype_format == "Q" && buffer_info_format == "L") {
    return true;
  }
  // The same thing happens with int64, but for it dtype_format will give 'q'
  // (numpy.longlong), whereas buffer_info_format gives 'l' (numpy.int_).
  if (dtype_format == "q" && buffer_info_format == "l") {
    return true;
  }
  return false;
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
          "__init__",
          [](FeatureVector& instance,
             const tiledb::Context& ctx,
             const std::string& uri,
             size_t first_col,
             size_t last_col,
             std::optional<TemporalPolicy> temporal_policy) {
            new (&instance)
                FeatureVector(ctx, uri, first_col, last_col, temporal_policy);
          },
          py::keep_alive<1, 2>(),
          py::arg("ctx"),
          py::arg("uri"),
          py::arg("first_col") = 0,
          py::arg("last_col") = 0,
          py::arg("temporal_policy") = std::nullopt)
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
      .def(py::init([](py::array vector) {
        /* Request a buffer descriptor from Python */
        py::buffer_info info = vector.request();
        if (info.ndim != 1) {
          throw std::runtime_error(
              "[type_erased_module@FeatureVector] Incompatible buffer "
              "dimension. Should be 1, but was " +
              std::to_string(info.ndim) + ".");
        }

        auto dtype_str = vector.dtype().str();
        tiledb_datatype_t datatype = string_to_datatype(dtype_str);
        auto datatype_format = datatype_to_format(datatype);
        if (!check_datatype_format(datatype_format, info.format)) {
          throw std::runtime_error(
              "[type_erased_module@FeatureVector] Incompatible format: "
              "expected array of " +
              datatype_to_string(datatype) + " (" + datatype_format +
              "), but was " + info.format + ".");
        }

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
             size_t first_col,
             size_t last_col,
             std::optional<TemporalPolicy> temporal_policy) {
            new (&instance) FeatureVectorArray(
                ctx, uri, ids_uri, first_col, last_col, temporal_policy);
          },
          py::keep_alive<1, 2>(),  // FeatureVectorArray should keep ctx alive.
          py::arg("ctx"),
          py::arg("uri"),
          py::arg("ids_uri") = "",
          py::arg("first_col") = 0,
          py::arg("last_col") = 0,
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
      .def(
          py::init([](py::array vectors, py::array ids) {
            // The vector buffer info.
            py::buffer_info info = vectors.request();
            if (info.ndim != 2) {
              throw std::runtime_error(
                  "[type_erased_module@FeatureVectorArray] Incompatible buffer "
                  "dimension. Should be 2, but was " +
                  std::to_string(info.ndim) + ".");
            }

            auto dtype_str = vectors.dtype().str();
            tiledb_datatype_t datatype = string_to_datatype(dtype_str);
            auto datatype_format = datatype_to_format(datatype);
            if (!check_datatype_format(datatype_format, info.format)) {
              throw std::runtime_error(
                  "[type_erased_module@FeatureVectorArray] Incompatible format "
                  "- expected array of " +
                  datatype_to_string(datatype) + " (" + datatype_format +
                  "), but was " + info.format + ".");
            }

            // The ids vector buffer info.
            py::buffer_info ids_info = ids.request();
            if (ids_info.ndim != 1) {
              throw std::runtime_error(
                  "[type_erased_module@FeatureVectorArray] Incompatible ids "
                  "buffer dimension. Should be 1, but was " +
                  std::to_string(ids_info.ndim) + ".");
            }

            std::string ids_dtype_str;
            tiledb_datatype_t ids_datatype = TILEDB_ANY;
            if (ids.size() != 0) {
              ids_dtype_str = ids.dtype().str();
              ids_datatype = string_to_datatype(ids_dtype_str);
              auto ids_datatype_format = datatype_to_format(ids_datatype);
              if (!check_datatype_format(
                      ids_datatype_format, ids_info.format)) {
                throw std::runtime_error(
                    "[type_erased_module@FeatureVectorArray] Incompatible ids "
                    "format - expected array of " +
                    datatype_to_string(ids_datatype) + " (" +
                    ids_datatype_format + "), but was " + ids_info.format +
                    ".");
              }
            }

            auto feature_vector_array = [&]() {
              auto order = vectors.flags() & py::array::f_style ?
                               TILEDB_COL_MAJOR :
                               TILEDB_ROW_MAJOR;
              if (order == TILEDB_COL_MAJOR) {
                return FeatureVectorArray(
                    info.shape[0], info.shape[1], dtype_str, ids_dtype_str);
              } else {
                return FeatureVectorArray(
                    info.shape[1], info.shape[0], dtype_str, ids_dtype_str);
              }
            }();

            auto data = (uint8_t*)feature_vector_array.data();
            std::memcpy(
                data,
                (uint8_t*)info.ptr,
                info.shape[0] * info.shape[1] * datatype_to_size(datatype));

            if (ids.size() != 0) {
              std::memcpy(
                  feature_vector_array.ids(),
                  (uint8_t*)ids_info.ptr,
                  ids_info.shape[0] * datatype_to_size(ids_datatype));
            }

            return feature_vector_array;
          }),
          py::arg("vectors"),
          py::arg("ids") = py::array());

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
          [](IndexFlatL2& index, const FeatureVectorArray& vectors, size_t k) {
            auto r = index.query(vectors, k);
            return make_python_pair(std::move(r));
          });

  py::class_<kmeans_init>(m, "kmeans_init")
      .def(py::init([](const std::string& s) {
        if (s == "kmeanspp") {
          return kmeans_init::kmeanspp;
        } else if (s == "random") {
          return kmeans_init::random;
        } else {
          throw std::runtime_error(
              "[type_erased_module@kmeans_init] Invalid kmeans_init value");
        }
      }));

  py::class_<IndexVamana>(m, "IndexVamana")
      .def(
          "__init__",
          [](IndexVamana& instance,
             const tiledb::Context& ctx,
             const std::string& index_uri,
             std::optional<TemporalPolicy> temporal_policy) {
            new (&instance) IndexVamana(ctx, index_uri, temporal_policy);
          },
          py::keep_alive<1, 2>(),  // IndexVamana should keep ctx alive.
          py::arg("ctx"),
          py::arg("index_uri"),
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
             const FeatureVectorArray& vectors,
             size_t k,
             uint32_t l_search) {
            auto r = index.query(vectors, k, l_search);
            return make_python_pair(std::move(r));
          },
          py::arg("vectors"),
          py::arg("k"),
          py::arg("l_search"))
      .def(
          "write_index",
          [](IndexVamana& index,
             const tiledb::Context& ctx,
             const std::string& index_uri,
             std::optional<TemporalPolicy> temporal_policy,
             const std::string& storage_version) {
            index.write_index(ctx, index_uri, temporal_policy, storage_version);
          },
          py::keep_alive<1, 2>(),  // IndexVamana should keep ctx alive.
          py::arg("ctx"),
          py::arg("index_uri"),
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
             const std::string& index_uri,
             uint64_t timestamp) {
            IndexVamana::clear_history(ctx, index_uri, timestamp);
          },
          py::keep_alive<1, 2>(),  // IndexVamana should keep ctx alive.
          py::arg("ctx"),
          py::arg("index_uri"),
          py::arg("timestamp"));

  py::class_<IndexIVFPQ>(m, "IndexIVFPQ")
      .def_static(
          "create",
          [](const tiledb::Context& ctx,
             const std::string& index_uri,
             uint64_t dimensions,
             const std::string& feature_type,
             const std::string& id_type,
             const std::string& partitioning_index_type,
             uint32_t num_subspaces,
             uint32_t max_iterations,
             float convergence_tolerance,
             float reassign_ratio,
             std::optional<TemporalPolicy> temporal_policy,
             DistanceMetric distance_metric,
             const std::string& storage_version) {
            IndexIVFPQ::create(
                ctx,
                index_uri,
                dimensions,
                string_to_datatype(feature_type),
                string_to_datatype(id_type),
                string_to_datatype(partitioning_index_type),
                num_subspaces,
                max_iterations,
                convergence_tolerance,
                reassign_ratio,
                temporal_policy,
                distance_metric,
                storage_version);
          },
          py::keep_alive<1, 2>(),  // IndexIVFPQ should keep ctx alive.
          py::arg("ctx"),
          py::arg("index_uri"),
          py::arg("dimensions"),
          py::arg("feature_type"),
          py::arg("id_type") = "uint64",
          py::arg("partitioning_index_type") = "uint64",
          py::arg("num_subspaces") = 16,
          py::arg("max_iterations") = 10,
          py::arg("convergence_tolerance") = 0.0001f,
          py::arg("reassign_ratio") = 0.075f,
          py::arg("temporal_policy") = std::nullopt,
          py::arg("distance_metric") = DistanceMetric::SUM_OF_SQUARES,
          py::arg("storage_version") = "")
      .def(
          "__init__",
          [](IndexIVFPQ& instance,
             const tiledb::Context& ctx,
             const std::string& index_uri,
             IndexLoadStrategy index_load_strategy,
             size_t memory_budget,
             std::optional<TemporalPolicy> temporal_policy) {
            new (&instance) IndexIVFPQ(
                ctx,
                index_uri,
                index_load_strategy,
                memory_budget,
                temporal_policy);
          },
          py::keep_alive<1, 2>(),  // IndexIVFPQ should keep ctx alive.
          py::arg("ctx"),
          py::arg("index_uri"),
          py::arg("index_load_strategy") = IndexLoadStrategy::PQ_INDEX,
          py::arg("memory_budget") = 0,
          py::arg("temporal_policy") = std::nullopt)
      .def(
          "create_temp_data_group",
          [](IndexIVFPQ& index, const std::string& partial_write_array_dir) {
            index.create_temp_data_group(partial_write_array_dir);
          },
          py::arg("partial_write_array_dir"))
      .def(
          "train",
          [](IndexIVFPQ& index,
             const FeatureVectorArray& vectors,
             size_t partitions,
             std::optional<TemporalPolicy> temporal_policy = std::nullopt) {
            index.train(vectors, partitions, temporal_policy);
          },
          py::arg("vectors"),
          py::arg("partitions") = 0,
          py::arg("temporal_policy") = std::nullopt)
      .def(
          "ingest",
          [](IndexIVFPQ& index,
             const FeatureVectorArray& input_vectors,
             const FeatureVector& external_ids) {
            index.ingest(input_vectors, external_ids);
          },
          py::arg("input_vectors"),
          py::arg("external_ids"))
      .def(
          "ingest",
          [](IndexIVFPQ& index, const FeatureVectorArray& input_vectors) {
            index.ingest(input_vectors);
          },
          py::arg("input_vectors"))
      .def(
          "ingest_parts",
          [](IndexIVFPQ& index,
             const FeatureVectorArray& input_vectors,
             const FeatureVector& external_ids,
             const FeatureVector& deleted_ids,
             size_t start,
             size_t end,
             size_t partition_start,
             const std::string& partial_write_array_dir) {
            index.ingest_parts(
                input_vectors,
                external_ids,
                deleted_ids,
                start,
                end,
                partition_start,
                partial_write_array_dir);
          },
          py::arg("input_vectors"),
          py::arg("external_ids"),
          py::arg("deleted_ids"),
          py::arg("start"),
          py::arg("end"),
          py::arg("partition_start"),
          py::arg("partial_write_array_dir"))
      .def(
          "consolidate_partitions",
          [](IndexIVFPQ& index,
             size_t partitions,
             size_t work_items,
             size_t partition_id_start,
             size_t partition_id_end,
             size_t batch,
             const std::string& partial_write_array_dir) {
            index.consolidate_partitions(
                partitions,
                work_items,
                partition_id_start,
                partition_id_end,
                batch,
                partial_write_array_dir);
          },
          py::arg("partitions"),
          py::arg("work_items"),
          py::arg("partition_id_start"),
          py::arg("partition_id_end"),
          py::arg("batch"),
          py::arg("partial_write_array_dir"))
      .def(
          "ingest",
          [](IndexIVFPQ& index, const FeatureVectorArray& input_vectors) {
            index.ingest(input_vectors);
          },
          py::arg("input_vectors"))
      .def(
          "query",
          [](IndexIVFPQ& index,
             const FeatureVectorArray& vectors,
             size_t k,
             size_t nprobe,
             float k_factor) {
            auto r = index.query(vectors, k, nprobe, k_factor);
            return make_python_pair(std::move(r));
          },
          py::arg("vectors"),
          py::arg("k"),
          py::arg("nprobe"),
          py::arg("k_factor") = 1.f)
      .def("feature_type_string", &IndexIVFPQ::feature_type_string)
      .def("id_type_string", &IndexIVFPQ::id_type_string)
      .def(
          "partitioning_index_type_string",
          &IndexIVFPQ::partitioning_index_type_string)
      .def("dimensions", &IndexIVFPQ::dimensions)
      .def("partitions", &IndexIVFPQ::partitions)
      .def_static(
          "clear_history",
          [](const tiledb::Context& ctx,
             const std::string& index_uri,
             uint64_t timestamp) {
            IndexIVFPQ::clear_history(ctx, index_uri, timestamp);
          },
          py::keep_alive<1, 2>(),  // IndexIVFPQ should keep ctx alive.
          py::arg("ctx"),
          py::arg("index_uri"),
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
              throw std::runtime_error(
                  "[type_erased_module@IndexIVFFlat@train] Invalid kmeans_init "
                  "value");
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
             size_t k,
             size_t nprobe) {
            auto r = index.query_infinite_ram(query, k, nprobe);
            return make_python_pair(std::move(r));
          })  //  , py::arg("vectors"), py::arg("k") = 1, py::arg("nprobe")
              //  = 10)
      .def(
          "query_finite_ram",
          [](IndexIVFFlat& index,
             const FeatureVectorArray& query,
             size_t k,
             size_t nprobe,
             size_t upper_bound) {
            auto r = index.query_finite_ram(query, k, nprobe, upper_bound);
            return make_python_pair(std::move(r));
          },
          py::arg("vectors"),
          py::arg("k") = 1,
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

  m.def("logging_string", []() { return logging_string(); });
}
