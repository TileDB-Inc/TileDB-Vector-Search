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

#include "api/feature_vector.h"
#include "api/feature_vector_array.h"

#include "api/flat_l2_index.h"
#include "api/ivf_flat_index.h"
#include "api/vamana_index.h"

#include "api/api_defs.h"

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
  py::class_<FeatureVector>(m, "FeatureVector", py::buffer_protocol())
      .def(py::init<const tiledb::Context&, const std::string&>())
      .def(py::init<size_t, const std::string&>())
      .def(py::init<size_t, void*, const std::string&>())
      .def("dimension", &FeatureVector::dimension)
      .def("feature_type", &FeatureVector::feature_type)
      .def("feature_type_string", &FeatureVector::feature_type_string)
      .def_buffer([](FeatureVector& v) -> py::buffer_info {
        return py::buffer_info(
            v.data(),                           /* Pointer to buffer */
            datatype_to_size(v.feature_type()), /* Size of one scalar */
            datatype_to_format(
                v.feature_type()), /* Python struct-style format descriptor */
            1,                     /* Number of dimensions */
            {v.dimension()},       /* Buffer dimension */
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
      .def(py::init<const tiledb::Context&, const std::string&>())
      //      .def(py::init<size_t, size_t, const std::string&>())
      //      .def(py::init<size_t, size_t void*, const std::string&>())
      .def("dimension", &FeatureVectorArray::dimension)
      .def("num_vectors", &FeatureVectorArray::num_vectors)
      .def("feature_type", &FeatureVectorArray::feature_type)
      .def("feature_type_string", &FeatureVectorArray::feature_type_string)
      .def_buffer([](FeatureVectorArray& v) -> py::buffer_info {
        return py::buffer_info(
            v.data(),                           /* Pointer to buffer */
            datatype_to_size(v.feature_type()), /* Size of one scalar */
            datatype_to_format(
                v.feature_type()), /* Python struct-style format descriptor */
            2,                     /* Number of dimensions */
            {v.num_vectors(),
             v.dimension()}, /* Buffer dimensions -- row major */
            {datatype_to_size(v.feature_type()) *
                 v.dimension(), /* Strides (in bytes) for each index */
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
      .def(py::init<const tiledb::Context&, const std::string&>())
      .def("add", &IndexFlatL2::add)
      .def("add_with_ids", &IndexFlatL2::add_with_ids)
      .def("train", &IndexFlatL2::train)
      .def("save", &IndexFlatL2::save)
      .def("feature_type_string", &IndexFlatL2::feature_type_string)
      .def("dimension", &IndexFlatL2::dimension)
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
      .def(py::init<const tiledb::Context&, const std::string&>())
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
          // TODO(paris): Update opt_l to be optional.
          [](IndexVamana& index,
             FeatureVectorArray& vectors,
             size_t top_k,
             size_t opt_l) {
            auto r = index.query(vectors, top_k, opt_l);
            return make_python_pair(std::move(r));
          },
          py::arg("vectors"),
          py::arg("top_k"),
          py::arg("opt_l"))
      .def("feature_type_string", &IndexVamana::feature_type_string)
      .def("id_type_string", &IndexVamana::id_type_string)
      .def("px_type_string", &IndexVamana::px_type_string)
      .def("dimension", &IndexVamana::dimension);

  py::class_<IndexIVFFlat>(m, "IndexIVFFlat")
      .def(py::init<const tiledb::Context&, const std::string&>())
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
      .def("dimension", &IndexIVFFlat::dimension);
}
