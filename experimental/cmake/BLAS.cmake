#
# Interface library for BLAS support
#
add_library(flat_blas INTERFACE)

find_package(BLAS REQUIRED)
target_link_libraries(flat_blas INTERFACE BLAS::BLAS)

get_target_property(foo BLAS::BLAS INCLUDE_DIRECTORIES)

if (USE_MKL_CBLAS)
  target_compile_definitions(flat_blas USE_MKL)
endif()

if (BLAS_LIBRARIES MATCHES "cblas" OR BLAS_LIBRARIES MATCHES "openblas")
  find_path (BLAS_INCLUDE_DIR
    NAMES cblas.h
    DOC "Include directory of the cblas.h header file."
    REQUIRED
    HINTS
      "/usr/include/openblas")
  target_include_directories(BLAS::BLAS INTERFACE ${BLAS_INCLUDE_DIR})
endif ()
