#
# Interface library for BLAS support
#
add_library(flat_blas INTERFACE)

find_package(BLAS REQUIRED)
target_link_libraries(flat_blas INTERFACE BLAS::BLAS)

if (USE_MKL_CBLAS)
  target_compile_definitions(flat_blas USE_MKL)
endif()