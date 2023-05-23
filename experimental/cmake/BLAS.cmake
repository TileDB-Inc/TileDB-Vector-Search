#
# Interface library for BLAS support
#
add_library(flat_blas INTERFACE)

set(ignore_me "${BLA_VENDOR}")
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  # include macOS-specific flags for native BLAS
  target_compile_options(flat_blas INTERFACE -framework Accelerate)
  target_link_options(flat_blas INTERFACE -framework Accelerate)
else()
  # Try to find MKL first. find_package(BLAS) cannot find it due to a bug.
  # Possible related: https://gitlab.kitware.com/cmake/cmake/-/issues/23520
  find_package(MKL)
  include_directories(${MKL_INCLUDE_DIRS})

  target_compile_options(flat_blas INTERFACE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
  target_include_directories(flat_blas INTERFACE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
  target_link_libraries(flat_blas INTERFACE $<LINK_ONLY:MKL::MKL>)

  if(NOT MKL_FOUND)
    # Fall back to CMake's mechanism for finding BLAS.
    find_package(BLAS REQUIRED)
    target_link_libraries(flat_blas INTERFACE Blas::Blas)
  endif()
endif()
