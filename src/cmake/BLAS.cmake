#
# Interface library for BLAS support
#
add_library(flat_blas INTERFACE)

if (USE_MKL_CBLAS)

  # find_package(MKL CONFIG REQUIRED)
  find_package(MKL REQUIRED)
  include_directories(${MKL_INCLUDE_DIRS})

  target_compile_options(flat_blas INTERFACE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
  target_include_directories(flat_blas INTERFACE $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
  target_link_libraries(flat_blas INTERFACE $<LINK_ONLY:MKL::MKL>)

  target_compile_definitions(flat_blas INTERFACE USE_MKL)
  target_compile_definitions(flat_blas INTERFACE TDB_BLAS)

else()
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set (BLA_VENDOR Apple)
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set (BLA_VENDOR OpenBLAS)
  endif()

  find_package(BLAS REQUIRED)

  if (BLAS_FOUND)
    message(STATUS "BLAS_FOUND ${BLAS_FOUND}, BLAS_VENDOR: ${BLA_VENDOR}")
    message(STATUS "  BLAS_LINKER_FLAGS: ${BLAS_LINKER_FLAGS}")
    message(STATUS "  BLAS_LIBRARIES: ${BLAS_LIBRARIES}")

    if (NOT DEFINED BLAS_INCLUDE_DIR)
        # https://gitlab.kitware.com/cmake/cmake/-/issues/20268
        cmake_path(GET BLAS_LIBRARIES PARENT_PATH PARENT_DIR)
        cmake_path(GET PARENT_DIR PARENT_PATH BLAS_SEARCH_ROOT)
        find_path(BLAS_INCLUDE_DIR
          NAMES cblas.h
          PATHS ${BLAS_SEARCH_ROOT}
          PATH_SUFFIXES include/openblas include
          NO_DEFAULT_PATH
        )
    endif()
    message(STATUS "  BLAS_INCLUDE_DIR: ${BLAS_INCLUDE_DIR}")

    target_include_directories(flat_blas INTERFACE ${BLAS_INCLUDE_DIR})
    target_link_libraries(flat_blas INTERFACE BLAS::BLAS)
    target_compile_definitions(flat_blas INTERFACE TDB_BLAS)
  else()
    message(STATUS "BLAS not found")
  endif()

endif()

target_compile_definitions(flat_blas INTERFACE TILEDB_VS_ENABLE_BLAS)
