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

  include(FindBLAS)

  if (BLAS_FOUND)
    message(STATUS "BLAS FOUND ${BLAS_FOUND}")
    message(STATUS "${BLAS_LINKER_FLAGS}")
    message(STATUS "${BLAS_LIBRARIES}")
    target_link_libraries(flat_blas INTERFACE $<LINK_ONLY:BLAS::BLAS>)
    target_compile_definitions(flat_blas INTERFACE TDB_BLAS)
  else()
    message(STATUS "BLAS not found")
  endif()

endif()

