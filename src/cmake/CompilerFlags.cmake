# https://software.intel.com/en-us/forums/intel-c-compiler/topic/799473

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS "-Wall -Wno-unused-variable ${CMAKE_CXX_FLAGS}")
endif ()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  set(CMAKE_CXX_FLAGS "-Wall -Wno-unused-variable ${CMAKE_CXX_FLAGS}")
endif ()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "-Wall -Wno-unused-variable ${CMAKE_CXX_FLAGS}")
  if ("${CMAKE_CXX_COMPILER}" MATCHES ".*dpcpp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas -Wno-unused-lambda-capture -Wno-unused-local-typedef -Wno-reorder-ctor")
  endif ()
endif ()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "-Wall -Wno-unused-variable ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "-Wno-psabi ${CMAKE_CXX_FLAGS}")
  if(CMAKE_OSX_ARCHITECTURES STREQUAL arm64 OR CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
    message(STATUS "++++ Setting CMAKE_CXX_FLAGS to ${CMAKE_CXX_FLAGS} -flax-vector-conversions")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flax-vector-conversions")
  endif()
endif ()
