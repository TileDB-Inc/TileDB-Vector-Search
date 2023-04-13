cmake_minimum_required(VERSION 3.25)

# The cmake options don't seem to really work
if ("${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
    message(FATAL_ERROR "In-source builds are disabled.
    Please create a subfolder and use `cmake ..` inside it.
    IMPORTANT: cmake will still have created CMakeCache.txt and CMakeFiles/*.
               You must delete them, or cmake will refuse to work.")
endif() # yes, end-markers and even else() need empty parens

project(tdb)

set(CMAKE_CXX_STANDARD 20)

# -----------------------------------------------------------------------------
# Extend the module path so we can find our custom modules
# -----------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# -----------------------------------------------------------------------------
# Build types and default flags
# -----------------------------------------------------------------------------

message("Architecture is ${CMAKE_HOST_SYSTEM_PROCESSOR}")

if (CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL arm64)
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fno-elide-constructors -fconcepts-diagnostics-depth=3 " CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG " CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-Ofast -g -DNDEBUG" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG " CACHE STRING "" FORCE)
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fno-elide-constructors -fconcepts-diagnostics-depth=3 " CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native -DNDEBUG " CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-Ofast -g -march=native -DNDEBUG" CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -march=native -DNDEBUG " CACHE STRING "" FORCE)
endif()

# Default to Release build
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)
endif()

# Control compiler-specific flags.
include(CompilerFlags)

include(Docopt)
add_subdirectory(src)
add_subdirectory(src/test)
