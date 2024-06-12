# -----------------------------------------------------------------------------
# Allow our executables to use mdspan.
# -----------------------------------------------------------------------------


message(STATUS "Fetching mdspan...")
include(FetchContent)

FetchContent_Declare(
        mdspan
        GIT_REPOSITORY https://github.com/kokkos/mdspan.git
        GIT_TAG 96b3985b291c2ed3e24f22302d5a372522655a2c)

FetchContent_MakeAvailable(mdspan)
