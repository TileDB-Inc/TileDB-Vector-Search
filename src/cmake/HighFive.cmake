

message(STATUS "Fetching HighFive...")
include(FetchContent)

FetchContent_Declare(
HighFive
GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
)

FetchContent_MakeAvailable(HighFive)
