#
# Support for nlohmann/json
#

find_package(nlohmann_json 3.2.0 QUIET)

if (NOT nlohmann_json_FOUND)
    message("Installed json not found -- fetching")
    FetchContent_Declare(json
            GIT_REPOSITORY https://github.com/nlohmann/json.git
            GIT_TAG v3.7.3)

    FetchContent_GetProperties(json)
    if(NOT json_POPULATED)
        FetchContent_Populate(json)
        set(JSON_BuildTests OFF CACHE INTERNAL "")
        add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
endif()
