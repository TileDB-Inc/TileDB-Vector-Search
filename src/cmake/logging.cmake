if(EXISTS ${CMAKE_SOURCE_DIR}/../.git)

# Get the current date and time
string(TIMESTAMP CURRENT_DATETIME "%Y-%m-%d %H:%M:%S")

# Execute "git rev-parse --abbrev-ref HEAD" command to get the current branch name
execute_process(
        COMMAND git rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the Git repository URL
execute_process(
        COMMAND git config --get remote.origin.url
        OUTPUT_VARIABLE GIT_REPO_URL
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the Git repository name
execute_process(
        COMMAND git rev-parse --show-toplevel
        OUTPUT_VARIABLE GIT_REPO_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
get_filename_component(GIT_REPO_NAME ${GIT_REPO_DIR} NAME)

# Get the short commit name
execute_process(
        COMMAND git rev-parse --short HEAD
        OUTPUT_VARIABLE GIT_COMMIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

execute_process(
        COMMAND git show -s --format=%cd --date=short HEAD
        OUTPUT_VARIABLE GIT_COMMIT_DATE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

execute_process(
        COMMAND git log -1 --format=%cd --date=format:%H:%M:%S
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_TIME
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

execute_process(
        COMMAND git describe --tags --abbrev=0
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_TAG
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

get_filename_component(IVF_HACK_CXX_COMPILER ${CMAKE_CXX_COMPILER} NAME)

configure_file(${CMAKE_SOURCE_DIR}/include/config.h.in ${CMAKE_SOURCE_DIR}/config.h)
message(STATUS "Config file \"config.h\" generated in ${CMAKE_SOURCE_DIR}")

set(LOGGING_INFO_QUERIED True)

elseif(NOT EXISTS ${CMAKE_SOURCE_DIR}/config.h)
    message(FATAL_ERROR ".git or pre generated config.h is required")
endif()
