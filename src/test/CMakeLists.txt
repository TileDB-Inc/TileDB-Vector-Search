
include(Catch2)

add_executable(unit_sift_array unit_sift_array.cpp)
target_link_libraries(unit_sift_array PRIVATE Catch2::Catch2WithMain)

add_executable(unit_sift_db unit_sift_db.cpp)
target_link_libraries(unit_sift_db PRIVATE Catch2::Catch2WithMain)

add_executable(unit_defs unit_defs.cpp)
target_link_libraries(unit_defs PRIVATE Catch2::Catch2WithMain)

add_executable(unit_gemm unit_gemm.cpp)
target_link_libraries(unit_gemm PRIVATE Catch2::Catch2WithMain)

set (SIFT_DIR ${CMAKE_SOURCE_DIR}/external/data/siftsmall)

add_custom_command(
    TARGET unit_sift_db POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${SIFT_DIR}
        "$<TARGET_PROPERTY:unit_sift_db,BINARY_DIR>/siftsmall"
    )
add_custom_target(copy_data_files DEPENDS ${EXECUTABLE})
