
include(Catch2)
add_executable(unit_sift_db unit_sift_db.cpp)
target_link_libraries(unit_sift_db PRIVATE Catch2::Catch2WithMain)

set (SIFT_DIR ${CMAKE_SOURCE_DIR}/../data/siftsmall)

add_custom_command(
    TARGET unit_sift_db POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${SIFT_DIR}
        "$<TARGET_PROPERTY:unit_sift_db,BINARY_DIR>/siftsmall"
    )
add_custom_target(copy_data_files DEPENDS ${EXECUTABLE})
