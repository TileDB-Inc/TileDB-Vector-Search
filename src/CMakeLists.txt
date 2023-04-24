

add_executable(flat flat.cpp query.h defs.h sift_db.h timer.h sift_array.h)

target_link_libraries(flat PUBLIC docopt)
target_link_libraries(flat PUBLIC flat_tbb)
target_link_libraries(flat PUBLIC flat_blas)
target_link_libraries(flat PUBLIC TileDB::tiledb_shared)

# add_executable(ingest ingest.cpp sift_db.h timer.h)
# target_include_directories(ingest PUBLIC
#     /Users/lums/Contrib/dist/include
# )
# target_link_libraries(ingest PUBLIC docopt TileDB::tiledb_shared)


#
# Some debugging code
#
# Get the compile options for flat
# get_target_property(TargetCompileOptions flat COMPILE_OPTIONS)

# Print the compile options
# message("Compile options for flat:")
# foreach(Option ${TargetCompileOptions})
#     message("  ${Option}")
# endforeach()

message (STATUS ${MKL_THREAD_LIB})
message (STATUS ${INTERFACE_COMPILE_OPTIONS})
message (STATUS ${INTERFACE_INCLUDE_DIRECTORIES})
