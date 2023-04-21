


# Add the external library
# add_subdirectory(external_library)

# Create a target for your project
# add_executable(my_project main.cpp)

# Link the external library to your project target
# target_link_libraries(my_project external_library)


add_executable(flat flat.cpp query.h defs.h sift_db.h timer.h sift_array.h)

# Define a target and set some compile options
# target_compile_options(flat PRIVATE -Wall -Wextra -pedantic)

if (NOT APPLE)
    target_compile_options(flat PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
    target_include_directories(flat PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(flat PUBLIC docopt $<LINK_ONLY:MKL::MKL> flat_tbb)
endif()

# target_link_libraries(flat PUBLIC docopt)
target_link_libraries(flat TileDB::tiledb_shared docopt)

add_executable(ingest ingest.cpp sift_db.h timer.h)
target_include_directories(ingest PUBLIC
    /Users/lums/Contrib/dist/include
)
target_link_libraries(ingest PUBLIC docopt TileDB::tiledb_shared)




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

# message (STATUS ${MKL_THREAD_LIB})
# message (STATUS ${INTERFACE_COMPILE_OPTIONS})
# message (STATUS ${INTERFACE_INCLUDE_DIRECTORIES})
