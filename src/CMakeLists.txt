
add_executable(flat flat.cpp)

# Define a target and set some compile options
# target_compile_options(flat PRIVATE -Wall -Wextra -pedantic)

if (NOT APPLE)
    target_compile_options(flat PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
    target_include_directories(flat PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(flat PUBLIC docopt $<LINK_ONLY:MKL::MKL> flat_tbb)
endif()

target_link_libraries(flat PUBLIC docopt)

# Get the compile options for flat
get_target_property(TargetCompileOptions flat COMPILE_OPTIONS)

# Print the compile options
message("Compile options for flat:")
foreach(Option ${TargetCompileOptions})
    message("  ${Option}")
endforeach()

message (STATUS ${MKL_THREAD_LIB})
message (STATUS ${INTERFACE_COMPILE_OPTIONS})
message (STATUS ${INTERFACE_INCLUDE_DIRECTORIES})



# target_link_libraries(flat docopt)
