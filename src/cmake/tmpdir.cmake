find_program(mktemp NAMES mkstemp mktemp)
execute_process(
  COMMAND ${mktemp}
  OUTPUT_VARIABLE TMPDIR
)

# These must be separate calls because CMake is dumb.
execute_process(
  COMMAND ${CMAKE_COMMAND} -E make_directory ${TMPDIR}
)
execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -f last_tmpdir
)
execute_process(
  COMMAND ${CMAKE_COMMAND} -E create_symlink  ${TMPDIR} last_tmpdir
)

message(STATUS "Running tests in temporary directory: ${TMPDIR}")