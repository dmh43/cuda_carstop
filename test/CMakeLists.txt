add_library(gtest ./gtest/gtest-all.cc)
include_directories(.)
target_link_libraries(gtest ${CMAKE_THREAD_LIBS_INIT})

set(test_src
  run_all.cc
  pf_test.cc
)

add_executable(pf_tests ${test_src})

target_link_libraries(pf_tests
  gtest
  pf
  ${CMAKE_THREAD_LIBS_INIT}
  ${MATH_LIBS}
)

add_test(
  NAME tests
  COMMAND pf_tests
)
