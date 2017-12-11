add_library(gtest ./gtest/gtest-all.cc)
include_directories(.)
target_link_libraries(gtest ${CMAKE_THREAD_LIBS_INIT})

set(test_src
  run_all.cc
  test_helper.cu
  pf_test.cc
)

cuda_add_executable(pf_tests ${test_src})

target_link_libraries(pf_tests
  gtest
  pf
  ${CUDA_CUBLAS_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
  ${MATH_LIBS}
)

add_test(
  NAME tests
  COMMAND pf_tests
)
