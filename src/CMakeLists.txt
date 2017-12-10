set(pf_src
  pf.cu
)

cuda_add_library(pf ${LIB_TYPE} ${pf_src})

target_link_libraries(pf
  ${CUDA_CUBLAS_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
  ${MATH_LIBS}
)

install(TARGETS pf LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY ../include/pf DESTINATION include)
