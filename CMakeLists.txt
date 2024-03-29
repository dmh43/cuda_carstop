cmake_minimum_required(VERSION 3.0)
project(pf)

# Set build type to RELEASE by default:
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type provided, default to RELEASE.")
  set(CMAKE_BUILD_TYPE "RELEASE")
endif ()

option(BuildTest "BuildTest" ON)
option(BuildShared "BuildShared" OFF)
option(Sanitize "Sanitize" OFF)

if (BuildShared)
  set(LIB_TYPE SHARED)
else ()
  set(LIB_TYPE STATIC)
endif ()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
#find_package(Boost)
find_package(Math)

include_directories(. include ${Boost_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-std=c++11 HAVE_FLAG_CXX_11)
  if (HAVE_FLAG_CXX_11)
    set(CMAKE_CXX_FLAGS "-Wall -std=c++11 ${CMAKE_CXX_FLAGS}")
  else ()
    message(FATAL_ERROR "Gorram! Your compiler is too frakking old. C++11 support is required.")
  endif ()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -pedantic -fno-omit-frame-pointer -fno-inline-functions")
endif ()

if (Sanitize)
  set(CMAKE_CXX_FLAGS "-fsanitize=address -fsanitize=undefined ${CMAKE_CXX_FLAGS}")
endif ()

add_subdirectory(src)

if (BuildTest)
  enable_testing()
  add_subdirectory(test)
endif ()

message(STATUS "")
message(STATUS "BUILD SUMMARY")
message(STATUS "  CMAKE_GENERATOR      : ${CMAKE_GENERATOR}")
message(STATUS "  Compiler ID          : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Build type           : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Library type         : ${LIB_TYPE}")
message(STATUS "  Build tests          : ${BuildTest}")
message(STATUS "  Sanitize flags       : ${Sanitize}")
message(STATUS "  FoundCUDA            : ${CUDA_FOUND}")
message(STATUS "  Boost include dirs   : ${Boost_INCLUDE_DIRS}")
message(STATUS "  CXX_FLAGS            : ${CMAKE_CXX_FLAGS}")
if ("${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
  message(STATUS "  CXX_FLAGS_RELEASE    : ${CMAKE_CXX_FLAGS_RELEASE}")
elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
  message(STATUS "  CXX_FLAGS_DEBUG      : ${CMAKE_CXX_FLAGS_DEBUG}")
endif ()
message(STATUS "")
