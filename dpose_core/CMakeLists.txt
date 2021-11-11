cmake_minimum_required(VERSION 3.6.0)
project(dpose_core)

add_compile_options(-Wpedantic -Wall -Wextra)

# the clang-tidy checks are optional
# set(CMAKE_CXX_CLANG_TIDY clang-tidy -checks=-*,clang-analyzer-cplusplus*,clang-diagnostic-*,clang-analyzer-*)

find_package(catkin REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(
 INCLUDE_DIRS src
 LIBRARIES  ${PROJECT_NAME}
 DEPENDS OpenCV
)

###########
## Build ##
###########

add_library(${PROJECT_NAME} src/dpose_core.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC src )
target_include_directories(${PROJECT_NAME} PUBLIC SYSTEM 
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES})

#############
## Install ##
#############

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY src/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
  find_package(rostest)
  catkin_add_gtest(${PROJECT_NAME}_test
    test/cost_data.cpp
    test/jacobian_data.cpp
    test/pose_gradient.cpp
    )

  target_include_directories(${PROJECT_NAME}_test PUBLIC src)
  if (TARGET ${PROJECT_NAME}_test)
    target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME} gtest_main)
  endif()
endif()

###############
## Benchmark ##
###############

set(${PROJECT_NAME}_BENCHMARK "Enable benchmarking" ON)
if(${PROJECT_NAME}_BENCHMARK)
  find_package(benchmark)
  if(benchmark_FOUND)
    add_executable(${PROJECT_NAME}_perf
      perf/dpose_core.cpp
    )
    target_link_libraries(${PROJECT_NAME}_perf benchmark::benchmark ${PROJECT_NAME})
  else()
    message("benchmark library not found")
  endif()
endif()
