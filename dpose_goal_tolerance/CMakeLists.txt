cmake_minimum_required(VERSION 3.6.0)
project(dpose_goal_tolerance)

set(CATKIN_PACKAGES
  angles
  dpose_core
  dynamic_reconfigure
  gpp_interface
  pluginlib
  tf2
  tf2_geometry_msgs)
add_compile_definitions("HAVE_CSTDDEF")
# add_compile_options(-Wpedantic -Wall -Wextra)

# the clang-tidy checks are optional
# set(CMAKE_CXX_CLANG_TIDY clang-tidy -checks=-*,clang-analyzer-cplusplus*,clang-diagnostic-*,clang-analyzer-*)

find_package(catkin REQUIRED COMPONENTS ${CATKIN_PACKAGES})
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)

# find ipopt. this works if you install the library through apt via
# sudo apt install coinor-libipopt-dev
pkg_check_modules(Ipopt REQUIRED ipopt)

generate_dynamic_reconfigure_options(
  cfg/DposeGoalTolerance.cfg
)

catkin_package(
 INCLUDE_DIRS src
 LIBRARIES  ${PROJECT_NAME}
 CATKIN_DEPENDS ${CATKIN_PACKAGES}
)

###########
## Build ##
###########

add_library(${PROJECT_NAME} src/dpose_goal_tolerance.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC src)
target_include_directories(${PROJECT_NAME} PUBLIC SYSTEM
  ${EIGEN3_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${Ipopt_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${Ipopt_LINK_LIBRARIES}
  ${Ipopt_LDFLAGS})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)

#############
## Install ##
#############

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY src/${PROJECT_NAME}/
                  # include/${PROJECT_NAME}
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(FILES plugin.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)

  # ros-tests
  add_rostest_gtest(${PROJECT_NAME}_gtest test/dpose_goal_tolerance.launch test/dpose_goal_tolerance.cpp)
  target_link_libraries(${PROJECT_NAME}_gtest ${PROJECT_NAME})
endif()
