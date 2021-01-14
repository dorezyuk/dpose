cmake_minimum_required(VERSION 3.6.0)
project(dpose_goal_tolerance)

set(CATKIN_PACKAGES dpose_core gpp_interface pluginlib xmlrpcpp)
# add_compile_options(-Wpedantic -Wall -Wextra)

# the clang-tidy checks are optional
# set(CMAKE_CXX_CLANG_TIDY clang-tidy -checks=-*,clang-analyzer-cplusplus*,clang-diagnostic-*,clang-analyzer-*)

find_package(catkin REQUIRED COMPONENTS ${CATKIN_PACKAGES})

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
target_include_directories(${PROJECT_NAME} PUBLIC SYSTEM ${catkin_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

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

install(FILES plugin.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
