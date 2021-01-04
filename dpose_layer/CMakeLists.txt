cmake_minimum_required(VERSION 3.0.2)
project(dpose_layer)

set(CATKIN_PACKAGES costmap_2d geometry_msgs tf2 tf2_geometry_msgs dpose_core pluginlib)
find_package(catkin REQUIRED COMPONENTS ${CATKIN_PACKAGES})

catkin_package(
 INCLUDE_DIRS src
 LIBRARIES ${PROJECT_NAME}
 CATKIN_DEPENDS ${CATKIN_PACKAGES}
 DEPENDS
)

###########
## Build ##
###########

add_library(${PROJECT_NAME} src/dpose_layer.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC src ${catkin_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

#############
## Install ##
#############

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY src/${PROJECT_NAME}/ DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

install(FILES plugin.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
