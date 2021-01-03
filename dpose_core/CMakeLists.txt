cmake_minimum_required(VERSION 3.0.2)
project(dpose_core)

set(CATKIN_PACKAGES costmap_2d geometry_msgs tf2 tf2_geometry_msgs )
find_package(catkin REQUIRED COMPONENTS ${CATKIN_PACKAGES})
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(
 INCLUDE_DIRS src
 LIBRARIES ${PROJECT_NAME}_ros ${PROJECT_NAME}
 CATKIN_DEPENDS ${CATKIN_PACKAGES}
 DEPENDS OpenCV
)

###########
## Build ##
###########

# the ros-independent part
add_library(${PROJECT_NAME} src/dpose.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC src ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# the ros-specific library
add_library(${PROJECT_NAME}_ros src/dpose_ros.cpp)
target_link_libraries(${PROJECT_NAME}_ros ${catkin_LIBRARIES} ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}_ros PUBLIC ${catkin_INCLUDE_DIRS} src)

# test-node
add_executable(${PROJECT_NAME}_node test/node.cpp)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME} ${PROJECT_NAME}_ros)

#############
## Install ##
#############

install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_ros
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
    ## Add gtest based cpp test target and link libraries
    catkin_add_gtest(${PROJECT_NAME}_gtest 
        test/draw_polygon.cpp
        test/get_circular_cells.cpp
        test/get_derivatives.cpp)
    if(TARGET ${PROJECT_NAME}_gtest)
        target_link_libraries(${PROJECT_NAME}_gtest ${PROJECT_NAME})
    endif()
endif()
