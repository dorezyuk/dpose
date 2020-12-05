cmake_minimum_required(VERSION 3.0.2)
project(laces)


set(CATKIN_PACKAGES costmap_2d)
find_package(catkin REQUIRED COMPONENTS ${CATKIN_PACKAGES})
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(
 INCLUDE_DIRS src
 LIBRARIES ${PROJECT_NAME}_ros
 CATKIN_DEPENDS ${CATKIN_PACKAGES}
 DEPENDS OpenCV
)
message(${OpenCV_INCLUDE_DIRS})

# the ros-independent part
add_library(${PROJECT_NAME} src/laces.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC src ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# the ros-specific library
add_library(${PROJECT_NAME}_ros src/laces_ros.cpp)
target_link_libraries(${PROJECT_NAME}_ros ${catkin_LIBRARIES} ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}_ros PUBLIC ${catkin_INCLUDE_DIRS} src)

# test-node
add_executable(${PROJECT_NAME}_node test/node.cpp)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME} ${PROJECT_NAME}_ros)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
    ## Add gtest based cpp test target and link libraries
    catkin_add_gtest(${PROJECT_NAME}_gtest 
        test/draw_polygon.cpp
        test/euclidean_distance_transform.cpp
        test/get_circular_cells.cpp
        test/get_derivatives.cpp)
    if(TARGET ${PROJECT_NAME}_gtest)
        target_link_libraries(${PROJECT_NAME}_gtest ${PROJECT_NAME})
    endif()
endif()


## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
