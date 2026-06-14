cmake_minimum_required(VERSION 3.10)
project(semantic_sim_env)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(RAYLIB REQUIRED raylib)

add_executable(simulator_node src/simulator_node.cpp)

target_link_libraries(simulator_node PRIVATE
  rclcpp::rclcpp
  sensor_msgs::sensor_msgs
  cv_bridge::cv_bridge
  ${RAYLIB_LIBRARIES}
  ${OpenCV_LIBS}
)

target_include_directories(simulator_node PRIVATE ${RAYLIB_INCLUDE_DIRS})

install(TARGETS simulator_node DESTINATION lib/${PROJECT_NAME})

ament_package()
