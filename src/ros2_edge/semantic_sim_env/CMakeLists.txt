cmake_minimum_required(VERSION 3.10)
project(semantic_sim_env)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(image_transport REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(nlohmann_json REQUIRED)

pkg_check_modules(RAYLIB REQUIRED raylib)

add_executable(simulator_node src/simulator_node.cpp)

ament_target_dependencies(simulator_node
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  cv_bridge
  image_transport
  nlohmann_json
)

target_link_libraries(simulator_node
  ${RAYLIB_LIBRARIES}
  ${OpenCV_LIBS}
)

target_include_directories(simulator_node PRIVATE
  ${RAYLIB_INCLUDE_DIRS}
)

install(TARGETS simulator_node DESTINATION lib/${PROJECT_NAME})

ament_package()