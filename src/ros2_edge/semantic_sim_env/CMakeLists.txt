cmake_minimum_required(VERSION 3.10)
project(semantic_sim_env)

# Set standard to C++17
  set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable compiler warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(image_transport REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(RAYLIB REQUIRED raylib)

# Define executable with header and source
add_executable(simulator_node 
  src/simulator_node.cpp 
  include/semantic_sim_env/simulator_node.hpp
)

# Set include directory for local headers
target_include_directories(simulator_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${RAYLIB_INCLUDE_DIRS}
)

# Link ROS 2 dependencies
ament_target_dependencies(simulator_node
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  cv_bridge
  image_transport
  nlohmann_json
)

# Link external libraries
target_link_libraries(simulator_node
  ${RAYLIB_LIBRARIES}
  ${OpenCV_LIBS}
)

# Install executable, headers, and resource files
install(TARGETS simulator_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)


ament_package()