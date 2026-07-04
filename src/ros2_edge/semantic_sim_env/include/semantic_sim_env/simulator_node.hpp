#ifndef SIMULATOR_NODE_HPP
#define SIMULATOR_NODE_HPP

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "image_transport/image_transport.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/image.hpp"

#include <algorithm>
#include <cv_bridge/cv_bridge.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <cmath>

// Represents static structural elements in the simulation
struct SimWall {
    std::string name;
    Vector3 position;
    Vector3 size;
    Color color;
};

// Represents interactive or dynamic objects in the simulation
struct SimObject {
    std::string name;
    std::string shape;
    Vector3 position;
    float size;
    Color color;
};

class SimulatorNode : public rclcpp::Node {
public:
    SimulatorNode();
    ~SimulatorNode();

    // Main simulation loop called by the executor
    void update();

private:
    // Rendering helper methods
    void draw_axes();
    void draw_scene(bool is_fpv);

    // Resource and message handling helpers
    Color parse_color(const nlohmann::json& j_color);
    void load_world(const std::string& filepath);
    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void publish_image();
    void publish_telemetry();

    // ROS 2 Communication interfaces
    image_transport::Publisher image_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr vel_sub_;

    // Simulation state and camera control
    Camera3D camera_external_{}, camera_fpv_{};
    bool use_fpv_ = false;
    float ext_cam_yaw_ = -2.356f, ext_cam_pitch_ = -0.615f;

    // Drone kinematics and flight dynamics state
    Vector3 drone_pos_{0.0f, 0.5f, 0.0f};
    float drone_roll_ = 0.0f, drone_pitch_ = 0.0f, drone_yaw_ = 0.0f;
    float cmd_vel_x_ = 0.0f, cmd_vel_y_ = 0.0f, cmd_vel_z_ = 0.0f;
    float cmd_roll_rate_ = 0.0f, cmd_pitch_rate_ = 0.0f, cmd_yaw_rate_ = 0.0f;

    float dt_;
    bool is_headless_;

    // World composition data
    std::vector<SimWall> walls_;
    std::vector<SimObject> objects_;
};

#endif
