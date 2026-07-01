#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "image_transport/image_transport.hpp"
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>

// Semantic map structures
struct SimWall {
    std::string name;
    Vector3 position;
    Vector3 size;
    Color color;
};

struct SimObject {
    std::string name;
    std::string shape;
    Vector3 position;
    float size;
    Color color;
};

class SimulatorNode : public rclcpp::Node {
public:
    SimulatorNode() : Node("simulator_node") {
        this->declare_parameter<bool>("headless", false);
        this->get_parameter("headless", is_headless_);

        image_pub_ = image_transport::create_publisher(this, "/camera/image_raw");
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);

        vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10, std::bind(&SimulatorNode::cmd_vel_callback, this, std::placeholders::_1));

        // Load world definition from JSON
        load_world("/workspace/world_config.json");

        if (is_headless_) {
            SetConfigFlags(FLAG_WINDOW_HIDDEN);
            RCLCPP_INFO(this->get_logger(), "Starting simulator in HEADLESS mode...");
        }

        InitWindow(640, 480, "Semantic Fleet Brain - Simulator");
        SetTraceLogLevel(LOG_WARNING);

        // External static camera setup
        camera_external_.position = Vector3{ 10.0f, 15.0f, 10.0f };
        camera_external_.target = Vector3{ 0.0f, 0.0f, 0.0f };
        camera_external_.up = Vector3{ 0.0f, 1.0f, 0.0f };
        camera_external_.fovy = 45.0f;
        camera_external_.projection = CAMERA_PERSPECTIVE;

        // FPV (First Person View) camera setup
        camera_fpv_ = camera_external_;
        camera_fpv_.fovy = 90.0f;

        SetTargetFPS(30); 

        RCLCPP_INFO(this->get_logger(), "Simulator started with full 6DOF support");
    }

    ~SimulatorNode() {
        CloseWindow();
    }

    void update() {
        if (IsKeyPressed(KEY_C)) use_fpv_ = !use_fpv_;

        dt_ = GetFrameTime(); 
        drone_roll_  += cmd_roll_rate_ * dt_;
        drone_pitch_ += cmd_pitch_rate_ * dt_;
        drone_yaw_   += cmd_yaw_rate_ * dt_;

        Matrix rot = MatrixIdentity();
        rot = MatrixMultiply(rot, MatrixRotateX(drone_roll_));   
        rot = MatrixMultiply(rot, MatrixRotateZ(drone_pitch_));  
        rot = MatrixMultiply(rot, MatrixRotateY(drone_yaw_));    

        Vector3 forward = Vector3Transform(Vector3{1.0f, 0.0f, 0.0f}, rot);
        Vector3 up      = Vector3Transform(Vector3{0.0f, 1.0f, 0.0f}, rot);
        Vector3 left    = Vector3Transform(Vector3{0.0f, 0.0f, -1.0f}, rot); 

        drone_pos_ = Vector3Add(drone_pos_, Vector3Scale(forward, cmd_vel_x_ * dt_));
        drone_pos_ = Vector3Add(drone_pos_, Vector3Scale(left, cmd_vel_y_ * dt_));
        drone_pos_ = Vector3Add(drone_pos_, Vector3Scale(up, cmd_vel_z_ * dt_));

        camera_fpv_.position = Vector3Add(drone_pos_, Vector3Scale(forward, 0.6f));
        camera_fpv_.target = Vector3Add(camera_fpv_.position, forward);
        camera_fpv_.up = up; 

        if (!use_fpv_) {
            float rot_speed = 1.5f * dt_;
            float move_speed = 8.0f * dt_;
            if (IsKeyDown(KEY_LEFT)) ext_cam_yaw_ -= rot_speed;
            if (IsKeyDown(KEY_RIGHT)) ext_cam_yaw_ += rot_speed;
            if (IsKeyDown(KEY_UP)) ext_cam_pitch_ += rot_speed;
            if (IsKeyDown(KEY_DOWN)) ext_cam_pitch_ -= rot_speed;
            ext_cam_pitch_ = std::max(-1.5f, std::min(1.5f, ext_cam_pitch_));

            Vector3 cam_forward = {cosf(ext_cam_pitch_) * cosf(ext_cam_yaw_), sinf(ext_cam_pitch_), cosf(ext_cam_pitch_) * sinf(ext_cam_yaw_)};
            Vector3 cam_right = { -sinf(ext_cam_yaw_), 0.0f, cosf(ext_cam_yaw_) };

            if (IsKeyDown(KEY_W)) camera_external_.position = Vector3Add(camera_external_.position, Vector3Scale(cam_forward, move_speed));
            if (IsKeyDown(KEY_S)) camera_external_.position = Vector3Subtract(camera_external_.position, Vector3Scale(cam_forward, move_speed));
            if (IsKeyDown(KEY_D)) camera_external_.position = Vector3Add(camera_external_.position, Vector3Scale(cam_right, move_speed));
            if (IsKeyDown(KEY_A)) camera_external_.position = Vector3Subtract(camera_external_.position, Vector3Scale(cam_right, move_speed));
            camera_external_.target = Vector3Add(camera_external_.position, cam_forward);
        }

        BeginDrawing();
            ClearBackground(Color{240, 240, 240, 255}); 
            BeginMode3D(camera_fpv_);
                draw_scene();
            EndMode3D();

            publish_image(); 

            if (!use_fpv_) {
                ClearBackground(Color{240, 240, 240, 255});
                BeginMode3D(camera_external_);
                    draw_scene();
                EndMode3D();
            }

            if (use_fpv_) {
                DrawText("MODE: FPV (Drone Nose)", 10, 10, 20, DARKGREEN);
            } else {
                DrawText("MODE: FREE CAMERA", 10, 10, 20, DARKBLUE);
                DrawText("WASD to Move | ARROWS to Rotate", 10, 35, 10, DARKGRAY);
            }
            DrawText("Press 'C' to toggle camera", 10, 55, 10, DARKGRAY);
            DrawText(TextFormat("Alt: %.2f | XYZ: (%.1f, %.1f, %.1f)", drone_pos_.y, drone_pos_.x, drone_pos_.y, drone_pos_.z), 10, 75, 15, RED);
            DrawText(TextFormat("Walls: %lu | Objects: %lu", walls_.size(), objects_.size()), 10, 95, 15, BLACK);

        EndDrawing();

        publish_telemetry();
    }

private:
    void draw_axes() {
        float len = 2.0f;
        float thickness = 0.05f;
        DrawCylinderEx(Vector3{0.0f, 0.0f, 0.0f}, Vector3{len, 0.0f, 0.0f}, thickness, thickness, 8, RED);
        DrawCylinderEx(Vector3{0.0f, 0.0f, 0.0f}, Vector3{0.0f, len, 0.0f}, thickness, thickness, 8, GREEN);
        DrawCylinderEx(Vector3{0.0f, 0.0f, 0.0f}, Vector3{0.0f, 0.0f, len}, thickness, thickness, 8, BLUE);

        DrawSphere(Vector3{len, 0.0f, 0.0f}, 0.1f, RED);
        DrawSphere(Vector3{0.0f, len, 0.0f}, 0.1f, GREEN);
        DrawSphere(Vector3{0.0f, 0.0f, len}, 0.1f, BLUE);
    }
    void draw_scene() {
        draw_axes();

        // Draw floor plane and grid
        DrawPlane(Vector3{0.0f, -0.01f, 0.0f}, Vector2{50.0f, 50.0f}, Color{160, 160, 160, 255}); 
        DrawGrid(50, 1.0f); 

        // Draw walls
        for (const auto& wall : walls_) {
            DrawCube(wall.position, wall.size.x, wall.size.y, wall.size.z, wall.color);
            DrawCubeWires(wall.position, wall.size.x, wall.size.y, wall.size.z, BLACK);
        }

        // Draw semantic objects
        for (const auto& obj : objects_) {
            if (obj.shape == "cube") {
                DrawCube(obj.position, obj.size, obj.size, obj.size, obj.color);
                DrawCubeWires(obj.position, obj.size, obj.size, obj.size, BLACK);
            } else if (obj.shape == "sphere") {
                DrawSphere(obj.position, obj.size / 2.0f, obj.color);
                DrawSphereWires(obj.position, obj.size / 2.0f, 16, 16, BLACK);
            } else if (obj.shape == "cylinder") {
                DrawCylinder(obj.position, obj.size / 2.0f, obj.size / 2.0f, obj.size, 16, obj.color);
                DrawCylinderWires(obj.position, obj.size / 2.0f, obj.size / 2.0f, obj.size, 16, BLACK);
            } else if (obj.shape == "pyramid") {
                DrawCylinder(obj.position, 0.0f, obj.size / 2.0f, obj.size, 4, obj.color);
                DrawCylinderWires(obj.position, 0.0f, obj.size / 2.0f, obj.size, 4, BLACK);
            } else if (obj.shape == "parallelepiped") {
                DrawCube(obj.position, obj.size * 1.5f, obj.size, obj.size * 0.5f, obj.color);
                DrawCubeWires(obj.position, obj.size * 1.5f, obj.size, obj.size * 0.5f, BLACK);
            }
        }

        // Draw drone with 6DOF orientation
        rlPushMatrix();
            rlTranslatef(drone_pos_.x, drone_pos_.y, drone_pos_.z);
            rlRotatef(drone_yaw_ * RAD2DEG, 0, 1, 0);
            rlRotatef(drone_pitch_ * RAD2DEG, 0, 0, 1);
            rlRotatef(drone_roll_ * RAD2DEG, 1, 0, 0);

            // Drone body
            DrawCube(Vector3{0.0f, 0.0f, 0.0f}, 0.15f, 0.04f, 0.15f, GREEN);
            DrawCubeWires(Vector3{0.0f, 0.0f, 0.0f}, 0.15f, 0.04f, 0.15f, BLACK);

            // Arms configuration
            rlPushMatrix();
                rlRotatef(45.0f, 0, 1, 0);
                DrawCube(Vector3{0.0f, 0.0f, 0.0f}, 0.4f, 0.015f, 0.015f, GRAY);
                DrawCube(Vector3{0.0f, 0.0f, 0.0f}, 0.015f, 0.015f, 0.4f, GRAY);
            rlPopMatrix();

            float arm_d = 0.1414f;
            float prop_y = 0.03f;
            float p_rad = 0.08f;

            // Motors
            DrawCylinder(Vector3{arm_d, 0.01f, -arm_d}, 0.015f, 0.015f, 0.03f, 8, BLACK);
            DrawCylinder(Vector3{arm_d, 0.01f, arm_d}, 0.015f, 0.015f, 0.03f, 8, BLACK);
            DrawCylinder(Vector3{-arm_d, 0.01f, -arm_d}, 0.015f, 0.015f, 0.03f, 8, BLACK);
            DrawCylinder(Vector3{-arm_d, 0.01f, arm_d}, 0.015f, 0.015f, 0.03f, 8, BLACK);

            // Propellers visualization
            DrawCylinder(Vector3{arm_d, prop_y, -arm_d}, p_rad, p_rad, 0.005f, 16, ColorAlpha(RED, 0.7f));
            DrawCylinder(Vector3{arm_d, prop_y, arm_d}, p_rad, p_rad, 0.005f, 16, ColorAlpha(RED, 0.7f));
            DrawCylinder(Vector3{-arm_d, prop_y, -arm_d}, p_rad, p_rad, 0.005f, 16, ColorAlpha(DARKGRAY, 0.7f));
            DrawCylinder(Vector3{-arm_d, prop_y, arm_d}, p_rad, p_rad, 0.005f, 16, ColorAlpha(DARKGRAY, 0.7f));

            // FPV Camera lens
            DrawCube(Vector3{0.08f, -0.01f, 0.0f}, 0.04f, 0.03f, 0.03f, BLACK);
            DrawSphere(Vector3{0.10f, -0.01f, 0.0f}, 0.012f, BLUE);
        rlPopMatrix();
    }

    Color parse_color(const nlohmann::json& j_color) {
        return Color{
            j_color[0].get<unsigned char>(),
            j_color[1].get<unsigned char>(),
            j_color[2].get<unsigned char>(),
            j_color[3].get<unsigned char>()
        };
    }

    void load_world(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            RCLCPP_WARN(this->get_logger(), "Could not open %s. Starting with empty world.", filepath.c_str());
            return;
        }

        nlohmann::json j;
        try {
            file >> j;
        } catch (const nlohmann::json::parse_error& e) {
            RCLCPP_ERROR(this->get_logger(), "JSON parsing error: %s", e.what());
            return;
        }

        if (j.contains("walls")) {
            for (const auto& w : j["walls"]) {
                SimWall wall;
                wall.name = w["name"];
                float forced_height = 3.0f;
                float center_y = forced_height / 2.0f;
                wall.size = Vector3{w["width"].get<float>(), forced_height, w["depth"].get<float>()};
                wall.position = Vector3{w["x"].get<float>(), center_y, w["z"].get<float>()};
                wall.color = parse_color(w["color"]);
                walls_.push_back(wall);
            }
        }

        if (j.contains("objects")) {
            for (const auto& o : j["objects"]) {
                SimObject obj;
                obj.name = o["name"];
                obj.shape = o["shape"];
                obj.size = o["size"];
                obj.position = Vector3{o["x"].get<float>(), o["y"].get<float>(), o["z"].get<float>()};
                obj.color = parse_color(o["color"]);
                objects_.push_back(obj);
            }
        }

        RCLCPP_INFO(this->get_logger(), "World loaded: %lu walls, %lu objects.", walls_.size(), objects_.size());
    }

    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        cmd_vel_x_ = msg->linear.x;
        cmd_vel_y_ = msg->linear.y;
        cmd_vel_z_ = msg->linear.z;
        cmd_roll_rate_ = msg->angular.x;
        cmd_pitch_rate_ = msg->angular.y;
        cmd_yaw_rate_ = msg->angular.z;
    }

    void publish_image() {
        Image img = LoadImageFromScreen();

        if (img.data == nullptr) return;

        cv::Mat mat(img.height, img.width, CV_8UC4, img.data);
        cv::Mat mat_bgr;
        cv::cvtColor(mat, mat_bgr, cv::COLOR_RGBA2BGR);

        std_msgs::msg::Header header;
        header.stamp = this->now();
        header.frame_id = "camera_fpv_link";

        auto msg = cv_bridge::CvImage(header, "bgr8", mat_bgr).toImageMsg();
        image_pub_.publish(*msg);

        UnloadImage(img); 
    }

    void publish_telemetry() {
        auto odom_msg = nav_msgs::msg::Odometry();
        odom_msg.header.stamp = this->now();
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id = "base_link";

        odom_msg.pose.pose.position.x = drone_pos_.x;
        odom_msg.pose.pose.position.y = -drone_pos_.z; 
        odom_msg.pose.pose.position.z = drone_pos_.y;

        double r = drone_roll_, p = -drone_pitch_, y = drone_yaw_;
        double cr = cos(r * 0.5), sr = sin(r * 0.5);
        double cp = cos(p * 0.5), sp = sin(p * 0.5);
        double cy = cos(y * 0.5), sy = sin(y * 0.5);

        odom_msg.pose.pose.orientation.w = cr * cp * cy + sr * sp * sy;
        odom_msg.pose.pose.orientation.x = sr * cp * cy - cr * sp * sy;
        odom_msg.pose.pose.orientation.y = cr * sp * cy + sr * sp * sy;
        odom_msg.pose.pose.orientation.z = cr * cp * sy - sr * sp * cy;

        odom_msg.twist.twist.linear.x = cmd_vel_x_;
        odom_msg.twist.twist.linear.y = cmd_vel_y_;
        odom_msg.twist.twist.linear.z = cmd_vel_z_;
        odom_msg.twist.twist.angular.x = cmd_roll_rate_;
        odom_msg.twist.twist.angular.y = cmd_pitch_rate_;
        odom_msg.twist.twist.angular.z = cmd_yaw_rate_;

        odom_pub_->publish(odom_msg);
    }

    image_transport::Publisher image_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr vel_sub_;

    Camera3D camera_external_{};
    Camera3D camera_fpv_{};
    bool use_fpv_ = false;

    float ext_cam_yaw_ = -2.356f;
    float ext_cam_pitch_ = -0.615f;

    Vector3 drone_pos_{0.0f, 0.5f, 0.0f}; 
    float drone_roll_ = 0.0f, drone_pitch_ = 0.0f, drone_yaw_ = 0.0f;
    float cmd_vel_x_ = 0.0f, cmd_vel_y_ = 0.0f, cmd_vel_z_ = 0.0f;
    float cmd_roll_rate_ = 0.0f, cmd_pitch_rate_ = 0.0f, cmd_yaw_rate_ = 0.0f;

    float dt_;
    bool is_headless_;

    std::vector<SimWall> walls_;
    std::vector<SimObject> objects_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SimulatorNode>();

    while (rclcpp::ok() && !WindowShouldClose()) {
        rclcpp::spin_some(node);
        node->update();
    }

    rclcpp::shutdown();
    return 0;
}