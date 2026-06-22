#include "raylib.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "image_transport/image_transport.hpp"
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>

class SimulatorNode : public rclcpp::Node {
public:
    SimulatorNode() : Node("simulator_node") {
        this->declare_parameter<bool>("headless", false);
        this->get_parameter("headless", is_headless_);

        image_pub_ = image_transport::create_publisher(this, "/camera/image_raw");
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);

        vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10, std::bind(&SimulatorNode::cmd_vel_callback, this, std::placeholders::_1));

        if (is_headless_) {
            SetConfigFlags(FLAG_WINDOW_HIDDEN);
            RCLCPP_INFO(this->get_logger(), "Starting simulator in HEADLESS mode...");
        }

        InitWindow(640, 480, "Semantic Fleet Brain - Simulator");
        
        camera_external_.position = Vector3{ 10.0f, 10.0f, 10.0f };
        camera_external_.target = Vector3{ 0.0f, 0.0f, 0.0f };
        camera_external_.up = Vector3{ 0.0f, 1.0f, 0.0f };
        camera_external_.fovy = 45.0f;
        camera_external_.projection = CAMERA_PERSPECTIVE;

        camera_fpv_ = camera_external_;
        camera_fpv_.fovy = 90.0f;

        SetTargetFPS(30); 

        RCLCPP_INFO(this->get_logger(), "Simulator started");
    }

    ~SimulatorNode() {
        CloseWindow();
    }

    void update() {
        if (IsKeyPressed(KEY_C)) {
            use_fpv_ = !use_fpv_;
        }

        dt_ = GetFrameTime(); 

        drone_yaw_ += cmd_yaw_rate_ * dt_;
        float dx = (cmd_vel_x_ * cos(drone_yaw_) - cmd_vel_y_ * sin(drone_yaw_)) * dt_;
        float dz = -(cmd_vel_x_ * sin(drone_yaw_) + cmd_vel_y_ * cos(drone_yaw_)) * dt_;

        drone_pos_.x += dx;
        drone_pos_.z += dz;

        Vector3 nose_pos = {
            drone_pos_.x + 0.6f * cos(drone_yaw_),
            drone_pos_.y,
            drone_pos_.z - 0.6f * sin(drone_yaw_)
        };

        if (use_fpv_) {
            camera_fpv_.position = nose_pos;
            camera_fpv_.target = {
                nose_pos.x + cos(drone_yaw_),
                nose_pos.y,
                nose_pos.z - sin(drone_yaw_)
            };
        } else {
            float rot_speed = 1.5f * dt_;
            float move_speed = 8.0f * dt_;

            if (IsKeyDown(KEY_LEFT)) ext_cam_yaw_ -= rot_speed;
            if (IsKeyDown(KEY_RIGHT)) ext_cam_yaw_ += rot_speed;
            if (IsKeyDown(KEY_UP)) ext_cam_pitch_ += rot_speed;
            if (IsKeyDown(KEY_DOWN)) ext_cam_pitch_ -= rot_speed;

            ext_cam_pitch_ = std::max(-1.5f, std::min(1.5f, ext_cam_pitch_));

            Vector3 forward = {
                cosf(ext_cam_pitch_) * cosf(ext_cam_yaw_),
                sinf(ext_cam_pitch_),
                cosf(ext_cam_pitch_) * sinf(ext_cam_yaw_)
            };

            Vector3 right = {
                -sinf(ext_cam_yaw_),
                0.0f,
                cosf(ext_cam_yaw_)
            };

            if (IsKeyDown(KEY_W)) {
                camera_external_.position.x += forward.x * move_speed;
                camera_external_.position.y += forward.y * move_speed;
                camera_external_.position.z += forward.z * move_speed;
            }
            if (IsKeyDown(KEY_S)) {
                camera_external_.position.x -= forward.x * move_speed;
                camera_external_.position.y -= forward.y * move_speed;
                camera_external_.position.z -= forward.z * move_speed;
            }
            if (IsKeyDown(KEY_D)) {
                camera_external_.position.x += right.x * move_speed;
                camera_external_.position.y += right.y * move_speed;
                camera_external_.position.z += right.z * move_speed;
            }
            if (IsKeyDown(KEY_A)) {
                camera_external_.position.x -= right.x * move_speed;
                camera_external_.position.y -= right.y * move_speed;
                camera_external_.position.z -= right.z * move_speed;
            }

            camera_external_.target.x = camera_external_.position.x + forward.x;
            camera_external_.target.y = camera_external_.position.y + forward.y;
            camera_external_.target.z = camera_external_.position.z + forward.z;
        }

        Camera3D active_camera = use_fpv_ ? camera_fpv_ : camera_external_;

        BeginDrawing();
            ClearBackground(SKYBLUE); 
            
            BeginMode3D(active_camera);
                DrawGrid(20, 1.0f);
                DrawCube(Vector3{5.0f, 0.0f, 5.0f}, 2.0f, 2.0f, 2.0f, RED); 
                DrawCube(drone_pos_, 1.0f, 1.0f, 1.0f, BLUE); 
                DrawCube(nose_pos, 0.4f, 0.4f, 0.4f, YELLOW); 
            EndMode3D();

            if (use_fpv_) {
                DrawText("MODE: FPV (Drone Nose)", 10, 10, 20, DARKGREEN);
            } else {
                DrawText("MODE: FREE CAMERA", 10, 10, 20, DARKBLUE);
                DrawText("WASD to Move | ARROWS to Rotate", 10, 35, 10, DARKGRAY);
            }
            
            DrawText("Press 'C' to toggle camera", 10, 55, 10, DARKGRAY);
            DrawText(TextFormat("X: %.2f | Z: %.2f | Yaw: %.2f", drone_pos_.x, drone_pos_.z, drone_yaw_), 10, 75, 15, RED);
        EndDrawing();

        publish_image();
        publish_telemetry();
    }

private:
    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        cmd_vel_x_ = msg->linear.x;
        cmd_vel_y_ = msg->linear.y;
        cmd_yaw_rate_ = msg->angular.z;
    }

    void publish_image() {
        Image img = LoadImageFromScreen();

        cv::Mat mat(img.height, img.width, CV_8UC4, img.data);
        cv::Mat mat_bgr;
        cv::cvtColor(mat, mat_bgr, cv::COLOR_RGBA2BGR);

        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", mat_bgr).toImageMsg();
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

        double half_yaw = -drone_yaw_ * 0.5;
        odom_msg.pose.pose.orientation.x = 0.0;
        odom_msg.pose.pose.orientation.y = 0.0;
        odom_msg.pose.pose.orientation.z = std::sin(half_yaw);
        odom_msg.pose.pose.orientation.w = std::cos(half_yaw);

        odom_msg.twist.twist.linear.x = cmd_vel_x_;
        odom_msg.twist.twist.linear.y = cmd_vel_y_;
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
    float drone_yaw_ = 0.0f;
    float cmd_vel_x_ = 0.0f;
    float cmd_vel_y_ = 0.0f;
    float cmd_yaw_rate_ = 0.0f;
    float dt_;
    bool is_headless_;
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