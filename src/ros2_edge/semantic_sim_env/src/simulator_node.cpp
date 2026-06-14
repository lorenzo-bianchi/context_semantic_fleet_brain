#include "raylib.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>

class SimulatorNode : public rclcpp::Node {
public:
    SimulatorNode() : Node("simulator_node") {
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", 10);
        vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10, std::bind(&SimulatorNode::cmd_vel_callback, this, std::placeholders::_1));

        InitWindow(640, 480, "Semantic Fleet Brain - Simulator");
        render_texture_ = LoadRenderTexture(640, 480);

        // Initialize External (Free) Camera
        camera_external_.position = (Vector3){ 10.0f, 10.0f, 10.0f };
        camera_external_.target = (Vector3){ 0.0f, 0.0f, 0.0f };
        camera_external_.up = (Vector3){ 0.0f, 1.0f, 0.0f };
        camera_external_.fovy = 45.0f;
        camera_external_.projection = CAMERA_PERSPECTIVE;

        // Initialize FPV Camera (Drone view)
        camera_fpv_ = camera_external_;
        camera_fpv_.fovy = 90.0f;

        dt_ = 0.033f;
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33),
            std::bind(&SimulatorNode::update, this));
    }

    ~SimulatorNode() {
        UnloadRenderTexture(render_texture_);
        CloseWindow();
    }

private:
    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        cmd_vel_x_ = msg->linear.x;
        cmd_vel_y_ = msg->linear.y;
        cmd_yaw_rate_ = msg->angular.z;
    }

    void update() {
        if (IsKeyPressed(KEY_C)) {
            use_fpv_ = !use_fpv_;
        }

        // Drone kinematics update
        drone_yaw_ += cmd_yaw_rate_ * dt_;
        float dx = (cmd_vel_x_ * cos(drone_yaw_) - cmd_vel_y_ * sin(drone_yaw_)) * dt_;
        float dz = -(cmd_vel_x_ * sin(drone_yaw_) + cmd_vel_y_ * cos(drone_yaw_)) * dt_;

        drone_pos_.x += dx;
        drone_pos_.z += dz;

        // Compute drone nose position for FPV camera anchor
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
            UpdateCamera(&camera_external_, CAMERA_FREE);
        }

        Camera3D active_camera = use_fpv_ ? camera_fpv_ : camera_external_;

        // Render to texture for ROS publishing
        BeginTextureMode(render_texture_);
        ClearBackground(RAYWHITE);
        BeginMode3D(active_camera);

            DrawGrid(20, 1.0f);
            DrawCube({5.0f, 0.0f, 5.0f}, 2.0f, 2.0f, 2.0f, RED);

            DrawCube(drone_pos_, 1.0f, 1.0f, 1.0f, BLUE);
            DrawCube(nose_pos, 0.4f, 0.4f, 0.4f, YELLOW);

        EndMode3D();
        EndTextureMode();

        // Render to physical window for local developer feedback
        BeginDrawing();
            ClearBackground(BLACK);
            DrawTextureRec(render_texture_.texture,
                (Rectangle){ 0, 0, (float)render_texture_.texture.width, (float)-render_texture_.texture.height },
                (Vector2){ 0, 0 }, WHITE);

            if (use_fpv_) {
                DrawText("MODE: FPV (Drone Nose)", 10, 10, 20, DARKGREEN);
            } else {
                DrawText("MODE: FREE CAMERA (Use W/A/S/D and Mouse)", 10, 10, 20, DARKBLUE);
            }
            DrawText("Press 'C' to toggle camera", 10, 40, 10, DARKGRAY);
        EndDrawing();

        publish_image();
    }

    void publish_image() {
        Image img = LoadImageFromTexture(render_texture_.texture);

        // Align OpenGL texture orientation with OpenCV
        ImageFlipVertical(&img); 

        cv::Mat mat(img.height, img.width, CV_8UC4, img.data);
        cv::Mat mat_bgr;
        cv::cvtColor(mat, mat_bgr, cv::COLOR_RGBA2BGR);

        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", mat_bgr).toImageMsg();
        image_pub_->publish(*msg);

        // Prevent memory leak
        UnloadImage(img);
    }

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr vel_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    RenderTexture2D render_texture_;
    Camera3D camera_external_ = { 0 };
    Camera3D camera_fpv_ = { 0 };
    bool use_fpv_ = false;

    Vector3 drone_pos_ = {0.0f, 0.5f, 0.0f};
    float drone_yaw_ = 0.0f;
    float cmd_vel_x_ = 0.0f;
    float cmd_vel_y_ = 0.0f;
    float cmd_yaw_rate_ = 0.0f;
    float dt_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SimulatorNode>());
    rclcpp::shutdown();
    return 0;
}
