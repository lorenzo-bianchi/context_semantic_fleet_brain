#include "raylib.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class SimulatorNode : public rclcpp::Node {
public:
    SimulatorNode() : Node("simulator_node") {
        // Setup ROS2
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", 10);
        vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10, std::bind(&SimulatorNode::cmd_vel_callback, this, std::placeholders::_1));

        // Setup Raylib (InitWindow obbligatorio anche per render texture)
        InitWindow(640, 480, "DroneSim");
        render_texture_ = LoadRenderTexture(640, 480);

        timer_ = this->create_wall_timer(std::chrono::milliseconds(33), std::bind(&SimulatorNode::update, this));
    }

    ~SimulatorNode() {
        UnloadRenderTexture(render_texture_);
        CloseWindow();
    }

private:
    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        drone_pos_.x += (float)msg->linear.x * 0.1f;
        drone_pos_.z += (float)msg->linear.y * 0.1f;
    }

    void update() {
        BeginTextureMode(render_texture_);
        ClearBackground(RAYWHITE);
        BeginMode3D({ {10.0f, 10.0f, 10.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 45.0f, 0 });
            DrawCube(drone_pos_, 1.0f, 1.0f, 1.0f, BLUE); // Il Drone
            DrawCube({5.0f, 0.0f, 5.0f}, 2.0f, 2.0f, 2.0f, RED);   // Ostacolo
            DrawGrid(10, 1.0f);
        EndMode3D();
        EndTextureMode();

        publish_image();
    }

    void publish_image() {
        // Estrai dati dalla texture Raylib e converti in OpenCV
        Image img = LoadImageFromTexture(render_texture_.texture);
        cv::Mat mat(img.height, img.width, CV_8UC4, img.data);
        cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGR);

        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", mat).toImageMsg();
        image_pub_->publish(*msg);
    }

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr vel_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
    RenderTexture2D render_texture_;
    Vector3 drone_pos_ = {0.0f, 0.0f, 0.0f};
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SimulatorNode>());
    rclcpp::shutdown();
    return 0;
}
