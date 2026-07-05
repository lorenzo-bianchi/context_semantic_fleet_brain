#include "semantic_sim_env/simulator_node.hpp"

SimulatorNode::SimulatorNode() : Node("simulator_node") {
    // Parameters
    this->declare_parameter<float>("fov", 0.0f);
    this->declare_parameter<bool>("headless", false);
    this->declare_parameter<int>("image_height", 0);
    this->declare_parameter<int>("image_width", 0);
    this->declare_parameter<float>("init_drone_x", 0.0f);
    this->declare_parameter<float>("init_drone_y", 0.0f);
    this->declare_parameter<float>("init_drone_z", 0.0f);
    this->declare_parameter<float>("speed_move", 0.0f);
    this->declare_parameter<float>("speed_rot", 0.0f);
    this->declare_parameter<int>("target_fps", 0);
    this->declare_parameter<std::string>("world_config_path", "");

    this->get_parameter("fov", fov);
    this->get_parameter("headless", is_headless_);
    this->get_parameter("image_height", image_height_);
    this->get_parameter("image_width", image_width_);
    this->get_parameter("init_drone_x", init_drone_x);
    this->get_parameter("init_drone_y", init_drone_y);
    this->get_parameter("init_drone_z", init_drone_z);
    this->get_parameter("speed_move", speed_move_);
    this->get_parameter("speed_rot", speed_rot_);
    this->get_parameter("target_fps", target_fps);
    this->get_parameter("world_config_path", world_config_path_);

    RCLCPP_INFO(this->get_logger(), "--- Simulator Parameters ---");
    RCLCPP_INFO(this->get_logger(), "Camera FOV: %.1f deg", fov);
    RCLCPP_INFO(this->get_logger(), "Drone Init Pos: [%.1f, %.1f, %.1f]", init_drone_x, init_drone_y, init_drone_z);
    RCLCPP_INFO(this->get_logger(), "Headless Mode: %s", is_headless_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "Move Speed: %.2f", speed_move_);
    RCLCPP_INFO(this->get_logger(), "Resolution: %dx%d @ %d FPS", image_width_, image_height_, target_fps);
    RCLCPP_INFO(this->get_logger(), "Rot Speed: %.2f", speed_rot_);
    RCLCPP_INFO(this->get_logger(), "World Config Path: %s", world_config_path_.c_str());
    RCLCPP_INFO(this->get_logger(), "----------------------------");

    load_world(world_config_path_);

    // Publishers
    image_pub_ = image_transport::create_publisher(this, "/camera/image_raw");
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);

    // Subscribers
    vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", 10, std::bind(&SimulatorNode::cmd_vel_callback, this, std::placeholders::_1));

    if (is_headless_) {
        SetConfigFlags(FLAG_WINDOW_HIDDEN);
        RCLCPP_INFO(this->get_logger(), "Starting simulator in HEADLESS mode...");
    }

    InitWindow(image_width_, image_height_, "Semantic Fleet Brain - Simulator");
    SetTraceLogLevel(LOG_WARNING);

    // Initial camera configurations for external and First-Person View (FPV)
    camera_external_.position = Vector3{ 10.0f, 15.0f, 10.0f };
    camera_external_.target = Vector3{ 0.0f, 0.0f, 0.0f };
    camera_external_.up = Vector3{ 0.0f, 1.0f, 0.0f };
    camera_external_.fovy = 45.0f;
    camera_external_.projection = CAMERA_PERSPECTIVE;

    camera_fpv_ = camera_external_;
    camera_fpv_.fovy = 90.0f;

    SetTargetFPS(30); 

    RCLCPP_INFO(this->get_logger(), "Simulator started with full 6DOF support");
}

SimulatorNode::~SimulatorNode() {
    CloseWindow();
}

void SimulatorNode::update() {
    if (IsKeyPressed(KEY_C)) use_fpv_ = !use_fpv_;

    dt_ = GetFrameTime(); 
    // Integrate angular rates to update drone attitude
    drone_roll_  += cmd_roll_rate_ * dt_;
    drone_pitch_ += cmd_pitch_rate_ * dt_;
    drone_yaw_   += cmd_yaw_rate_ * dt_;

    float current_move_speed = speed_move_ * dt_;
    float current_rot_speed = speed_rot_ * dt_;

    if (use_fpv_) {
        if (IsKeyDown(KEY_LEFT)) drone_yaw_ += current_rot_speed;
        if (IsKeyDown(KEY_RIGHT)) drone_yaw_ -= current_rot_speed;
    }

    // Compute rotation matrix based on drone orientation
    Matrix rot = MatrixIdentity();
    rot = MatrixMultiply(rot, MatrixRotateX(drone_roll_));   
    rot = MatrixMultiply(rot, MatrixRotateZ(drone_pitch_));  
    rot = MatrixMultiply(rot, MatrixRotateY(drone_yaw_));    

    Vector3 forward = Vector3Transform(Vector3{1.0f, 0.0f, 0.0f}, rot);
    Vector3 up      = Vector3Transform(Vector3{0.0f, 1.0f, 0.0f}, rot);
    Vector3 left    = Vector3Transform(Vector3{0.0f, 0.0f, -1.0f}, rot); 

    // Update drone world position based on velocities
    drone_pos_ = Vector3Add(drone_pos_, Vector3Scale(forward, cmd_vel_x_ * dt_));
    drone_pos_ = Vector3Add(drone_pos_, Vector3Scale(left, cmd_vel_y_ * dt_));
    drone_pos_ = Vector3Add(drone_pos_, Vector3Scale(up, cmd_vel_z_ * dt_));

    if (use_fpv_) {
        // Manual control override for drone position
        if (IsKeyDown(KEY_W)) drone_pos_ = Vector3Add(drone_pos_, Vector3Scale(forward, current_move_speed));
        if (IsKeyDown(KEY_S)) drone_pos_ = Vector3Subtract(drone_pos_, Vector3Scale(forward, current_move_speed));
        if (IsKeyDown(KEY_A)) drone_pos_ = Vector3Add(drone_pos_, Vector3Scale(left, current_move_speed));
        if (IsKeyDown(KEY_D)) drone_pos_ = Vector3Subtract(drone_pos_, Vector3Scale(left, current_move_speed));

        if (IsKeyDown(KEY_UP)) drone_pos_.y += current_move_speed;
        if (IsKeyDown(KEY_DOWN)) drone_pos_.y -= current_move_speed;
    }

    // Follow drone with FPV camera
    camera_fpv_.position = Vector3Add(drone_pos_, Vector3Scale(forward, 0.6f));
    camera_fpv_.target = Vector3Add(camera_fpv_.position, forward);
    camera_fpv_.up = up; 

    if (!use_fpv_) {
        // Free camera control logic
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
            draw_scene(true);
        EndMode3D();

        publish_image();

        if (!use_fpv_) {
            ClearBackground(Color{240, 240, 240, 255});
            BeginMode3D(camera_external_);
                draw_scene(false);
            EndMode3D();
        }

        // Render HUD text overlays
        if (use_fpv_) {
            DrawText("MODE: FPV", 10, 10, 20, DARKGREEN);
            DrawText("WASD to Move | ARROWS to Rotate/Alt", 10, 35, 10, DARKGRAY);
        } else {
            DrawText("MODE: FREE CAMERA", 10, 10, 20, DARKBLUE);
            DrawText("WASD to Move | ARROWS to Rotate", 10, 35, 10, DARKGRAY);
        }
        DrawText("Press 'C' to toggle camera", 10, 55, 10, DARKGRAY);

        float yaw_deg = fmod(drone_yaw_ * RAD2DEG + 180.0f, 360.0f);
        if (yaw_deg < 0) yaw_deg += 360.0f;
        yaw_deg -= 180.0f;
        DrawText(TextFormat("Alt(Z): %.2f | X: %.1f | Y: %.1f | Yaw: %.1f deg", drone_pos_.y, drone_pos_.x, -drone_pos_.z, yaw_deg), 10, 75, 15, RED);

    EndDrawing();

    publish_telemetry();
}

void SimulatorNode::draw_axes() {
    float len = 2.0f;
    float thickness = 0.05f;
    DrawCylinderEx(Vector3{0.0f, 0.0f, 0.0f}, Vector3{len, 0.0f, 0.0f}, thickness, thickness, 8, RED);
    DrawCylinderEx(Vector3{0.0f, 0.0f, 0.0f}, Vector3{0.0f, 0.0f, -len}, thickness, thickness, 8, GREEN);
    DrawCylinderEx(Vector3{0.0f, 0.0f, 0.0f}, Vector3{0.0f, len, 0.0f}, thickness, thickness, 8, BLUE);

    DrawSphere(Vector3{len, 0.0f, 0.0f}, 0.1f, RED);
    DrawSphere(Vector3{0.0f, 0.0f, -len}, 0.1f, GREEN);
    DrawSphere(Vector3{0.0f, len, 0.0f}, 0.1f, BLUE);
}

void SimulatorNode::draw_scene(bool is_fpv) {
    if (!is_fpv) {
        draw_axes();
    }

    DrawPlane(Vector3{0.0f, -0.01f, 0.0f}, Vector2{50.0f, 50.0f}, Color{160, 160, 160, 255}); 
    DrawGrid(50, 1.0f); 

    for (const auto& wall : walls_) {
        DrawCube(wall.position, wall.size.x, wall.size.y, wall.size.z, wall.color);
        DrawCubeWires(wall.position, wall.size.x, wall.size.y, wall.size.z, BLACK);
    }

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

    // Render drone model at current state
    rlPushMatrix();
        rlTranslatef(drone_pos_.x, drone_pos_.y, drone_pos_.z);
        rlRotatef(  drone_yaw_ * RAD2DEG, 0, 1, 0);
        rlRotatef(drone_pitch_ * RAD2DEG, 0, 0, 1);
        rlRotatef( drone_roll_ * RAD2DEG, 1, 0, 0);

        DrawCube(Vector3{0.0f, 0.0f, 0.0f}, 0.15f, 0.04f, 0.15f, GREEN);
        DrawCubeWires(Vector3{0.0f, 0.0f, 0.0f}, 0.15f, 0.04f, 0.15f, BLACK);

        rlPushMatrix();
            rlRotatef(45.0f, 0, 1, 0);
            DrawCube(Vector3{0.0f, 0.0f, 0.0f}, 0.4f, 0.015f, 0.015f, GRAY);
            DrawCube(Vector3{0.0f, 0.0f, 0.0f}, 0.015f, 0.015f, 0.4f, GRAY);
        rlPopMatrix();

        DrawCylinder(Vector3{ arm_d, 0.01f, -arm_d}, 0.015f, 0.015f, 0.03f, 8, BLACK);
        DrawCylinder(Vector3{ arm_d, 0.01f,  arm_d}, 0.015f, 0.015f, 0.03f, 8, BLACK);
        DrawCylinder(Vector3{-arm_d, 0.01f, -arm_d}, 0.015f, 0.015f, 0.03f, 8, BLACK);
        DrawCylinder(Vector3{-arm_d, 0.01f,  arm_d}, 0.015f, 0.015f, 0.03f, 8, BLACK);

        DrawCylinder(Vector3{ arm_d, prop_y, -arm_d}, p_rad, p_rad, 0.005f, 16, ColorAlpha(RED, 0.7f));
        DrawCylinder(Vector3{ arm_d, prop_y,  arm_d}, p_rad, p_rad, 0.005f, 16, ColorAlpha(RED, 0.7f));
        DrawCylinder(Vector3{-arm_d, prop_y, -arm_d}, p_rad, p_rad, 0.005f, 16, ColorAlpha(DARKGRAY, 0.7f));
        DrawCylinder(Vector3{-arm_d, prop_y,  arm_d}, p_rad, p_rad, 0.005f, 16, ColorAlpha(DARKGRAY, 0.7f));

        DrawCube(Vector3{0.08f, -0.01f, 0.0f}, 0.04f, 0.03f, 0.03f, BLACK);
        DrawSphere(Vector3{0.10f, -0.01f, 0.0f}, 0.012f, BLUE);
    rlPopMatrix();
}

Color SimulatorNode::parse_color(const nlohmann::json& j_color) {
    return Color{
        j_color[0].get<unsigned char>(),
        j_color[1].get<unsigned char>(),
        j_color[2].get<unsigned char>(),
        j_color[3].get<unsigned char>()
    };
}

void SimulatorNode::load_world(const std::string& filepath) {
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

            float forced_height = w["height"].get<float>();
            float center_y = forced_height / 2.0f;

            wall.size = Vector3{w["width"].get<float>(), forced_height, w["depth"].get<float>()};

            wall.position = Vector3{
                w["x"].get<float>(), 
                center_y, 
                -w["y"].get<float>()
            };

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

            obj.position = Vector3{
                o["x"].get<float>(), 
                o["z"].get<float>(), 
                -o["y"].get<float>()
            };

            obj.color = parse_color(o["color"]);
            objects_.push_back(obj);
        }
    }

    RCLCPP_INFO(this->get_logger(), "World loaded: %lu walls, %lu objects.", walls_.size(), objects_.size());
}

void SimulatorNode::cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
    cmd_vel_x_ = msg->linear.x;
    cmd_vel_y_ = msg->linear.y;
    cmd_vel_z_ = msg->linear.z;
    cmd_roll_rate_ = msg->angular.x;
    cmd_pitch_rate_ = msg->angular.y;
    cmd_yaw_rate_ = msg->angular.z;
}

void SimulatorNode::publish_image() {
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

void SimulatorNode::publish_telemetry() {
    auto odom_msg = nav_msgs::msg::Odometry();
    odom_msg.header.stamp = this->now();
    odom_msg.header.frame_id = "odom";
    odom_msg.child_frame_id = "base_link";

    odom_msg.pose.pose.position.x = drone_pos_.x;
    odom_msg.pose.pose.position.y = -drone_pos_.z; 
    odom_msg.pose.pose.position.z = drone_pos_.y;

    // Convert Euler angles to quaternion
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
