#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage
import redis
import json
import time
import math
import cv2
import numpy as np
import base64

class RedisBridgeNode(Node):
    def __init__(self):
        super().__init__('redis_bridge_node')

        # Parameters
        self.declare_parameter('control.kp_angular', 0.0)
        self.declare_parameter('control.kp_linear', 0.0)
        self.declare_parameter('control.kp_visual_x', 0.0)
        self.declare_parameter('control.kp_visual_yaw', 0.0)
        self.declare_parameter('control.kp_visual_z', 0.0)
        self.declare_parameter('control.kp_z', 0.0)
        self.declare_parameter('nav.distance_tolerance', 0.0)
        self.declare_parameter('nav.search_yaw_rate', 0.0)
        self.declare_parameter('nav.yaw_tolerance', 0.0)
        self.declare_parameter('redis.host', '')
        self.declare_parameter('redis.memory_queue', '')
        self.declare_parameter('redis.port', 0)
        self.declare_parameter('redis.task_queue', '')
        self.declare_parameter('vision.capture_cooldown', 0.0)
        self.declare_parameter('vision.hsv_lower.h', 0)
        self.declare_parameter('vision.hsv_lower.s', 0)
        self.declare_parameter('vision.hsv_lower.v', 0)
        self.declare_parameter('vision.hsv_upper.h', 0)
        self.declare_parameter('vision.hsv_upper.s', 0)
        self.declare_parameter('vision.hsv_upper.v', 0)
        self.declare_parameter('vision.jpeg_quality.high', 0)
        self.declare_parameter('vision.jpeg_quality.low', 0)
        self.declare_parameter('vision.min_contour_area', 0.0)
        self.declare_parameter('vision.tol_visual_area', 0.0)
        self.declare_parameter('vision.tol_visual_xy', 0.0)
        self.declare_parameter('vision.visual_approach_timeout', 0.0)
        self.declare_parameter('vision.visual_target_area', 0.0)

        self.kp_angular = self.get_parameter('control.kp_angular').value
        self.kp_linear = self.get_parameter('control.kp_linear').value
        self.kp_visual_x = self.get_parameter('control.kp_visual_x').value
        self.kp_visual_yaw = self.get_parameter('control.kp_visual_yaw').value
        self.kp_visual_z = self.get_parameter('control.kp_visual_z').value
        self.kp_z = self.get_parameter('control.kp_z').value
        self.nav_distance_tolerance = self.get_parameter('nav.distance_tolerance').value
        self.search_yaw_rate = self.get_parameter('nav.search_yaw_rate').value
        self.nav_yaw_tolerance = self.get_parameter('nav.yaw_tolerance').value
        self.redis_host = self.get_parameter('redis.host').value
        self.memory_queue = self.get_parameter('redis.memory_queue').value
        self.redis_port = self.get_parameter('redis.port').value
        self.task_queue = self.get_parameter('redis.task_queue').value
        self.capture_cooldown = self.get_parameter('vision.capture_cooldown').value
        self.hsv_lower_h = self.get_parameter('vision.hsv_lower.h').value
        self.hsv_lower_s = self.get_parameter('vision.hsv_lower.s').value
        self.hsv_lower_v = self.get_parameter('vision.hsv_lower.v').value
        self.hsv_upper_h = self.get_parameter('vision.hsv_upper.h').value
        self.hsv_upper_s = self.get_parameter('vision.hsv_upper.s').value
        self.hsv_upper_v = self.get_parameter('vision.hsv_upper.v').value
        self.jpeg_quality_high = self.get_parameter('vision.jpeg_quality.high').value
        self.jpeg_quality_low = self.get_parameter('vision.jpeg_quality.low').value
        self.min_contour_area = self.get_parameter('vision.min_contour_area').value
        self.tol_visual_area = self.get_parameter('vision.tol_visual_area').value
        self.tol_visual_xy = self.get_parameter('vision.tol_visual_xy').value
        self.visual_approach_timeout = self.get_parameter('vision.visual_approach_timeout').value
        self.visual_target_area = self.get_parameter('vision.visual_target_area').value

        # --- LOG PARAMETERS ---
        self.get_logger().info("--- PARAMETERS LOADED ---")
        self.get_logger().info("[CONTROL]")
        self.get_logger().info(f"  - kp_angular: {self.kp_angular}")
        self.get_logger().info(f"  - kp_linear: {self.kp_linear}")
        self.get_logger().info(f"  - kp_visual_x: {self.kp_visual_x}")
        self.get_logger().info(f"  - kp_visual_yaw: {self.kp_visual_yaw}")
        self.get_logger().info(f"  - kp_visual_z: {self.kp_visual_z}")
        self.get_logger().info(f"  - kp_z: {self.kp_z}")
        self.get_logger().info("[NAV]")
        self.get_logger().info(f"  - distance_tolerance: {self.nav_distance_tolerance}")
        self.get_logger().info(f"  - search_yaw_rate: {self.search_yaw_rate}")
        self.get_logger().info(f"  - yaw_tolerance: {self.nav_yaw_tolerance}")
        self.get_logger().info("[REDIS]")
        self.get_logger().info(f"  - host: {self.redis_host}")
        self.get_logger().info(f"  - memory_queue: {self.memory_queue}")
        self.get_logger().info(f"  - port: {self.redis_port}")
        self.get_logger().info(f"  - task_queue: {self.task_queue}")
        self.get_logger().info("[VISION]")
        self.get_logger().info(f"  - capture_cooldown: {self.capture_cooldown}")
        self.get_logger().info(f"  - hsv_lower: [h:{self.hsv_lower_h}, s:{self.hsv_lower_s}, v:{self.hsv_lower_v}]")
        self.get_logger().info(f"  - hsv_upper: [h:{self.hsv_upper_h}, s:{self.hsv_upper_s}, v:{self.hsv_upper_v}]")
        self.get_logger().info(f"  - jpeg_quality_high: {self.jpeg_quality_high}")
        self.get_logger().info(f"  - jpeg_quality_low: {self.jpeg_quality_low}")
        self.get_logger().info(f"  - min_contour_area: {self.min_contour_area}")
        self.get_logger().info(f"  - tol_visual_area: {self.tol_visual_area}")
        self.get_logger().info(f"  - tol_visual_xy: {self.tol_visual_xy}")
        self.get_logger().info(f"  - visual_approach_timeout: {self.visual_approach_timeout}")
        self.get_logger().info(f"  - visual_target_area: {self.visual_target_area}")
        self.get_logger().info("---------------------------")

        self.odom_cb_group = MutuallyExclusiveCallbackGroup()
        self.control_cb_group = MutuallyExclusiveCallbackGroup()
        self.vision_cb_group = MutuallyExclusiveCallbackGroup()

        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=True)
            self.redis_client.ping()
            self.get_logger().info("🔗 Successfully connected to Redis Broker.")
        except redis.ConnectionError as e:
            self.get_logger().error(f"❌ Failed to connect to Redis: {e}")
            raise SystemExit

        self.is_executing = False
        self.last_capture_time = 0.0

        # ROS 2 publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.annotated_image_pub = self.create_publisher(
            CompressedImage,
            '/vision/annotated_image/compressed',
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10,
            callback_group=self.odom_cb_group
        )

        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            10,
            callback_group=self.vision_cb_group
        )

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_yaw = 0.0

        self.target_cx = None
        self.target_cy = None
        self.target_area = None

        self.image_width = None
        self.image_height = None

        self.semantic_map = {
            "north corridor": {"x": 5.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
            "red box": {"x": 5.0, "y": 3.0, "z": 1.5, "yaw": 1.57}
        }

        self.exploration_waypoints = [
            (-4.0,  6.0, 1.5),
            ( 5.0, -8.0, 1.5),
            ( 7.0,  4.0, 1.5),
            (-6.0, -4.0, 1.5),
            ( 2.0, -1.0, 1.5),
        ]

        self.timer = self.create_timer(0.5, self.poll_queue, callback_group=self.control_cb_group)

    def euler_from_quaternion(self, x, y, z, w):
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t3, t4)

    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        self.current_x = pos.x
        self.current_y = pos.y
        self.current_z = pos.z
        self.current_yaw = self.euler_from_quaternion(ori.x, ori.y, ori.z, ori.w)

    def image_callback(self, msg: CompressedImage):
        # Decode image
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        self.image_height, self.image_width = frame.shape[:2]

        annotated_frame = frame.copy()

        # HSV color thresholding for target detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([self.hsv_lower_h, self.hsv_lower_s, self.hsv_lower_v])
        upper_bound = np.array([self.hsv_upper_h, self.hsv_upper_s, self.hsv_upper_v])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Detect contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        best_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area and area > max_area:
                max_area = area
                best_contour = contour

        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)

            self.target_cx = x + (w / 2)
            self.target_cy = y + (h / 2)
            self.target_area = max_area

            # Draw visual feedback
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Area: {int(max_area)}px", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(annotated_frame, (int(self.target_cx), int(self.target_cy)), 5, (0, 0, 255), -1)

            # Upload detected object data to Redis
            if self.is_executing:
                current_time = time.time()

                if current_time - self.last_capture_time > self.capture_cooldown:
                    self.last_capture_time = current_time
                    self.get_logger().info(f"📸 Object detected! Bounding Box: [x:{x}, y:{y}, w:{w}, h:{h}]. Sending to Redis...")

                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality_high])
                    b64_image = base64.b64encode(buffer).decode('utf-8')

                    payload = {
                        "x": round(self.current_x, 2),
                        "y": round(self.current_y, 2),
                        "z": round(self.current_z, 2),
                        "yaw": round(self.current_yaw, 2),
                        "image": b64_image,
                        "timestamp": current_time
                    }
                    self.redis_client.rpush(self.memory_queue, json.dumps(payload))
        else:
            self.target_cx = None
            self.target_cy = None
            self.target_area = None

        # Publish annotated frame to ROS 2
        _, annot_buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality_high])
        annot_msg = CompressedImage()
        annot_msg.header.stamp = self.get_clock().now().to_msg()
        annot_msg.format = "jpeg"
        annot_msg.data = annot_buffer.tobytes()
        self.annotated_image_pub.publish(annot_msg)

        # Publish live stream via Redis Pub/Sub
        _, stream_buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality_low])
        b64_stream = base64.b64encode(stream_buffer).decode('utf-8')
        self.redis_client.publish("live_video_stream", b64_stream)

    def poll_queue(self):
        if self.is_executing:
            return

        task_data = self.redis_client.lpop(self.task_queue)
        if task_data:
            try:
                task = json.loads(task_data)
                self.get_logger().info(f"📦 New Plan Received: Task ID [{task.get('task_id')}]")
                self.is_executing = True
                self.execute_plan(task.get('plan', []))
            except json.JSONDecodeError as e:
                self.get_logger().error(f"Failed to parse task JSON: {e}")
            finally:
                self.is_executing = False

    def execute_plan(self, plan):
        for step in plan:
            action = step.get('action', 'UNKNOWN')
            target = step.get('target', 'UNKNOWN')
            explicit_goal = step.get("explicit_goal", None)

            self.get_logger().info(f"   ---> Executing: {action} towards '{target}'")

            if action == "NAVIGATE":
                self.handle_navigate(target, explicit_goal=explicit_goal)
            elif action == "SEARCH":
                self.handle_search(target)
            elif action == "EXPLORE":
                self.handle_explore()

            time.sleep(0.5)
        self.get_logger().info("✅ Plan fully executed.\n")

    def handle_navigate(self, target, explicit_goal=None, is_exploration=False):
        target_yaw = None

        if explicit_goal:
            if len(explicit_goal) == 4:
                goal_x, goal_y, goal_z, target_yaw = explicit_goal
            else:
                goal_x, goal_y, goal_z = explicit_goal[:3]
        else:
            goal = self.semantic_map.get(target.lower(), {"x": 0.0, "y": 0.0, "z": 1.5, "yaw": 0.0})
            goal_x, goal_y, goal_z = goal["x"], goal["y"], goal["z"]
            target_yaw = goal.get("yaw", None)

        msg = Twist()
        self.get_logger().info(f"        🚁 In flight towards {target} ({goal_x:.1f}, {goal_y:.1f}, {goal_z:.1f})...")

        while rclpy.ok():
            dx = goal_x - self.current_x
            dy = goal_y - self.current_y
            dz = goal_z - self.current_z

            distance_3d = math.sqrt(dx**2 + dy**2 + dz**2)

            desired_yaw = target_yaw if target_yaw is not None else math.atan2(dy, dx)
            yaw_error = desired_yaw - self.current_yaw
            yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

            if distance_3d < self.nav_distance_tolerance:
                if target_yaw is None or abs(yaw_error) < self.nav_yaw_tolerance:
                    break

            local_dx = dx * math.cos(self.current_yaw) + dy * math.sin(self.current_yaw)
            local_dy = -dx * math.sin(self.current_yaw) + dy * math.cos(self.current_yaw)

            msg.linear.x = self.kp_linear * local_dx
            msg.linear.y = self.kp_linear * local_dy
            msg.linear.z = self.kp_z * dz
            msg.angular.z = self.kp_angular * yaw_error

            self.cmd_vel_pub.publish(msg)
            time.sleep(0.05)

        msg.linear.x = msg.linear.y = msg.linear.z = msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)
        self.get_logger().info(f"        ✅ Target '{target}' reached.")

        if not is_exploration and target.lower() != "coordinates":
            self.handle_visual_approach()

    def handle_search(self, target):
        self.get_logger().info(f"        👁️ [Vision] Scanning 360° for '{target}'...")
        msg = Twist()
        msg.angular.z = float(self.search_yaw_rate)
        scan_duration = (2.0 * math.pi) / self.search_yaw_rate
        start_time = time.time()

        while rclpy.ok() and (time.time() - start_time) < scan_duration:
            self.cmd_vel_pub.publish(msg)
            time.sleep(0.05)

        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)
        self.get_logger().info(f"        ✅ [Vision] Scan completed.")

    def handle_return_home(self):
        self.get_logger().info("        🏠 Returning to Home Base (0, 0, 1.5)...")
        self.handle_navigate("Home Base", explicit_goal=(0.0, 0.0, 1.5), is_exploration=True)
        self.get_logger().info("        ✅ Safely returned home.")

    def handle_explore(self):
        self.get_logger().info("        🗺️ Initiating Systematic Exploration...")
        for i, waypoint in enumerate(self.exploration_waypoints):
            self.get_logger().info(f"        🧭 Moving to waypoint {i+1}/{len(self.exploration_waypoints)}")
            self.handle_navigate(f"Waypoint {i+1}", explicit_goal=waypoint, is_exploration=True)
            self.handle_search("environment")
            time.sleep(1.0)
        self.handle_return_home()

    def handle_visual_approach(self):
        self.get_logger().info("        👀 Object in sight! Centering...")
        msg = Twist()

        img_center_x = self.image_width / 2.0
        img_center_y = self.image_height / 2.0

        start_time = time.time()

        while rclpy.ok() and (time.time() - start_time) < self.visual_approach_timeout:
            if self.target_cx is None:
                msg.linear.x = msg.linear.z = msg.angular.z = 0.0
                self.cmd_vel_pub.publish(msg)
                time.sleep(0.05)
                continue

            error_x = img_center_x - self.target_cx
            error_y = img_center_y - self.target_cy
            error_area = self.visual_target_area - self.target_area

            msg.angular.z = self.kp_visual_yaw * error_x
            msg.linear.z = self.kp_visual_z * error_y
            msg.linear.x = self.kp_visual_x * error_area if abs(error_area) > self.tol_visual_area else 0.0

            self.cmd_vel_pub.publish(msg)

            # Check tolerances
            if (abs(error_x) < self.tol_visual_xy and
                abs(error_y) < self.tol_visual_xy and
                abs(error_area) < self.tol_visual_area):
                break

            time.sleep(0.05)

        msg.linear.x = msg.linear.y = msg.linear.z = msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RedisBridgeNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
