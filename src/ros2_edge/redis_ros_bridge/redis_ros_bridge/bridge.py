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

        self.odom_cb_group = MutuallyExclusiveCallbackGroup()
        self.control_cb_group = MutuallyExclusiveCallbackGroup()
        self.vision_cb_group = MutuallyExclusiveCallbackGroup()

        # Redis Setup
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
            self.get_logger().info("🔗 Successfully connected to Redis Broker.")
        except redis.ConnectionError as e:
            self.get_logger().error(f"❌ Failed to connect to Redis: {e}")
            raise SystemExit

        self.task_queue = "robot_tasks_queue"
        self.memory_queue = "semantic_memory_queue"
        self.is_executing = False

        self.last_capture_time = 0.0
        self.capture_cooldown = 2.0  # Cooldown between snapshots

        # ROS 2 interfaces
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Publisher for visual debug feed with bounding boxes
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
        # 1. Decode incoming image
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        annotated_frame = frame.copy()

        # 2. Convert color space and extract HSV mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0, 100, 50])
        upper_bound = np.array([179, 255, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # 3. Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)

                # Draw bounding box and label on annotated frame
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Area: {int(area)}px", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if self.is_executing:
                    current_time = time.time()

                    # Check debounce timer before Redis transmission
                    if current_time - self.last_capture_time > self.capture_cooldown:
                        self.last_capture_time = current_time

                        self.get_logger().info(f"📸 Object detected! Bounding Box: [x:{x}, y:{y}, w:{w}, h:{h}]. Sending to Redis...")

                        # Encode original clean frame for server
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
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
                        break 

        # 4. Publish annotated frame to ROS 2
        _, annot_buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        annot_msg = CompressedImage()
        annot_msg.header.stamp = self.get_clock().now().to_msg()
        annot_msg.format = "jpeg"
        annot_msg.data = np.array(annot_buffer).tobytes()

        self.annotated_image_pub.publish(annot_msg)

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

    def handle_navigate(self, target, explicit_goal=None):
        if explicit_goal:
            goal_x, goal_y, goal_z = explicit_goal
        else:
            goal = self.semantic_map.get(target.lower(), {"x": 0.0, "y": 0.0, "z": 1.5, "yaw": 0.0})
            goal_x, goal_y, goal_z = goal["x"], goal["y"], goal["z"]

        Kp_linear = 0.6
        Kp_z = 0.8
        Kp_angular = 1.5
        distance_tolerance = 0.20 

        msg = Twist()
        self.get_logger().info(f"        🚁 In flight towards {target} ({goal_x:.1f}, {goal_y:.1f}, {goal_z:.1f})...")

        while rclpy.ok():
            dx = goal_x - self.current_x
            dy = goal_y - self.current_y
            dz = goal_z - self.current_z

            distance_xy = math.sqrt(dx**2 + dy**2)
            distance_3d = math.sqrt(dx**2 + dy**2 + dz**2)

            if distance_3d < distance_tolerance:
                self.get_logger().info(f"        📍 Target '{target}' reached.")
                break

            angle_to_goal = math.atan2(dy, dx)
            yaw_error = angle_to_goal - self.current_yaw
            yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

            msg.angular.z = Kp_angular * yaw_error

            if abs(yaw_error) > 0.5:
                msg.linear.x = 0.0
            else:
                msg.linear.x = Kp_linear * distance_xy

            msg.linear.z = Kp_z * dz

            self.cmd_vel_pub.publish(msg)
            time.sleep(0.05) 

        msg.linear.x = msg.linear.y = msg.linear.z = msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)

    def handle_search(self, target):
        self.get_logger().info(f"        👁️ [Vision] Scanning environment 360° for '{target}'...")

        msg = Twist()
        yaw_rate = 1.0 
        msg.angular.z = yaw_rate

        scan_duration = (2.0 * math.pi) / yaw_rate 
        start_time = time.time()

        while rclpy.ok() and (time.time() - start_time) < scan_duration:
            self.cmd_vel_pub.publish(msg)
            time.sleep(0.05) 

        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)
        self.get_logger().info(f"        ✅ [Vision] Scan completed.")

    def handle_return_home(self):
        self.get_logger().info("        🏠 Returning to Home Base (0, 0, 1.5)...")
        home_waypoint = (0.0, 0.0, 1.5)
        self.handle_navigate("Home Base", explicit_goal=home_waypoint)
        self.get_logger().info("        ✅ Safely returned home.")

    def handle_explore(self):
        self.get_logger().info("        🗺️ Initiating Systematic Exploration Plan...")

        for i, waypoint in enumerate(self.exploration_waypoints):
            self.get_logger().info(f"        🧭 Moving to waypoint {i+1}/{len(self.exploration_waypoints)}: {waypoint}")
            self.handle_navigate(f"Waypoint {i+1}", explicit_goal=waypoint)
            self.handle_search("environment")
            time.sleep(1.0)

        self.handle_return_home()
        self.get_logger().info("        ✅ Systematic exploration completed.")

def main(args=None):
    rclpy.init(args=args)
    node = RedisBridgeNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Redis Bridge Node...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()