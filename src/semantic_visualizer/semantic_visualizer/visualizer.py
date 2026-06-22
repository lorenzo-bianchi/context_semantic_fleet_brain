#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        
        # In ROS 2 Python, we subscribe directly to the standard topic
        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10 # QoS depth
        )

    def listener_callback(self, msg):
        try:
            # cv_bridge converts the ROS message into an OpenCV matrix (NumPy array)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Display the image on screen
            cv2.imshow("Semantic Fleet Brain - Visualizer", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'Error during image conversion: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Handle clean exit when pressing Ctrl+C in the terminal
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        # Ensure that OpenCV windows close correctly
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()