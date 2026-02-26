#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

# Try to import OpenCV, but handle the case where it's not available
try:
    import cv2
    import cv2.cv2 as cv
    from cv_bridge import CvBridge, CvBridgeError
    OPENCV_AVAILABLE = True
except ImportError:
    rospy.logwarn("OpenCV not available, camera driver will not capture real images")
    OPENCV_AVAILABLE = False

class MultiCameraTestNode:
    def __init__(self):
        rospy.init_node('multi_camera_driver_node')
        
        # Get parameters
        self.device_id = rospy.get_param('~device_id', 0)
        self.frame_rate = rospy.get_param('~frame_rate', 30)
        self.image_width = rospy.get_param('~image_width', 640)
        self.image_height = rospy.get_param('~image_height', 480)
        
        # Publishers for multiple cameras
        self.camera_pubs = {}
        self.camera_info_pubs = {}
        
        # Create publishers for each camera position
        camera_positions = ['chassis', 'hand', 'top']
        for position in camera_positions:
            topic_name = f'/camera/{position}/dual/color/image_raw'
            info_topic_name = f'/camera/{position}/camera_info'
            self.camera_pubs[position] = rospy.Publisher(topic_name, Image, queue_size=1)
            self.camera_info_pubs[position] = rospy.Publisher(info_topic_name, CameraInfo, queue_size=1)
        
        # Also maintain the original single camera topics for compatibility
        self.image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=1)
        self.camera_info_pub = rospy.Publisher('/camera/camera_info', CameraInfo, queue_size=1)
        
        if OPENCV_AVAILABLE:
            # Initialize CV bridge
            self.bridge = CvBridge()
            
            # Open camera device
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                rospy.logerr("Failed to open camera device %d", self.device_id)
                raise RuntimeError("Failed to open camera device")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.frame_rate)
        else:
            # Create a timer to publish dummy images if OpenCV is not available
            self.timer = rospy.Timer(rospy.Duration(1.0/self.frame_rate), self.publish_dummy_image)
        
        # Initialize camera info message
        self.camera_info_msg = CameraInfo()
        self.camera_info_msg.header.frame_id = "camera_link"
        self.camera_info_msg.width = self.image_width
        self.camera_info_msg.height = self.image_height
        # These are example values - should be calibrated for your specific camera
        self.camera_info_msg.K = [525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0]
        self.camera_info_msg.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.camera_info_msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.camera_info_msg.P = [525.0, 0.0, 319.5, 0.0, 0.0, 525.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0]
        
        rospy.loginfo("Camera driver node initialized with device %d", self.device_id)
    
    def publish_dummy_image(self, event=None):
        """Publish a dummy image when OpenCV is not available"""
        # Create a dummy image
        dummy_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        # Add some pattern to make it visible
        dummy_image[::50, :, :] = 255  # Horizontal lines
        dummy_image[:, ::50, :] = 255  # Vertical lines
        
        # Create image message
        if OPENCV_AVAILABLE:
            try:
                image_msg = self.bridge.cv2_to_imgmsg(dummy_image, "bgr8")
            except CvBridgeError as e:
                rospy.logerr("CV bridge error: %s", str(e))
                return
        else:
            image_msg = Image()
            image_msg.height = self.image_height
            image_msg.width = self.image_width
            image_msg.encoding = "bgr8"
            image_msg.is_bigendian = 0
            image_msg.step = self.image_width * 3
            image_msg.data = dummy_image.tobytes()
        
        # Set common header
        image_msg.header.stamp = rospy.Time.now()
        image_msg.header.frame_id = "camera_link"
        
        # Update camera info
        self.camera_info_msg.header.stamp = image_msg.header.stamp
        
        # Publish to all camera topics
        for position in self.camera_pubs:
            # Publish image
            image_msg.header.frame_id = f"{position}_camera_link"
            self.camera_pubs[position].publish(image_msg)
            
            # Publish camera info
            self.camera_info_msg.header.frame_id = f"{position}_camera_link"
            self.camera_info_pubs[position].publish(self.camera_info_msg)
        
        # Also publish to original single camera topics for compatibility
        image_msg.header.frame_id = "camera_link"
        self.image_pub.publish(image_msg)
        self.camera_info_msg.header.frame_id = "camera_link"
        self.camera_info_pub.publish(self.camera_info_msg)
        
        rospy.logdebug("Published dummy camera images: %dx%d", dummy_image.shape[1], dummy_image.shape[0])
    
    def capture_and_publish(self):
        """Capture image from camera and publish it to all camera topics"""
        if not OPENCV_AVAILABLE:
            return
            
        ret, cv_image = self.cap.read()
        if not ret:
            rospy.logwarn("Failed to capture image from camera")
            return
        
        try:
            # Convert OpenCV image to ROS Image message
            image_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            image_msg.header.stamp = rospy.Time.now()
            
            # Update camera info
            self.camera_info_msg.header.stamp = image_msg.header.stamp
            
            # Publish to all camera topics
            for position in self.camera_pubs:
                # Publish image
                image_msg.header.frame_id = f"{position}_camera_link"
                self.camera_pubs[position].publish(image_msg)
                
                # Publish camera info
                self.camera_info_msg.header.frame_id = f"{position}_camera_link"
                self.camera_info_pubs[position].publish(self.camera_info_msg)
            
            # Also publish to original single camera topics for compatibility
            image_msg.header.frame_id = "camera_link"
            self.image_pub.publish(image_msg)
            self.camera_info_msg.header.frame_id = "camera_link"
            self.camera_info_pub.publish(self.camera_info_msg)
            
            rospy.logdebug("Published camera images: %dx%d", cv_image.shape[1], cv_image.shape[0])
            
        except CvBridgeError as e:
            rospy.logerr("CV bridge error: %s", str(e))
        except Exception as e:
            rospy.logerr("Error publishing image: %s", str(e))
    
    def run(self):
        """Main loop"""
        if OPENCV_AVAILABLE:
            rate = rospy.Rate(self.frame_rate)
            while not rospy.is_shutdown():
                self.capture_and_publish()
                rate.sleep()
        else:
            # Just spin to keep the node alive
            rospy.spin()
    
    def shutdown(self):
        """Cleanup on shutdown"""
        if OPENCV_AVAILABLE and self.cap and self.cap.isOpened():
            self.cap.release()
        rospy.loginfo("Camera driver node shutdown")

if __name__ == '__main__':
    try:
        node = MultiCameraTestNode()
        rospy.on_shutdown(node.shutdown)
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("Multi-camera driver node error: %s", str(e))
