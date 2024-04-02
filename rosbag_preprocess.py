import rospy
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import numpy as np
from PIL import Image

from unet import Unet

name_classes = ["background", "crack", "spallation", "exposedbars"]
count = False

detect_param = {
    'name_classes': name_classes,
    'count': False,
}

unet = Unet()


def segment_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 转变成Image
    image = Image.fromarray(np.uint8(image))

    segmented_image = unet.detect_image(image, count=count, name_classes=name_classes)
    return np.array(segmented_image)


def process_image(image_msg):
    # Convert ROS Image message to OpenCV image
    np_arr = np.fromstring(image_msg.data, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    processed_image = segment_image(cv_image)

    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)

    # Convert processed image back to ROS Image message
    processed_image_msg = CvBridge().cv2_to_imgmsg(processed_image, encoding="bgr8")

    processed_image_msg.header = image_msg.header

    return processed_image_msg


def main():
    input_bag_filename = "/home/liwei/catkin_ws/src/r3live/dataset/test_1.bag"
    output_bag_filename = "/home/liwei/catkin_ws/src/r3live/dataset/test_1_preprocess.bag"

    with rosbag.Bag(output_bag_filename, 'w') as output_bag:
        for topic, msg, t in rosbag.Bag(input_bag_filename).read_messages():
            if topic == "/camera/color/image_raw/compressed":
                processed_image_msg = process_image(msg)
                output_bag.write("/camera/detect_image_color", processed_image_msg, t)
            else:
                output_bag.write(topic, msg, t)


if __name__ == '__main__':
    rospy.init_node('image_processing_node')
    main()
