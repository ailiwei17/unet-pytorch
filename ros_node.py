import rospy
from sensor_msgs.msg import CompressedImage as ros_compressed_image
from sensor_msgs.msg import Image as ros_image
from cv_bridge import CvBridge
import cv2

import numpy as np
from PIL import Image

from unet import Unet


class ImageProcessorNode:
    def __init__(self, subscriber_name, publisher_name, model, param):
        rospy.init_node('image_processor_node', anonymous=True)

        # 分割参数
        self.model = model
        self.param = param

        # 设置订阅图像的话题
        self.image_subscriber = rospy.Subscriber(subscriber_name, ros_compressed_image, self.image_callback)

        # 设置发布处理后图像的话题
        self.processed_image_publisher = rospy.Publisher(publisher_name, ros_image, queue_size=10)

        # 初始化CvBridge
        self.bridge = CvBridge()

    def image_callback(self, data):
        try:
            np_arr = np.fromstring(data.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            processed_image = self.process_image(cv_image)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            img_msg = self.bridge.cv2_to_imgmsg(processed_image, "bgr8")
            img_msg.header = data.header

            self.processed_image_publisher.publish(img_msg)

        except Exception as e:
            print(e)

    def process_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 转变成Image
        image = Image.fromarray(np.uint8(image))

        segment_image = self.model.detect_image(image, count=count, name_classes=name_classes)
        return np.array(segment_image)


if __name__ == '__main__':
    name_classes = ["background", "crack", "spallation", "exposedbars"]
    count = False

    detect_param = {
        'name_classes': name_classes,
        'count': False,
    }

    subscriber_topic = '/camera/color/image_raw/compressed'
    publisher_topic = '/camera/detect_image_color'

    unet = Unet()
    try:
        image_processor_node = ImageProcessorNode(subscriber_topic, publisher_topic, unet, detect_param)
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
