import anyio
import asyncio
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

import aioros
from std_msgs.msg import String

from aioros import init_node
from aioros._node._node import Node
from aioros.master import init_master


# Instantiate CvBridge
bridge = CvBridge()


async def main() -> None:
    async with init_node(
        "test_subscriber1"
    ) as publisher_node, init_node(
        "test_subscriber2"
      
    ) as subscriber_node:
        async with publisher_node.create_subscription(
            "/t265/stereo_ir/left/fisheye_image_raw", Image
        ) as subscription_left, subscriber_node.create_subscription(
            "/t265/stereo_ir/right/fisheye_image_raw", Image
        ) as subscription_right:
            iter_left = subscription_left.__aiter__()
            iter_right = subscription_right.__aiter__()
            print(iter_left, iter_right)
            while True:
                img_left, img_right = await asyncio.gather(iter_left.__anext__(), iter_right.__anext__())
                cv2_img_left = bridge.imgmsg_to_cv2(img_left, "bgr8")
                cv2_img_right = bridge.imgmsg_to_cv2(img_right, "bgr8")
                cv2_img = np.concatenate((cv2_img_left, cv2_img_right), axis=1)
                cv2.imshow("frame", cv2_img)
                if cv2.waitKey(1) & 0xFF == ord('q'): # wait for 1 millisecond
                    break

                


            

              


if __name__ == "__main__":
    anyio.run(main)