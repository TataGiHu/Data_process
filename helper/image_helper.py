import cv2
import rosbag
from cv_bridge import CvBridge
import os
import os.path as osp

def save_img_compressed(brige, msg, img_name):
    cv_image = brige.compressed_imgmsg_to_cv2(msg, "bgr8")
    cv2.imwrite(img_name, cv_image)

def save_img(brige, msg, img_name):
    cv_image = brige.imgmsg_to_cv2(msg, "bgr8")
    cv2.imwrite(img_name, cv_image)

