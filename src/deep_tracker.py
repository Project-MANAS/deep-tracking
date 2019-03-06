#! /usr/bin/python3
import rospy
import torch
import numpy as np
import cv2

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from model import DeepTrackerLSTM

class DeepTracker:
    def __init__(self):
        batch_size = 1
        self.img_dim = (51, 51)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else cpu)
        self.dt = torch.load('saved_models/model_8.pt') #DeepTrackerLSTM((3, batch_size, 16, img_dim, img_dim), False, False).to(device)
        zero_tensor = torch.zeros((batch_size, 2, self.img_dim[0], self.img_dim[0])).to(self.device)
        #self.dt = torch.jit.trace(self.dt, (zero_tensor))
        print(self.dt)
        self.pub = rospy.Publisher('tracker_frames', CompressedImage, queue_size=10)
        self.sub = rospy.Subscriber('/move_base/global_costmap/obstacle_layer/obstacle_map', OccupancyGrid, self.callback)
        self.bridge = CvBridge()
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down deep tracker node...")

    def callback(self, data):
        img = np.expand_dims(data.data, 1)
        img = img.astype(np.float64)
        img = np.reshape(img, (data.info.height, data.info.width))
        img = cv2.resize(img, (500,500))
        img = cv2.GaussianBlur(img, (19,19), 0)
        cv2.imshow('input', img)
        img = cv2.resize(img, self.img_dim)
        img = np.reshape(img, (1, 1, self.img_dim[0], self.img_dim[0]))
        img = np.concatenate([img, np.zeros((1, 1, self.img_dim[0], self.img_dim[0]))], axis=1)
        with torch.no_grad():
            t_img = torch.tensor(img, dtype=torch.float).to(self.device)
            op = self.dt(t_img)
            op_img = torch.squeeze(op).detach().cpu().numpy()
            cv2.imshow('image',op_img)
            cv2.waitKey(1)
            msg = CompressedImage()
            msg.header = data.header
            msg.format = "jpeg"
            jpg_data = cv2.imencode('.jpg', torch.squeeze(op).detach().cpu().numpy())
            msg.data = np.array(jpg_data).tostring()
            
            self.pub.publish(msg)

if __name__ == "__main__":
    rospy.init_node('deep_tracker')
    DeepTracker()

