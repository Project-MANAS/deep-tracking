#! /usr/bin/python
import rospy
import torch
import numpy as np
import cv2
import message_filters

from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import OccupancyGrid

class DeepTracker:
    def __init__(self, costmap_topic, visibility_layer_topic, publisher_topic, model_path, img_dim=(51,51)):
        batch_size = 1
        self.img_dim = img_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else cpu)
        self.dt = torch.load(model_path) # or DeepTrackerLSTM((3, batch_size, 16, img_dim, img_dim), False, False).to(device)
        zero_tensor = torch.zeros((batch_size, 2, self.img_dim[0], self.img_dim[0])).to(self.device)
        print(self.dt)

        self.pub = rospy.Publisher(publisher_topic, CompressedImage, queue_size=10)
        
        self.img_sub = message_filters.Subscriber(costmap_topic, OccupancyGrid)
        self.visibility_sub = message_filters.Subscriber(visibility_layer_topic, CompressedImage)
        
        ts = message_filters.ApproximateTimeSynchronizer([self.img_sub, self.visibility_sub], 10,10)
        ts.registerCallback(self.callback)
        
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down deep tracker node...")

    def callback(self, img_msg, visibility_layer_msg):
        img = np.expand_dims(img_msg.data, 1)
        img = img.astype(np.float64)
        img_dims = (img_msg.info.height, img_msg.info.width)
        img = np.reshape(img, img_dims)
        img = cv2.resize(img, (500,500))
        img = cv2.GaussianBlur(img, (19,19), 0)
        cv2.imshow('input', img)
        img = cv2.resize(img, self.img_dim)
        img = np.reshape(img, (1, 1, self.img_dim[0], self.img_dim[0]))

        visibility_layer = visibility_layer_msg.data
        visibility_layer = cv2.imdecode(np.fromstring(visibility_layer, np.uint8), cv2.IMREAD_GRAYSCALE)
        visibility_layer = np.resize(visibility_layer, self.img_dim)
        cv2.imshow('visibility', visibility_layer)
        
        visibility_layer = np.reshape(visibility_layer, [1, 1, self.img_dim[0], self.img_dim[1]])
        img = np.concatenate([img, visibility_layer], axis=1)
        
        with torch.no_grad():
            t_img = torch.tensor(img, dtype=torch.float).to(self.device)
            op = self.dt(t_img)
            op_img = torch.squeeze(op).detach().cpu().numpy()
            cv2.imshow('image',op_img)
            cv2.waitKey(1)
            msg = CompressedImage()
            msg.header = img_msg.header
            msg.format = "jpeg"
            jpg_data = cv2.imencode('.jpg', torch.squeeze(op).detach().cpu().numpy())[0]
            msg.data = np.array(jpg_data).tostring()
            
            self.pub.publish(msg)

if __name__ == "__main__":
    rospy.init_node('deep_tracker')
    publisher_topic = 'tracker_frames'
    costmap_topic = '/move_base/local_costmap/obstacle_layer/obstacle_map'
    visibility_layer_topic = 'visibility_layer'
    model_path = 'saved_models/model_8.pt'
    DeepTracker(costmap_topic, visibility_layer_topic, publisher_topic, model_path)

