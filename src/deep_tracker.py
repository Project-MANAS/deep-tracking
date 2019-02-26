#! /usr/bin/python3
import rospy
import torch
import numpy as np

from sensor_msgs.msg import CompressedImage
from model import DeepTrackerLSTM

class DeepTracker:
    def __init__(self):
        batch_size = 1
        img_dim = 51
        device = torch.device("cuda:0" if torch.cuda.is_available() else cpu)
        self.dt = torch.load('saved_models/model_8.pt') #DeepTrackerLSTM((3, batch_size, 16, img_dim, img_dim), False, False).to(device)
        zero_tensor = torch.zeros((batch_size, 2, img_dim, img_dim)).to(device)
        self.dt = torch.jit.trace(self.dt, (zero_tensor))
        print(self.dt)
        self.pub = rospy.Publisher('tracker_frames', CompressedImage, queue_size=10)
        self.sub = rospy.Subscriber('tracker_input', CompressedImage, self.callback)

        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down deep tracker node...")

    def callback(self, data):
        np_arr = np.fromstring(data.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        t_img = torch.tensor(img)
        op = self.dt(t_img)
        
        msg = CompressedImage()
        msg.header = data.header
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', op)[1]).tostring()

        self.pub.publish(msg)

if __name__ == "__main__":
    rospy.init_node('deep_tracker')
    DeepTracker()

