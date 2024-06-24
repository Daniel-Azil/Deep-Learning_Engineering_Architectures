"""
    A simple module that uses yolo version 8 to detect objects in an image.
"""

from ultralytics import YOLO

# we import cv2 to make the detection for the image last longer when it displays show 
import cv2




# Choose the model weight to use, there are different types. The nano weights, medium weights and the large weights.
# The n, m and l characters in yolov8n.pt, yolov8m.pt and yolov8l.pt stands for nano, medium and large weights. pt is an entension for pytorch files.
# You either write ""YOLO('yolov8n.pt') or any of the other weight to download the weight from github into the current dir for the first time or add the path
# of the weight of the model after it being downloaded and moved into the weight directory.
# To download to weight from github, the codes below are used to download the nano, medium and large weight relatively.
YOLO('yolov8n.pt')
YOLO('yolov8m.pt')
YOLO('yolov8l.pt')

#load the relative path to the weight of the model we want to use, eg nano weights
model = YOLO('model_weights/yolov8n.pt')

# pass in the image relative path to the model. Pass in show=True to see vebose results
results = model("images/school_bus2.png", show=True)

cv2.waitKey(0)
