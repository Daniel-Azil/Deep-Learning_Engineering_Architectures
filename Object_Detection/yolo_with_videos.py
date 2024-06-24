from ultralytics import YOLO
import cv2
import cvzone
import math


# pass in path to video file for detection operation
cap = cv2.VideoCapture("videos/highway_traffic_flow.mp4")



# retrieve the model weights for detection
model = YOLO("model_weights/yolov8n.pt")


# class name for detection

className = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
             'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
             'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
             'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
             'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
             'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
             'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
             'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
             'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
             'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
             'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]


while True:
    success, img = cap.read()

    
    results = model(img, stream=True) #it's recommended to set stream as True as it utilizes generators for efficiency.

    # get the bounding box
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box

             # get the angles of the bounding box, x and y of all four corner angle
             # using xyxy which stands for x1, y1, x2, y2
            # x1, y1, x2, y2 = box.xyxy[0]
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)

            # # add colour to the bounding box
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # we have the colour pixel of the box and 3 being the colour tickness


            # get the angles of the bounding box, x and y of all four corner angle
            # using xywh which stands for x1, y1, w, h
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            # get prediction confidence/rate
            conf = math.ceil(box.conf[0]*100) / 100   # we do this operation "math.ceil(box.conf[0]*100) / 100" to get the confidence to 2 decimal places
            

            # Class Name detection

            # retieve the index of the class name from the model's prediction so we can pass it to the className list to retrieve the actual class name in string
            cls = int(box.cls[0])  # the int function is used in converting the original number being float to an integer 
            print()


            # in include class name and prediction confidence into the bounding box
            # we use (max(0, x1), max(35, y1)) to ensure the text is not out of the image frame when objects are detected at the top of the frame
            cvzone.putTextRect(img, f"{className[cls]} {conf}", (max(0, x1), max(35, y1)), scale=0.7, thickness=1)




    cv2.imshow("Image", img)
    cv2.waitKey(1)  

