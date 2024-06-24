from ultralytics import YOLO
import cv2
import cvzone
import math


# pass in path to video file for detection operation
cap = cv2.VideoCapture("vidoe/cards.mp4")



# retrieve the model weights for detection
model = YOLO("model_weights/playingCards.pt")


# class name for detection

className = ['10C', '10D', '10H', '10S',
              '2C', '2D', '2H', '2S',
              '3C', '3D', '3H', '3S',
              '4C', '4D', '4H', '4S',
              '5C', '5D', '5H', '5S',
              '6C', '6D', '6H', '6S',
              '7C', '7D', '7H', '7S',
              '8C', '8D', '8H', '8S',
              '9C', '9D', '9H', '9S',
              'AC', 'AD', 'AH', 'AS',
              'JC', 'JD', 'JH', 'JS',
              'KC', 'KD', 'KH', 'KS',
              'QC', 'QD', 'QH', 'QS']


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

