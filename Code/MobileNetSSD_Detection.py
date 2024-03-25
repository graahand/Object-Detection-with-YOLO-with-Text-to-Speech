import cv2
import numpy as np
import time
from gtts import gTTS
import pygame

# def speak(text):
#     tts = gTTS(text=text, lang='en')
#     tts.save("output.mp3")
#     pygame.mixer.init()
#     pygame.mixer.music.load("output.mp3")
#     pygame.mixer.music.play()
#     while pygame.mixer.music.get_busy():
#         pygame.time.Clock().tick(10)

def detect_objects(classNames):
    cap = cv2.VideoCapture(0)  # For Webcam
    cap.set(3, 1280)
    cap.set(4, 720)

    # Load MobileNet SSD model and its configuration
    prototxt_path = '../MobileNetSSD-Weights/MobileNetSSD_deploy.prototxt.txt'
    caffemodel_path = '../MobileNetSSD-Weights/MobileNetSSD_deploy.caffemodel'

    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        new_frame_time = time.time()
        success, img = cap.read()
        (h, w) = img.shape[:2]

        # Preprocess the input image for object detection
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)

        # Set the blob as input to the network
        net.setInput(blob)

        # Run forward pass to get output predictions
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Filter out weak detections
                class_id = int(detections[0, 0, i, 1])
                label = classNames[class_id]

                # Draw bounding box and label on the image
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Speak out the detected object
                # speak(f'I see a {label}')

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        print("FPS:", fps)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

classNames = ["background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
              "sofa", "train", "tvmonitor"]

detect_objects(classNames)
  