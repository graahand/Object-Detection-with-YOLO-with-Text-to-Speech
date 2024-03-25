import cv2
import cvzone
import math
import time
import torch
from ultralytics import YOLO
from gtts import gTTS
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Function to perform text-to-speech
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("../Audio/output.mp3")
    pygame.mixer.music.load("../Audio/output.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

def detect_objects(model, classNames):
    cap = cv2.VideoCapture(0)  # For Webcam
    cap.set(3, 1280)
    cap.set(4, 720)

    # Initialize variables to control TTS frequency
    last_detection_time = 0
    tts_interval = 5  # Minimum time interval between TTS calls (in seconds)

    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])
                object_detected = classNames[cls]

                cvzone.cornerRect(img, (x1, y1, w, h))
                print("Object Detected:", object_detected)

                # Check if enough time has passed since the last TTS call
                current_time = time.time()
                if current_time - last_detection_time >= tts_interval:
                    speak(f'Detected Object Name is {object_detected}')  # Perform TTS
                    last_detection_time = current_time  # Update last detection time

                cvzone.putTextRect(img, f'{object_detected} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo_path = '../YOLO-Weights/yolov5nu.pt'
    model = YOLO(yolo_path).to(device)
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
                  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
                  "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
                  "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]

    detect_objects(model, classNames)
