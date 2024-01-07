import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2,
                      min_tracking_confidence=0.5, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

prevTime = 0
currTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLnmrks in results.multi_hand_landmarks:
            for id, lm in enumerate(handLnmrks.landmark):
                height, width, channels = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                print(id, cx, cy)
                if id == 4:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLnmrks, mpHands.HAND_CONNECTIONS)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Cam", img)
    cv2.waitKey(1)
