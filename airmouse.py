import cv2
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

pyautogui.FAILSAFE = False

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

 
    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            height, width, _ = frame.shape
            cx, cy = int(index_finger_landmark.x * width), int(index_finger_landmark.y * height)
            screenWidth, screenHeight = pyautogui.size()
            pyautogui.moveTo(cx * screenWidth / width, cy * screenHeight / height)

            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
