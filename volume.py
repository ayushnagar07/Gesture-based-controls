import cv2
import mediapipe as mp
import numpy as np
import pyautogui  


cap = cv2.VideoCapture(0)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils


prev_finger_pos = None


SOUND_CHANGE_THRESHOLD = 20  
SOUND_CHANGE_AMOUNT = 5 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            height, width, _ = frame.shape

            x = int(index_finger_tip.x * width)
            y = int(index_finger_tip.y * height)

            if prev_finger_pos is None:
                prev_finger_pos = (x, y)

            finger_movement = y - prev_finger_pos[1]

            
            prev_finger_pos = (x, y)

            if abs(finger_movement) > SOUND_CHANGE_THRESHOLD:
                
                if finger_movement < 0: 
                    pyautogui.press('volumeup', presses=SOUND_CHANGE_AMOUNT)
                else: 
                    pyautogui.press('volumedown', presses=SOUND_CHANGE_AMOUNT)

    
    cv2.imshow('Hand Gesture Control', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
