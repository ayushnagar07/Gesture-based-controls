import cv2
import mediapipe as mp
import time
import webbrowser
import sys


def identify_index_finger(frame, hand_landmarks):
    if hand_landmarks:

        index_finger_landmarks = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

        height, width, _ = frame.shape
        cx, cy = int(index_finger_landmarks.x * width), int(index_finger_landmarks.y * height)

        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        return cx, cy
    return None, None


def main():
    
    mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    
    cap = cv2.VideoCapture(0)

    finger_detected_time = None
    youtube_url = 'https://www.youtube.com'

    while cap.isOpened():
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = mp_hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)


                cx, cy = identify_index_finger(frame, hand_landmarks)
                if cx is not None and cy is not None:
                    
                    if finger_detected_time is None:
                        finger_detected_time = time.time()
                    else:
                    
                        if time.time() - finger_detected_time > 5:
                            webbrowser.open(youtube_url)
                            sys.exit()  
                else:
                    finger_detected_time = None

        cv2.imshow('Hand Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
