import cv2
import time
from gesture_detector import GestureDetector

def main():
    cap = cv2.VideoCapture(0)
    detector = GestureDetector()
    
    last_gesture = None
    p_time = 0

    print("--- Gesture Recognition Started ---")
    print("Gestures: up, down, left, right, off (open), on (closed)")
    print("Press 'q' to quit.")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        # Flip the image horizontally for a later selfie-view display
        img = cv2.flip(img, 1)
        
        img = detector.find_hands(img)
        lm_list = detector.get_landmarks(img)
        
        gesture = detector.classify_gesture(lm_list)
        
        if gesture and gesture != last_gesture:
            print(f"Detected: {gesture}")
            last_gesture = gesture

        # FPS Calculation
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        if gesture:
            cv2.putText(img, f"Gesture: {gesture}", (10, 120), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow("Gesture Recognition", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
