import cv2
import mediapipe as mp
import math

class GestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def get_landmarks(self, img):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[0]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
        return lm_list

    def classify_gesture(self, lm_list):
        if not lm_list:
            return None

        # Finger states: 1 for extended, 0 for closed
        fingers = []

        # Thumb (Special case, depends on orientation; assuming right hand palm facing camera)
        # For simplicity, we compare x-coordinates
        if lm_list[self.tip_ids[0]][1] > lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other 4 fingers
        for id in range(1, 5):
            if lm_list[self.tip_ids[id]][2] < lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        total_fingers = fingers.count(1)

        # Open hand: All fingers extended (roughly)
        if total_fingers >= 4:
            return "off"
        
        # Closed hand: All fingers closed
        if total_fingers == 0:
            return "on"

        # Index gestures: Only index extended
        if fingers[1] == 1 and total_fingers == 1:
            index_tip = lm_list[8]
            index_pip = lm_list[6] # PIP or MCP (5)
            
            dx = index_tip[1] - index_pip[1]
            dy = index_tip[2] - index_pip[2]

            if abs(dy) > abs(dx):
                if dy < 0:
                    return "up"
                else:
                    return "down"
            else:
                if dx < 0:
                    return "left" # Note: depends on camera flip, usually left is negative X
                else:
                    return "right"

        return None
