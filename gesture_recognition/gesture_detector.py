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

        # Helper function to calculate euclidean distance
        def get_dist(p1, p2):
            return math.hypot(p1[1] - p2[1], p1[2] - p2[2])

        # Finger Landmarks
        # Thumb: 1-4, Index: 5-8, Middle: 9-12, Ring: 13-16, Pinky: 17-20
        # Wrist: 0
        
        wrist = lm_list[0]
        
        # Determine which fingers are open
        fingers_open = []
        
        # Thumb: Harder to define "open" via distance to wrist alone.
        # We check if the tip is far from the Index Finger MCP (Metacarpophalangeal joint, id 5).
        # Adjust threshold as needed, or compare to the length of the thumb itself.
        # A robust way is to check if the Tip is "outside" the palm compared to the IP joint.
        # For simplicity in this quick improvement:
        # Check if Thumb Tip is farther from Pinky MCP (17) than Thumb IP (3) is.
        # This implies abduction.
        thumb_tip = lm_list[4]
        thumb_ip = lm_list[3]
        pinky_mcp = lm_list[17]
        
        if get_dist(thumb_tip, pinky_mcp) > get_dist(thumb_ip, pinky_mcp):
             fingers_open.append(True)
        else:
             fingers_open.append(False)

        # Other 4 fingers: 
        # Open if distance(Wrist, Tip) > distance(Wrist, PIP)
        # We use PIP (Proximal Interphalangeal) joints: 6, 10, 14, 18
        for tip_id, pip_id in zip([8, 12, 16, 20], [6, 10, 14, 18]):
            if get_dist(lm_list[0], lm_list[tip_id]) > get_dist(lm_list[0], lm_list[pip_id]):
                fingers_open.append(True)
            else:
                fingers_open.append(False)

        total_open = fingers_open.count(True)

        # 1. Open Hand (All 5 open)
        if total_open == 5:
            return "off"
        
        # 2. Closed Hand (0 open)
        if total_open == 0:
            return "on"

        # 3. Directional Gestures (Only Index Open)
        # Note: Sometimes thumb might be naturally loose, so we strictly check
        # if Index is OPEN and Middle, Ring, Pinky are CLOSED.
        # We can be lenient with the thumb or force it closed.
        # Let's check: Index Open, Middle+Ring+Pinky Closed.
        if fingers_open[1] and not any(fingers_open[2:]):
            # Thumb state doesn't strictly matter for pointing, but user requested "Index ... others closed"
            # If we enforce thumb closed, it might be hard for some users. 
            # Let's enforce it loosely or ignore it.
            # "index extended, others closed" usually implies thumb is tucked or neutral.
            # Let's enforce thumb closed for STRICT definition, or just ignore it to be easier.
            # User prompt: "index up= up... open hand = off"
            # Let's try to enforce thumb closed if possible, but maybe allow it if it's not super wide.
            # For now, let's rely on Index Open + Others Closed.
            
            # Determine direction using Index Tip vs Index PIP (or MCP)
            indx_tip = lm_list[8]
            indx_mcp = lm_list[5] # MCP is more stable base than PIP for overall direction
            
            dx = indx_tip[1] - indx_mcp[1]
            dy = indx_tip[2] - indx_mcp[2]
            
            # Use a threshold to avoid jitter when diagonal
            if abs(dx) > abs(dy):
                # Horizontal
                return "right" if dx > 0 else "left"
            else:
                # Vertical
                return "down" if dy > 0 else "up"

        return None
