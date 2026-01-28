import cv2, time
import mediapipe as mp
import numpy as np
import pyautogui as pag

# ================== PyAutoGUI ==================
pag.FAILSAFE = False
sw, sh = pag.size()

# ================== MediaPipe ==================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ================== Hand Indexes ==================
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_BASES = [5, 6, 10, 14, 18]


HAND_LANDMARK_COLOR = (0, 0, 255)
HAND_CONNECTION_COLOR = (0, 200, 0)

# ================== Gesture Thresholds ==================
PINCH_THRESH = 0.20
FINGER_CLOSE_THRESH = 0.25

# ================== Debounce ==================
last_click = 0
CLICK_DELAY = 0

last_scroll = 0
SCROLL_DELAY = 0.15

# ================== Advanced Smoothing ==================
ema_x, ema_y = sw // 2, sh // 2
EMA_ALPHA = 0.12        # lower = smoother (0.1–0.2 sweet spot)

DEADZONE = 8            # pixels (ignore micro jitter)

last_cursor_update = 0
CURSOR_INTERVAL = 1 / 120   # max 120 updates/sec


def euclidean(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def normalize_hand(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    scale = np.linalg.norm(pts[9])
    return pts / (scale + 1e-6)

def finger_states(hand):
    states = []

    # Thumb logic
    thumb_tip = hand[4]
    refs = [5, 9, 13, 17, 8, 12, 16, 20]
    dists = [np.linalg.norm(thumb_tip - hand[i]) for i in refs]
    states.append("O" if min(dists) > 0.35 else "C")

    # Other fingers
    for tip, base in zip(FINGER_TIPS[1:], FINGER_BASES[1:]):
        states.append("O" if np.linalg.norm(hand[tip]) > np.linalg.norm(hand[base]) else "C")

    return states

# ================== Cursor Smoothing ==================
prev_x, prev_y = sw // 2, sh // 2
SMOOTHING = 0.35

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mesh_canvas = np.zeros_like(frame)

    hand_result = hands.process(rgb)

    # ================== HANDS ==================
    if hand_result.multi_hand_landmarks:
        for hlm, handedness in zip(hand_result.multi_hand_landmarks, hand_result.multi_handedness):
            hand = normalize_hand(hlm.landmark)

            # ---------- Finger state checks ----------

            # CLOSED fingers
            ring_closed = np.linalg.norm(hand[15] - hand[13]) < FINGER_CLOSE_THRESH
            pinky_closed = np.linalg.norm(hand[19] - hand[17]) < FINGER_CLOSE_THRESH

            # OPEN fingers
            thumb_open  = np.linalg.norm(hand[4])  > np.linalg.norm(hand[2])
            index_open  = np.linalg.norm(hand[8])  > np.linalg.norm(hand[5])
            middle_open = np.linalg.norm(hand[12]) > np.linalg.norm(hand[9])

            # Cursor enable condition
            cursor_enabled = (
                ring_closed and
                pinky_closed and
                thumb_open and
                index_open and
                middle_open
            )

            # ---------- Pinch checks ----------
            index_pinch = np.linalg.norm(hand[4] - hand[8]) < PINCH_THRESH
            middle_pinch = np.linalg.norm(hand[4] - hand[12]) < PINCH_THRESH

            now = time.time()

            mp_draw.draw_landmarks(
                mesh_canvas,
                hlm,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=HAND_LANDMARK_COLOR, thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=HAND_CONNECTION_COLOR, thickness=2)
            )

            # Cursor control (thumb tip)
            if cursor_enabled:
                # print("Cursor Enabled")
                lm = hlm.landmark[4]
                raw_x = (1 - lm.x) * sw
                raw_y = lm.y * sh

                # cx = int(prev_x + SMOOTHING * (raw_x - prev_x))
                # cy = int(prev_y + SMOOTHING * (raw_y - prev_y))

                # pag.moveTo(cx, cy, duration=0)
                # prev_x, prev_y = cx, cy

                now_cursor = time.time()

                if now_cursor - last_cursor_update >= CURSOR_INTERVAL:
                    # ---------- EMA smoothing ----------
                    ema_x = ema_x + EMA_ALPHA * (raw_x - ema_x)
                    ema_y = ema_y + EMA_ALPHA * (raw_y - ema_y)

                    cx, cy = int(ema_x), int(ema_y)

                    # ---------- Deadzone ----------
                    if abs(cx - prev_x) > DEADZONE or abs(cy - prev_y) > DEADZONE:
                        pag.moveTo(cx, cy, duration=0)
                        prev_x, prev_y = cx, cy

                    last_cursor_update = now_cursor


                # ---------- CLICK ----------
                if index_pinch and now - last_click > CLICK_DELAY:
                    pag.click()
                    last_click = now

                # ---------- SCROLL ----------
                if middle_pinch and now - last_scroll > SCROLL_DELAY:
                    pag.scroll(-40)
                    last_scroll = now


    # ================== DISPLAY ==================
    combined = np.hstack((frame, mesh_canvas))
    if __name__ == "__main__":
        cv2.imshow("Hand Mesh (Logic Active)", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
