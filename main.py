from cProfile import label
import cv2
import mediapipe as mp
import numpy as np

# ------------------ MediaPipe Setup ------------------
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# ------------------ Face Landmark Indexes ------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
UPPER_LIP = 13
LOWER_LIP = 14
NOSE_TIP = 1

# ------------------ Hand Landmark Indexes ------------------
WRIST = 0
THUMB = [1, 2, 3, 4]
INDEX = [5, 6, 7, 8]
MIDDLE = [9, 10, 11, 12]
RING = [13, 14, 15, 16]
PINKY = [17, 18, 19, 20]

STABLE_LANDMARKS = [
    1, 33, 133, 362, 263, 61, 291, 199, 234, 454
]

# ------------------ Colors (BGR) ------------------
COLOR_ALL = (120, 120, 120) #gray
COLOR_STABLE = (0, 0, 255) #red
COLOR_EYES = (255, 0, 0) #blue
COLOR_MOUTH = (0, 255, 255) #yellow

HAND_LANDMARK_COLOR = (0, 0, 255) #red
HAND_CONNECTION_COLOR = (0, 200, 0) #dark green

# ------------------ Helpers ------------------
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(eye, pts):
    A = euclidean(pts[eye[1]], pts[eye[5]])
    B = euclidean(pts[eye[2]], pts[eye[4]])
    C = euclidean(pts[eye[0]], pts[eye[3]])
    return (A + B) / (2.0 * C)

def normalize_hand(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    wrist = pts[0]
    pts -= wrist

    scale = np.linalg.norm(pts[9])  # middle finger MCP
    pts /= (scale + 1e-6)

    return pts

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_BASES = [5, 6, 10, 14, 18]

def finger_states(hand):
    states = []

    # ---------- THUMB (special logic) ----------
    thumb_tip = hand[4]

    # landmarks to compare against
    other_points = [5, 9, 13, 17, 8, 12, 16, 20]

    dists = [
        np.linalg.norm(thumb_tip - hand[i])
        for i in other_points
    ]

    THUMB_CLOSE_THRESH = 0.35  # tune if needed

    thumb_open = min(dists) > THUMB_CLOSE_THRESH
    states.append("O" if thumb_open else "C")

    # ---------- OTHER FINGERS (normal logic) ----------
    for tip, base in zip(FINGER_TIPS[1:], FINGER_BASES[1:]):
        open_ = np.linalg.norm(hand[tip]) > np.linalg.norm(hand[base])
        states.append("O" if open_ else "C")

    return states

def angle(a, b, c):
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


# ------------------ Blink State ------------------
BLINK_THRESHOLD = 0.10
blink_frames = 0

# ------------------ Main Loop ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_result = face_mesh.process(rgb)
    hand_result = hands.process(rgb)

    mesh_canvas = np.zeros_like(frame)

    # ================== FACE ==================
    if face_result.multi_face_landmarks:
        face_landmarks = face_result.multi_face_landmarks[0]
        points = []

        for idx, lm in enumerate(face_landmarks.landmark):
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y))

            color = COLOR_ALL
            r = 1

            if idx in STABLE_LANDMARKS:
                color = COLOR_STABLE
                r = 2
            if idx in LEFT_EYE or idx in RIGHT_EYE:
                color = COLOR_EYES
                r = 2
            if idx in [UPPER_LIP, LOWER_LIP]:
                color = COLOR_MOUTH
                r = 3

            cv2.circle(mesh_canvas, (x, y), r, color, -1)

        # Blink detection
        ear = (
            eye_aspect_ratio(LEFT_EYE, points) +
            eye_aspect_ratio(RIGHT_EYE, points)
        ) / 2

        blink = False
        if ear < BLINK_THRESHOLD:
            blink_frames += 1
        else:
            if blink_frames >= 2:
                blink = True
            blink_frames = 0

        # Mouth detection
        mouth_ratio = euclidean(points[UPPER_LIP], points[LOWER_LIP]) / \
                      euclidean(points[33], points[263])

        mouth_open = mouth_ratio > 0.20
        speaking = 0.015 < mouth_ratio < 0.08

        # Head direction
        nx, ny = points[NOSE_TIP]
        cx, cy = w // 2, h // 2

        direction = "Center"
        if nx < cx - 40:
            direction = "Looking Left"
        elif nx > cx + 40:
            direction = "Looking Right"
        elif ny < cy - 40:
            direction = "Looking Up"
        elif ny > cy + 40:
            direction = "Looking Down"
        # ================== HANDS → mesh_canvas ==================
        multi_hands = {}  # default state
        if hand_result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(hand_result.multi_hand_landmarks, hand_result.multi_handedness):  
                label = handedness.classification[0].label
                mp_draw.draw_landmarks(
                    mesh_canvas,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=HAND_LANDMARK_COLOR, thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=HAND_CONNECTION_COLOR, thickness=2)
                )
                hand = normalize_hand(hand_landmarks.landmark)
                multi_hands[label] = finger_states(hand)

        # UI (on frame)
        cv2.putText(frame, f"Blink: {blink}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"Mouth Open: {mouth_open}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"Speaking: {speaking}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, direction, (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        y = 160
        for hand_label, gesture in multi_hands.items():
            cv2.putText(
                frame,
                f"{hand_label} Hand: {gesture}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            y += 30

        # ================== DISPLAY ==================
        combined = np.hstack((frame, mesh_canvas))
        cv2.imshow("Face + Hand Mesh (Canvas)", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
