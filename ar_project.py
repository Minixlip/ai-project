import cv2
import numpy as np
from ultralytics import YOLO
import threading
from collections import deque, Counter
import mediapipe as mp

# --- CONFIGURATION ---
model = YOLO('yolov8n.pt')

# Initialize MediaPipe Face Mesh (The Geometric Tracker)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- GLOBAL VARIABLES ---
clicked_coords = None
input_lock = threading.Lock()
current_moods = {}          
mood_histories = {}         
privacy_states = {}         
is_analyzing = False 

# --- HELPER FUNCTIONS ---

def mouse_callback(event, x, y, flags, param):
    global clicked_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_coords = (x, y)

def analyze_mood_geometry(frame_copy, x1, y1, x2, y2, dict_key):
    """
    GEOMETRIC MOOD ANALYSIS (v3: Center-Relative Math)
    Fixes 'Sad' detection by comparing corners to the MOUTH CENTER,
    not the lips (which deform during pouting).
    """
    global is_analyzing, current_moods, mood_histories
    try:
        person_roi = frame_copy[y1:y2, x1:x2]
        if person_roi.size == 0: return

        roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(roi_rgb)

        raw_mood = "Neutral"

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # --- GEOMETRY CALCULATION ---
            
            # 1. Get Coordinates
            left_corner_y = landmarks[61].y
            right_corner_y = landmarks[291].y
            upper_lip_y = landmarks[13].y
            lower_lip_y = landmarks[14].y
            
            # 2. Calculate Centers
            mouth_corners_avg_y = (left_corner_y + right_corner_y) / 2
            mouth_center_y = (upper_lip_y + lower_lip_y) / 2
            
            # 3. Calculate "Smile Curve"
            # Positive (+) = Corners are LOWER than center (Frown/Sad)
            # Negative (-) = Corners are HIGHER than center (Smile/Happy)
            smile_curve = mouth_corners_avg_y - mouth_center_y

            # 4. Eyebrow Lift (Inner brow vs Mid brow)
            # Positive = Inner brow is raised (Sad/Puppy eyes)
            left_brow_lift = landmarks[105].y - landmarks[70].y
            right_brow_lift = landmarks[334].y - landmarks[300].y
            avg_brow_lift = (left_brow_lift + right_brow_lift) / 2
            
            # 5. Mouth Openness
            mouth_height = lower_lip_y - upper_lip_y

            # --- LOGIC GATES ---

            # 1. SURPRISE (Mouth Wide Open)
            if mouth_height > 0.035:
                raw_mood = "Surprised"

            # 2. HAPPY (Corners significantly ABOVE center)
            elif smile_curve < -0.01:
                raw_mood = "Happy"

            # 3. SAD (Corners BELOW center OR Brow Lift)
            # We set the threshold to 0.005 (Tiny downward curve triggers Sad)
            # OR if eyebrows are pulled up (Puppy dog eyes)
            elif (smile_curve > 0.005) or (avg_brow_lift > 0.01):
                raw_mood = "Sad"

            # 4. NEUTRAL (Flat line)
            else:
                raw_mood = "Neutral"

        # SMOOTHING
        with input_lock:
            if dict_key not in mood_histories:
                mood_histories[dict_key] = deque(maxlen=4) 
            
            mood_histories[dict_key].append(raw_mood)
            final_mood = Counter(mood_histories[dict_key]).most_common(1)[0][0]
            current_moods[dict_key] = final_mood

    except Exception as e:
        print(f"Error: {e}")
        pass
    finally:
        is_analyzing = False

def calculate_ui_positions(x1, y1):
    if y1 < 60:
        label_y = y1 + 25
        btn_y1 = y1 + 5
        btn_y2 = y1 + 35
    else:
        label_y = y1 - 10
        btn_y1 = y1 - 35
        btn_y2 = y1 - 5
    btn_x1 = x1 + 140
    btn_x2 = btn_x1 + 160
    return label_y, btn_x1, btn_y1, btn_x2, btn_y2

def draw_ar_overlay(frame, x1, y1, x2, y2, label, conf, privacy_active, mood_text):
    color = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    label_y, btn_x1, btn_y1, btn_x2, btn_y2 = calculate_ui_positions(x1, y1)
    
    label_text = f"{label.upper()} {int(conf*100)}%"
    cv2.putText(frame, label_text, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if label == 'person':
        if privacy_active:
            btn_color = (0, 0, 255)
            btn_text = "PRIVACY: ON"
        else:
            btn_color = (0, 255, 0)
            btn_text = "PRIVACY: OFF"
            
        cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), btn_color, -1)
        cv2.putText(frame, btn_text, (btn_x1 + 10, btn_y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if privacy_active:
            face_h = int((y2 - y1) * 0.7)
            if face_h > 0:
                roi = frame[y1:y1+face_h, x1:x2]
                blur = cv2.GaussianBlur(roi, (99, 99), 30)
                frame[y1:y1+face_h, x1:x2] = blur
                text_y = y1 + 60 if y1 < 60 else y1 + 50
                cv2.putText(frame, "PROTECTED", (x1 + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        elif mood_text:
            mood_y = y1 + 60 if y1 < 60 else y1 + 30
            cv2.putText(frame, f"Mood: {mood_text}", (x1 + 5, mood_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    else:
        info_y = y1 + 25 if y1 < 60 else y1 + 25
        cv2.putText(frame, "Object Detected", (x1 + 5, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

def main():
    global clicked_coords, is_analyzing, current_moods, privacy_states, mood_histories
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('AR Final Project')
    cv2.setMouseCallback('AR Final Project', mouse_callback)

    print("System Ready. Detects: Person, Cup, Laptop, Phone, Bottle, Chair.")
    print("Mode: GEOMETRIC ANALYSIS (Lighting Resistant)")
    
    allowed_objects = ['person', 'cup', 'cell phone', 'laptop', 'bottle', 'chair']

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        results = model(frame, stream=True, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])

                if label in allowed_objects and conf > 0.5:
                    if label == 'person':
                        _, btn_x1, btn_y1, btn_x2, btn_y2 = calculate_ui_positions(x1, y1)
                        person_id = "main_user" 
                        
                        if clicked_coords:
                            cx, cy = clicked_coords
                            if btn_x1 < cx < btn_x2 and btn_y1 < cy < btn_y2:
                                current_state = privacy_states.get(person_id, False)
                                privacy_states[person_id] = not current_state
                                clicked_coords = None 

                        is_privacy_on = privacy_states.get(person_id, False)

                        mood_str = ""
                        if not is_privacy_on:
                            mood_str = current_moods.get(person_id, "Detecting...")
                            if not is_analyzing:
                                is_analyzing = True
                                frame_copy = frame.copy()
                                # Call the NEW Geometric Analysis Function
                                t = threading.Thread(target=analyze_mood_geometry, args=(frame_copy, x1, y1, x2, y2, person_id))
                                t.daemon = True
                                t.start()
                        
                        frame = draw_ar_overlay(frame, x1, y1, x2, y2, label, conf, is_privacy_on, mood_str)
                    else:
                        frame = draw_ar_overlay(frame, x1, y1, x2, y2, label, conf, False, None)

        clicked_coords = None 
        cv2.imshow('AR Final Project', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()