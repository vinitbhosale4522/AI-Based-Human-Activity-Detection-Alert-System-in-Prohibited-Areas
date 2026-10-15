import cv2
import face_recognition
import numpy as np
import os
import threading
import time
import csv
from datetime import datetime
import winsound
from twilio.rest import Client
from ultralytics import YOLO
import torch

cv2.setNumThreads(1)

# ==========================================
# 🛑 CONFIGURATION: ENTER YOUR KEYS HERE 🛑
# ==========================================
TWILIO_SID = "YOUR_TWILIO_SID_HERE"
TWILIO_AUTH_TOKEN = "YOUR_TWILIO_AUTH_TOKEN_HERE"
TWILIO_PHONE_NUMBER = "YOUR_TWILIO_PHONE_NUMBER"     
TARGET_PHONE_NUMBER = "YOUR_TARGET_PHONE_NUMBER"     

global_last_alert_time = 0
alert_lock = threading.Lock()
csv_lock = threading.Lock()
ai_inference_lock = threading.Lock() 

# Directory Initialization for Separated Logging
AUTH_LOG_DIR = "static/logs/authorized"
UNAUTH_LOG_DIR = "static/logs/unauthorized"
CSV_FILE_PATH = "static/logs/detection_data.csv"

os.makedirs(AUTH_LOG_DIR, exist_ok=True)
os.makedirs(UNAUTH_LOG_DIR, exist_ok=True)

# Initialize CSV Log File
if not os.path.exists(CSV_FILE_PATH):
    with open(CSV_FILE_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Camera_ID", "Status", "Person_Name", "Humans_Detected", "Accuracy_Percent"])

print("Loading YOLOv8 Model into memory...")
global_yolo_model = YOLO('yolov8n.pt')
print("Model loaded successfully!")

def trigger_full_alert():
    try:
        winsound.Beep(1500, 1500)
    except Exception:
        pass

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message_body = f"SECURITY ALERT: Unauthorized person detected at {timestamp}!"

    if "YOUR_" not in TWILIO_SID:
        try:
            client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
            client.messages.create(
                body=message_body,
                from_=TWILIO_PHONE_NUMBER,
                to=TARGET_PHONE_NUMBER
            )
            print(f"✅ Alert SMS sent to {TARGET_PHONE_NUMBER}")
        except Exception as e:
            print(f"❌ Failed to send SMS: {e}")

def log_event_to_csv(cam_id, status, name, total_humans, accuracy):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with csv_lock:
        try:
            with open(CSV_FILE_PATH, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, f"Cam {cam_id}", status, name, total_humans, f"{accuracy:.1f}%"])
        except Exception as e:
            print(f"CSV Logging Error: {e}")

class VideoCamera(object):
    def __init__(self, camera_url, cam_id):
        self.cam_id = cam_id
        self.camera_url = camera_url
        self.video = None
        self.current_frame = None
        self.current_boxes = [] 
        
        # Real-Time Statistics
        self.person_count = 0  
        self.auth_count = 0
        self.unauth_count = 0
        self.accuracy_percent = 100.0
        
        self.last_log_time = 0
        self.log_cooldown = 5.0 # Log image snapshot every 5s per camera
        self.stopped = False

        self.known_face_encodings = []
        self.known_face_names = []
        self.load_authorized_faces('static/authorized_faces/')
        
        self.alert_cooldown = 15.0 

        self.cam_thread = threading.Thread(target=self.update_camera_stream, daemon=True)
        self.cam_thread.start()

        self.ai_thread = threading.Thread(target=self.process_ai, daemon=True)
        self.ai_thread.start()

    def load_authorized_faces(self, path):
        new_encodings = []
        new_names = []
        
        if not os.path.exists(path): return
            
        for filename in os.listdir(path):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(path, filename)
                image = face_recognition.load_image_file(img_path)
                try:
                    encoding = face_recognition.face_encodings(image)[0]
                    new_encodings.append(encoding)
                    
                    display_name = filename.split('__')[0].replace('_', ' ').title()
                    new_names.append(display_name)
                except:
                    pass
        
        self.known_face_encodings = new_encodings
        self.known_face_names = new_names

    def update_camera_stream(self):
        # 🟢 Use FFMPEG backend which handles DroidCam HTTP streams much better
        self.video = cv2.VideoCapture(self.camera_url, cv2.CAP_FFMPEG)
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while not self.stopped:
            if self.video is None or not self.video.isOpened():
                # Try reconnecting if connection dropped
                time.sleep(1.0)
                self.video = cv2.VideoCapture(self.camera_url, cv2.CAP_FFMPEG)
                continue

            success, frame = self.video.read()
            if success and frame is not None:
                height, width = frame.shape[:2]
                
                # Even-Pixel Resizing to prevent byte-alignment issues
                if width > 640:
                    scale = 640.0 / float(width)
                    new_width = 640
                    new_height = int(height * scale)
                    if new_height % 2 != 0:
                        new_height += 1
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                
                self.current_frame = frame
                time.sleep(0.01) 
            else:
                # If frame read fails, wait briefly so we don't freeze the CPU
                time.sleep(0.05)
                
    def process_ai(self):
        global global_last_alert_time
        last_ai_time = 0
        ai_interval = 0.50 
        
        while not self.stopped:
            if self.current_frame is None:
                time.sleep(0.02)
                continue

            current_time = time.time()
            if current_time - last_ai_time < ai_interval:
                time.sleep(0.02)
                continue

            ai_image = self.current_frame.copy()
            temp_boxes = []
            trigger_alert = False
            
            temp_auth = 0
            temp_unauth = 0
            detected_names = []

            with ai_inference_lock:
                with torch.no_grad():
                    results = global_yolo_model(ai_image, classes=[0], conf=0.30, imgsz=480, verbose=False)
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                        person_crop = ai_image[max(0, y1):y2, max(0, x1):x2]
                        
                        if person_crop.size == 0: continue

                        rgb_frame = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                        
                        if not face_locations:
                            name = "Unauthorized Intruder"
                            color = (0, 0, 255) 
                            trigger_alert = True
                            temp_unauth += 1 
                            detected_names.append(name)
                        else:
                            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                            name = "Unauthorized Intruder"
                            color = (0, 0, 255) 
                            trigger_alert = True
                            is_authorized = False

                            for face_encoding in face_encodings:
                                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.55)
                                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                                if len(face_distances) > 0:
                                    best_match_index = np.argmin(face_distances)
                                    if matches[best_match_index]:
                                        name = f"Authorized: {self.known_face_names[best_match_index]}"
                                        color = (0, 255, 0) 
                                        trigger_alert = False 
                                        is_authorized = True
                            
                            if is_authorized:
                                temp_auth += 1
                            else:
                                temp_unauth += 1
                            detected_names.append(name)
                        
                        temp_boxes.append((x1, y1, x2, y2, name, color))

            total_humans = len(temp_boxes)
            processed_faces = temp_auth + temp_unauth
            
            # Accuracy Ratio Calculation: Classified Faces vs Total Human Bounding Boxes
            if total_humans > 0:
                accuracy = (processed_faces / float(total_humans)) * 100.0
            else:
                accuracy = 100.0

            # Logging Logic: Separate Authorized & Unauthorized Folders + CSV
            now = time.time()
            if total_humans > 0 and (now - self.last_log_time > self.log_cooldown):
                self.last_log_time = now
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

                if temp_auth > 0:
                    for n in detected_names:
                        if "Authorized:" in n:
                            clean_n = n.replace("Authorized: ", "").replace(" ", "_")
                            auth_img_path = f"{AUTH_LOG_DIR}/auth_cam{self.cam_id}_{clean_n}_{timestamp_str}.jpg"
                            cv2.imwrite(auth_img_path, ai_image)
                            log_event_to_csv(self.cam_id, "Authorized", clean_n, total_humans, accuracy)

                if temp_unauth > 0:
                    unauth_img_path = f"{UNAUTH_LOG_DIR}/unauth_cam{self.cam_id}_{timestamp_str}.jpg"
                    cv2.imwrite(unauth_img_path, ai_image)
                    log_event_to_csv(self.cam_id, "Unauthorized", "Unknown", total_humans, accuracy)

            if trigger_alert:
                with alert_lock:
                    if now - global_last_alert_time > self.alert_cooldown:
                        global_last_alert_time = now
                        threading.Thread(target=trigger_full_alert, daemon=True).start()

            self.current_boxes = temp_boxes
            self.person_count = total_humans
            self.auth_count = temp_auth
            self.unauth_count = temp_unauth
            self.accuracy_percent = accuracy
            
            last_ai_time = time.time() 
            time.sleep(0.02)

    def stop(self):
        self.stopped = True
        
        if hasattr(self, 'cam_thread') and self.cam_thread.is_alive():
            self.cam_thread.join(timeout=1.0)
            
        if hasattr(self, 'ai_thread') and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=1.0)
            
        if self.video: 
            self.video.release()
            self.video = None

    def __del__(self):
        self.stop()

    def get_frame(self):
        if self.current_frame is None: 
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"CAM {self.cam_id} CONNECTING...", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, jpeg = cv2.imencode('.jpg', placeholder)
            return jpeg.tobytes()
            
        display_image = self.current_frame.copy()
        
        count_text = f"CAM {self.cam_id} | Humans: {self.person_count} | Auth: {self.auth_count} | Unauth: {self.unauth_count}"
        cv2.putText(display_image, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        for (x1, y1, x2, y2, name, color) in self.current_boxes:
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_image, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        ret, jpeg = cv2.imencode('.jpg', display_image)
        return jpeg.tobytes()