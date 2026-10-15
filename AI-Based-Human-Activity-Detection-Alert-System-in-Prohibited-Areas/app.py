import os
import threading
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from camera import VideoCamera

app = Flask(__name__)

# ==========================================
# 🛑 DIRECTORY SETUP
# ==========================================
AUTH_FACES_DIR = "static/authorized_faces"
LOGS_DIR = "static/logs"
AUTH_LOGS_DIR = os.path.join(LOGS_DIR, "authorized")
UNAUTH_LOGS_DIR = os.path.join(LOGS_DIR, "unauthorized")

os.makedirs(AUTH_FACES_DIR, exist_ok=True)
os.makedirs(AUTH_LOGS_DIR, exist_ok=True)
os.makedirs(UNAUTH_LOGS_DIR, exist_ok=True)

# Global dictionary to store live camera streams (2 Cameras)
video_streams = {
    1: None,
    2: None
}

def generate_frames(cam):
    while True:
        if cam is None or getattr(cam, 'stopped', True):
            break
        frame = cam.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# ==========================================
# 🌐 MAIN ROUTE
# ==========================================
@app.route('/')
def index():
    active_cams = [cid for cid, cam in video_streams.items() if cam is not None and not cam.stopped]
    camera_active = len(active_cams) > 0
    return render_template('index.html', camera_active=camera_active, active_cams=active_cams)

# ==========================================
# ⚙️ MULTI-CAMERA CONTROL ROUTES
# ==========================================
@app.route('/start_camera', methods=['POST'])
def start_camera():
    ip_inputs = {
        1: request.form.get('camera_ip_1', '').strip(),
        2: request.form.get('camera_ip_2', '').strip()
    }

    for cam_id, url in ip_inputs.items():
        # Stop existing stream if running
        if video_streams[cam_id] is not None:
            video_streams[cam_id].stop()
            video_streams[cam_id] = None
        
        # Start new stream if URL is provided
        if url:
            # If user typed just an IP without http/rtsp prefix, prepend http://
            if not url.startswith(('http://', 'https://', 'rtsp://')):
                url = f"http://{url}"
            try:
                print(f"[INFO] Starting Cam {cam_id} at {url}...")
                video_streams[cam_id] = VideoCamera(url, cam_id)
            except Exception as e:
                print(f"[ERROR] Failed to start Cam {cam_id}: {e}")

    return redirect(url_for('index'))

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    for cam_id in video_streams:
        if video_streams[cam_id] is not None:
            video_streams[cam_id].stop()
            video_streams[cam_id] = None
    print("[INFO] All cameras stopped.")
    return redirect(url_for('index'))

# ==========================================
# 📺 LIVE VIDEO STREAM ENDPOINT
# ==========================================
@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    cam = video_streams.get(cam_id)
    if cam is None or cam.stopped:
        return "Camera Offline", 404
    return Response(generate_frames(cam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ==========================================
# 👤 AUTHORIZED PERSON REGISTRATION
# ==========================================
@app.route('/register_person', methods=['POST'])
def register_person():
    auth_name = request.form.get('auth_name', '').strip().replace(' ', '_')
    auth_id = request.form.get('auth_id', '').strip()
    files = request.files.getlist('auth_images')

    if not auth_name or not files:
        return redirect(url_for('index'))

    saved_count = 0
    for idx, file in enumerate(files):
        if file and file.filename != '':
            ext = os.path.splitext(file.filename)[1]
            if ext.lower() in ['.jpg', '.jpeg', '.png']:
                filename = f"{auth_name}__{auth_id}_{idx+1}{ext}"
                filepath = os.path.join(AUTH_FACES_DIR, filename)
                file.save(filepath)
                saved_count += 1

    print(f"[INFO] Registered {auth_name} with {saved_count} face images.")

    # Hot-reload known faces into active camera streams without restarting them
    for cam in video_streams.values():
        if cam is not None and not cam.stopped:
            threading.Thread(target=cam.load_authorized_faces, args=(AUTH_FACES_DIR,), daemon=True).start()

    return redirect(url_for('index'))

# ==========================================
# 📊 REAL-TIME DASHBOARD API & LOGS
# ==========================================
@app.route('/get_data')
def get_data():
    # Fetch separated authorized and unauthorized images
    auth_logs = sorted(os.listdir(AUTH_LOGS_DIR), reverse=True)[:10] if os.path.exists(AUTH_LOGS_DIR) else []
    unauth_logs = sorted(os.listdir(UNAUTH_LOGS_DIR), reverse=True)[:10] if os.path.exists(UNAUTH_LOGS_DIR) else []
    
    total_humans = 0
    total_auth = 0
    total_unauth = 0
    accuracy_sum = 0
    active_cams_count = 0
    
    for cam_id, cam in video_streams.items():
        if cam is not None and not getattr(cam, 'stopped', True):
            total_humans += getattr(cam, 'person_count', 0)
            total_auth += getattr(cam, 'auth_count', 0)
            total_unauth += getattr(cam, 'unauth_count', 0)
            accuracy_sum += getattr(cam, 'accuracy_percent', 100.0)
            active_cams_count += 1
            
    avg_accuracy = (accuracy_sum / active_cams_count) if active_cams_count > 0 else 100.0

    return jsonify({
        'auth_logs': auth_logs,
        'unauth_logs': unauth_logs,
        'total_humans': total_humans,
        'total_auth': total_auth,
        'total_unauth': total_unauth,
        'accuracy': round(avg_accuracy, 1)
    })

# ==========================================
# 🚀 SERVER START
# ==========================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)