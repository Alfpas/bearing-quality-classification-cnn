from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os
import threading
import time
from datetime import datetime, timedelta
import csv
import json
import pandas as pd
from collections import defaultdict
import io

app = Flask(__name__)
app.config['SECRET_KEY'] = 'industrial-bearing-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Load model
model = YOLO('model/best.pt')

# Database sederhana (gunakan file JSON untuk penyimpanan)
DATA_FILE = 'inspection_log.json'

# Warna untuk bounding box - warna industrial
COLORS = {
    'no_bearing': (128, 128, 128),
    'small': (0, 255, 0),      # Hijau - OK
    'medium': (0, 165, 255),    # Orange - Warning
    'large': (0, 0, 255)        # Merah - Reject
}

# Kelas dan status produksi
CLASS_STATUS = {
    'small': 'OK',
    'medium': 'WARNING',
    'large': 'REJECT',
    'no_bearing': 'NO_PRODUCT'
}

# Data inspeksi
inspection_log = []
production_target = {
    'daily': 1000,
    'hourly': 125
}
current_shift = 'Morning'  # Morning, Afternoon, Night
shift_hours = {
    'Morning': (6, 14),
    'Afternoon': (14, 22),
    'Night': (22, 6)
}

# ===== INI YANG DITAMBAHKAN =====
# Variabel global untuk kamera
camera = None
camera_active = False
camera_index = 1
# ================================

# Load existing data
def load_data():
    global inspection_log
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                inspection_log = json.load(f)
        except:
            inspection_log = []
    else:
        inspection_log = []

def save_data():
    with open(DATA_FILE, 'w') as f:
        json.dump(inspection_log, f, indent=2)

load_data()

# Helper functions
def get_current_shift():
    hour = datetime.now().hour
    for shift, (start, end) in shift_hours.items():
        if start <= end:
            if start <= hour < end:
                return shift
        else:  # Night shift crosses midnight
            if hour >= start or hour < end:
                return shift
    return 'Morning'

def calculate_statistics():
    today = datetime.now().date()
    today_logs = [log for log in inspection_log if datetime.fromisoformat(log['timestamp']).date() == today]
    
    total = len(today_logs)
    small_count = sum(1 for log in today_logs if log['class'] == 'small')
    medium_count = sum(1 for log in today_logs if log['class'] == 'medium')
    large_count = sum(1 for log in today_logs if log['class'] == 'large')
    no_bearing_count = sum(1 for log in today_logs if log['class'] == 'no_bearing')
    
    reject_count = large_count
    warning_count = medium_count
    ok_count = small_count
    
    return {
        'total': total,
        'small': small_count,
        'medium': medium_count,
        'large': large_count,
        'no_bearing': no_bearing_count,
        'ok': ok_count,
        'warning': warning_count,
        'reject': reject_count,
        'defect_rate': (reject_count / total * 100) if total > 0 else 0,
        'target_achievement': (total / production_target['daily'] * 100) if production_target['daily'] > 0 else 0
    }

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dashboard')
def dashboard_data():
    stats = calculate_statistics()
    stats['shift'] = get_current_shift()
    stats['target'] = production_target
    stats['recent_logs'] = inspection_log[-20:]  # Last 20 inspections
    return jsonify(stats)

@app.route('/api/logs')
def get_logs():
    limit = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)
    class_filter = request.args.get('class', None)
    date_from = request.args.get('date_from', None)
    date_to = request.args.get('date_to', None)
    
    logs = inspection_log[::-1]  # Reverse, newest first
    
    if class_filter:
        logs = [l for l in logs if l['class'] == class_filter]
    if date_from:
        logs = [l for l in logs if l['timestamp'] >= date_from]
    if date_to:
        logs = [l for l in logs if l['timestamp'] <= date_to]
    
    return jsonify({
        'total': len(logs),
        'logs': logs[offset:offset+limit]
    })

@app.route('/api/export', methods=['POST'])
def export_data():
    format_type = request.json.get('format', 'csv')
    start_date = request.json.get('start_date', None)
    end_date = request.json.get('end_date', None)
    
    logs = inspection_log
    
    if start_date:
        logs = [l for l in logs if l['timestamp'] >= start_date]
    if end_date:
        logs = [l for l in logs if l['timestamp'] <= end_date]
    
    if format_type == 'csv':
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=['timestamp', 'class', 'confidence', 'shift', 'image_name'])
        writer.writeheader()
        writer.writerows(logs)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'inspection_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    else:
        return jsonify(logs)

@app.route('/detect', methods=['POST'])
def detect_upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file'}), 400
    
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = model.predict(img, conf=0.4, verbose=False)
    
    detections = []
    annotated_img = img.copy()
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                detections.append({
                    'class': class_name,
                    'confidence': round(confidence * 100, 1),
                    'bbox': [x1, y1, x2, y2],
                    'status': CLASS_STATUS.get(class_name, 'UNKNOWN')
                })
                
                color = COLORS.get(class_name, (255, 255, 255))
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
                
                label = f"{class_name}: {confidence*100:.1f}% | {CLASS_STATUS.get(class_name, '')}"
                cv2.putText(annotated_img, label, (x1, y1-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'success': True,
        'detections': detections,
        'total': len(detections),
        'image': img_base64
    })

@socketio.on('inspection_result')
def handle_inspection(data):
    """Simpan hasil inspeksi"""
    global current_shift
    current_shift = get_current_shift()
    
    inspection_entry = {
        'timestamp': datetime.now().isoformat(),
        'class': data['class'],
        'confidence': data['confidence'],
        'shift': current_shift,
        'image_name': data.get('image_name', 'live_feed')
    }
    
    inspection_log.append(inspection_entry)
    save_data()
    
    # Update dashboard
    emit('dashboard_update', calculate_statistics())

@socketio.on('start_camera')
def handle_start_camera(data):
    global camera, camera_active, camera_index
    
    camera_index = data.get('index', 1)
    
    # Close existing camera if any
    if camera is not None:
        camera.release()
    
    try:
        camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera = cv2.VideoCapture(camera_index)
        
        if camera.isOpened():
            camera_active = True
            emit('camera_status', {'status': 'started', 'index': camera_index})
            start_streaming()
        else:
            emit('camera_status', {'status': 'error', 'message': f'Kamera USB index {camera_index} tidak ditemukan'})
    except Exception as e:
        emit('camera_status', {'status': 'error', 'message': str(e)})

@socketio.on('stop_camera')
def handle_stop_camera():
    global camera, camera_active
    camera_active = False
    if camera is not None:
        camera.release()
        camera = None
    emit('camera_status', {'status': 'stopped'})

@socketio.on('test_cameras')
def handle_test_cameras():
    available = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available.append({'index': i, 'name': f'Camera {i}'})
            cap.release()
    emit('cameras_available', {'cameras': available})

def start_streaming():
    global camera, camera_active
    
    def stream():
        while camera_active and camera is not None and camera.isOpened():
            ret, frame = camera.read()
            if not ret:
                break
            
            results = model.predict(frame, conf=0.4, verbose=False)
            
            annotated_frame = frame.copy()
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        detections.append({
                            'class': class_name,
                            'confidence': round(confidence * 100, 1),
                            'status': CLASS_STATUS.get(class_name, 'UNKNOWN')
                        })
                        
                        color = COLORS.get(class_name, (255, 255, 255))
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(annotated_frame, f"{class_name}", (x1, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add production info overlay
            stats = calculate_statistics()
            overlay_info = [
                f"Production: {stats['total']} / {production_target['daily']}",
                f"Defect Rate: {stats['defect_rate']:.1f}%",
                f"Shift: {current_shift}"
            ]
            
            for i, info in enumerate(overlay_info):
                cv2.putText(annotated_frame, info, (10, 30 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            socketio.emit('frame', {
                'image': frame_base64,
                'detections': detections,
                'total': len(detections)
            })
            
            # Auto-save if defect detected
            if any(d['class'] == 'large' for d in detections):
                socketio.emit('defect_alert', {
                    'message': f'⚠️ DEFECT DETECTED! Large bearing found',
                    'timestamp': datetime.now().isoformat()
                })
            
            time.sleep(0.03)
        
        if camera:
            camera.release()
    
    threading.Thread(target=stream, daemon=True).start()

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)