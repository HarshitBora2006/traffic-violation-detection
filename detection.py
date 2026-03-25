latest_violations = []

import os, cv2, json, uuid, time
from ultralytics import YOLO
import pytesseract
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# MODELS
vehicle_model = YOLO("yolov8n.pt")
person_model = YOLO("yolov8n.pt")
helmet_model = YOLO("best.pt")
plate_model  = YOLO("plate.pt")

# ✅ FILTER ONLY VEHICLES
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

VIOLATION_FILE = "violations.json"
VIOLATION_FOLDER = "static/violations"
os.makedirs(VIOLATION_FOLDER, exist_ok=True)


# ---------- LOAD ----------
def load_violations():
    if not os.path.exists(VIOLATION_FILE):
        return []
    try:
        with open(VIOLATION_FILE, "r") as f:
            return json.load(f)
    except:
        return []


# ---------- SAVE ----------
def save_violation(clean_frame, vehicle_name, plate, reason, bbox):
    x1, y1, x2, y2 = bbox

    crop = clean_frame[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return

    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(VIOLATION_FOLDER, filename)

    cv2.imwrite(path, crop)

    data = load_violations()

    record = {
        "image": path.replace("\\", "/"),
        "vehicle": vehicle_name,
        "plate": plate,
        "reason": reason,
        "time": datetime.now().strftime("%d %b %Y, %I:%M %p")
    }

    data.insert(0, record)
    data = data[:500]

    with open(VIOLATION_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ---------- LOGIC ----------
def is_rider(person_box, bike_box):
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box

    overlap = min(px2, bx2) - max(px1, bx1)
    width_ratio = overlap / (px2 - px1 + 1e-5)

    return width_ratio > 0.5


def detect_helmet(head_crop):
    if head_crop is None or head_crop.size == 0:
        return False
    results = helmet_model(head_crop, conf=0.4)
    return len(results[0].boxes) > 0


def detect_number_plate(vehicle_crop):
    if vehicle_crop is None or vehicle_crop.size == 0:
        return "Unknown"

    results = plate_model(vehicle_crop, conf=0.3)
    if len(results[0].boxes) == 0:
        return "Unknown"

    x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[0])
    plate_crop = vehicle_crop[y1:y2, x1:x2]

    text = pytesseract.image_to_string(plate_crop, config='--psm 8')
    return text.strip()


# ---------- PROCESS FRAME ----------
def process_frame(frame):
    clean_frame = frame.copy()
    display_frame = frame.copy()

    results = vehicle_model(display_frame, conf=0.4)
    person_results = person_model(display_frame, conf=0.4, classes=[0])

    boxes = results[0].boxes.xyxy.cpu().numpy()
    person_boxes = person_results[0].boxes.xyxy.cpu().numpy()

    frame_data = []

    for i, r in enumerate(boxes):
        x1, y1, x2, y2 = map(int, r)

        cls_id = int(results[0].boxes.cls[i])
        vehicle_name = vehicle_model.names[cls_id]

        # ✅ FILTER
        if vehicle_name not in VEHICLE_CLASSES:
            continue

        vehicle_crop = display_frame[y1:y2, x1:x2]

        riders = [p for p in person_boxes if is_rider(p, [x1,y1,x2,y2])]
        triple = "Yes" if len(riders) >= 3 else "No"

        helmet = "Yes"
        for rider in riders:
            rx1, ry1, rx2, ry2 = map(int, rider)
            h = ry2 - ry1
            head_crop = display_frame[ry1:ry1+int(0.25*h), rx1:rx2]

            if not detect_helmet(head_crop):
                helmet = "No"
                break

        plate = detect_number_plate(vehicle_crop)

        is_violation = (helmet == "No" or triple == "Yes")

        if is_violation:
            reason = "No Helmet" if helmet == "No" else "Triple Riding"
            save_violation(clean_frame, vehicle_name, plate, reason, (x1,y1,x2,y2))

        frame_data.append({
            "vehicle": vehicle_name,
            "helmet": helmet,
            "triple_riding": triple,
            "plate": plate,
            "status": "Violation" if is_violation else "Safe"
        })

        color = (0,0,255) if is_violation else (0,255,0)
        cv2.rectangle(display_frame, (x1,y1), (x2,y2), color, 3)

    return display_frame, frame_data


# ---------- IMAGE ----------
def process_image(file_path):
    img = cv2.imread(file_path)

    display, data = process_frame(img)

    filename = os.path.basename(file_path)
    out_path = os.path.join("static/uploads", "annotated_" + filename)

    cv2.imwrite(out_path, display)

    return "annotated_" + filename, data


# ---------- VIDEO ----------
def generate_video_stream(file_path):
    global latest_violations

    cap = cv2.VideoCapture(file_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (900, 550))

        display, data = process_frame(frame)

        latest_violations = data  # ✅ KEY FIX

        _, buffer = cv2.imencode('.jpg', display)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        time.sleep(0.1)

    cap.release()


# ---------- STREAM ----------
def generate_violation_stream():
    global latest_violations

    while True:
        yield f"data: {json.dumps(latest_violations)}\n\n"
        time.sleep(0.5)