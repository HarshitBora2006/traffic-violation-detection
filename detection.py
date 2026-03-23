import os
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import json
from time import sleep
from datetime import datetime
import uuid

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

vehicle_model = YOLO("yolov8n.pt")
person_model = YOLO("yolov8n.pt")
helmet_model = YOLO("best.pt")
plate_model  = YOLO("plate.pt")

# ---------- PATHS ----------
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
def save_violation(frame, vehicle_name, plate, reason, bbox):
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]

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

    # limit size
    data = data[:500]

    with open(VIOLATION_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ---------- LOGIC ----------
def is_rider(person_box, bike_box):
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box

    overlap = min(px2, bx2) - max(px1, bx1)
    width_ratio = overlap / (px2 - px1 + 1e-5)
    if width_ratio < 0.5:
        return False

    if py2 < by1 + int(0.4*(by2 - by1)):
        return False

    return True


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


def mark_violations(frame, boxes, viol_info):
    for i, box in enumerate(boxes):
        if i >= len(viol_info):
            continue
        x1, y1, x2, y2 = map(int, box)
        info = viol_info[i]

        color = (0,0,255) if info['helmet']=="No" or info['triple_riding']=="Yes" else (0,255,0)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

    return frame


# ---------- IMAGE ----------
def process_image(file_path):
    img = cv2.imread(file_path)

    results = vehicle_model(img, conf=0.3)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    person_results = person_model(img, conf=0.3, classes=[0])
    person_boxes = person_results[0].boxes.xyxy.cpu().numpy()

    viol_info = []

    for i, r in enumerate(boxes):
        x1, y1, x2, y2 = map(int, r)
        vehicle_crop = img[y1:y2, x1:x2]
        vehicle_name = vehicle_model.names[int(results[0].boxes.cls[i])]

        riders = [p for p in person_boxes if is_rider(p, [x1,y1,x2,y2])]
        triple = "Yes" if len(riders) >=3 else "No"

        helmet = "Yes"
        for rider in riders:
            rx1, ry1, rx2, ry2 = map(int, rider)
            h = ry2 - ry1
            head_crop = img[ry1:ry1+int(0.25*h), rx1:rx2]
            if not detect_helmet(head_crop):
                helmet="No"
                break

        plate = detect_number_plate(vehicle_crop)

        # ✅ SAVE VIOLATION
        if helmet == "No" or triple == "Yes":
            reason = "No Helmet" if helmet == "No" else "Triple Riding"
            save_violation(img, vehicle_name, plate, reason, (x1,y1,x2,y2))

        viol_info.append({
            "vehicle":vehicle_name,
            "helmet":helmet,
            "triple_riding":triple,
            "plate":plate
        })

    annotated_img = mark_violations(img, boxes, viol_info)

    os.makedirs("static/uploads", exist_ok=True)
    filename = os.path.basename(file_path)
    out_path = os.path.join("static/uploads", "annotated_" + filename)

    cv2.imwrite(out_path, annotated_img)

    return "annotated_" + filename, viol_info


# ---------- VIDEO ----------
def generate_video_stream(file_path):
    cap = cv2.VideoCapture(file_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = vehicle_model(frame, conf=0.3)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        person_results = person_model(frame, conf=0.3, classes=[0])
        person_boxes = person_results[0].boxes.xyxy.cpu().numpy()

        viol_info = []

        for i,r in enumerate(boxes):
            x1,y1,x2,y2 = map(int,r)
            vehicle_crop = frame[y1:y2, x1:x2]
            vehicle_name = vehicle_model.names[int(results[0].boxes.cls[i])]

            riders = [p for p in person_boxes if is_rider(p,[x1,y1,x2,y2])]
            triple = "Yes" if len(riders)>=3 else "No"

            helmet="Yes"
            for rider in riders:
                rx1,ry1,rx2,ry2 = map(int,rider)
                h = ry2 - ry1
                head_crop = frame[ry1:ry1+int(0.25*h), rx1:rx2]
                if not detect_helmet(head_crop):
                    helmet="No"
                    break

            plate = detect_number_plate(vehicle_crop)

            # ✅ SAVE
            if helmet == "No" or triple == "Yes":
                reason = "No Helmet" if helmet == "No" else "Triple Riding"
                save_violation(frame, vehicle_name, plate, reason, (x1,y1,x2,y2))

            viol_info.append({
                "vehicle":vehicle_name,
                "helmet":helmet,
                "triple_riding":triple,
                "plate":plate
            })

        frame = mark_violations(frame, boxes, viol_info)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()


# ---------- STREAM TABLE ----------
def generate_violation_stream(file_path):
    cap = cv2.VideoCapture(file_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = vehicle_model(frame, conf=0.3)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        person_results = person_model(frame, conf=0.3, classes=[0])
        person_boxes = person_results[0].boxes.xyxy.cpu().numpy()

        viol_info = []

        for i,r in enumerate(boxes):
            x1,y1,x2,y2 = map(int,r)
            vehicle_crop = frame[y1:y2, x1:x2]
            vehicle_name = vehicle_model.names[int(results[0].boxes.cls[i])]

            riders = [p for p in person_boxes if is_rider(p,[x1,y1,x2,y2])]
            triple = "Yes" if len(riders)>=3 else "No"

            helmet="Yes"
            for rider in riders:
                rx1,ry1,rx2,ry2 = map(int,rider)
                h = ry2 - ry1
                head_crop = frame[ry1:ry1+int(0.25*h), rx1:rx2]
                if not detect_helmet(head_crop):
                    helmet="No"
                    break

            plate = detect_number_plate(vehicle_crop)

            viol_info.append({
                "vehicle":vehicle_name,
                "helmet":helmet,
                "triple_riding":triple,
                "plate":plate
            })

        yield f"data: {json.dumps(viol_info)}\n\n"
        sleep(0.1)

    cap.release()