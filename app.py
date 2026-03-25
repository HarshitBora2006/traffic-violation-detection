from flask import Flask, render_template, request, Response, jsonify
from detection import (
    process_image,
    generate_video_stream,
    generate_violation_stream,
    load_violations
)
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

uploaded_video_path = ""


# ---------------- DASHBOARD ----------------
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


# ---------------- UPLOAD ----------------
@app.route("/upload", methods=["POST"])
def upload():
    global uploaded_video_path

    file = request.files.get("file")

    if not file or file.filename == "":
        return jsonify({"error": "No file"})

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    uploaded_video_path = file_path

    # IMAGE
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov")):
        output_image, result_data = process_image(file_path)

        return jsonify({
            "type": "image",
            "image": output_image,
            "data": result_data
        })

    # VIDEO
    return jsonify({"type": "video"})


# ---------------- VIDEO STREAM ----------------
@app.route("/video_feed")
def video_feed():
    global uploaded_video_path

    if not uploaded_video_path:
        return "No video uploaded"

    return Response(
        generate_video_stream(uploaded_video_path),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ✅ FIXED STREAM
@app.route("/violations_stream")
def violations_stream():
    return Response(
        generate_violation_stream(),
        mimetype='text/event-stream'
    )


# ---------------- API ----------------
@app.route("/api/violations")
def api_violations():
    try:
        return jsonify(load_violations())
    except:
        return jsonify([])


# ---------------- RUN ----------------
if __name__ == "__main__":
    print("🚀 Running: http://127.0.0.1:5000")
    app.run(debug=True, threaded=True)