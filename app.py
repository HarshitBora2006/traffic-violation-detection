from flask import Flask, render_template, request, Response, jsonify
from detection import process_image, generate_video_stream, generate_violation_stream, load_violations
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

uploaded_video_path = ""


# ---------------- MAIN DASHBOARD ----------------
@app.route("/", methods=["GET", "POST"])
def dashboard():
    global uploaded_video_path

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            return render_template("dashboard.html")

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        uploaded_video_path = file_path

        # VIDEO
        if file.filename.lower().endswith((".mp4", ".avi", ".mov")):
            return render_template("dashboard.html")

        # IMAGE
        else:
            output_image, result_data = process_image(file_path)

            return render_template(
                "dashboard.html",
                output_image=output_image,
                result_data=result_data
            )

    return render_template("dashboard.html")


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


# ---------------- LIVE TABLE STREAM ----------------
@app.route("/violations_stream")
def violations_stream():
    global uploaded_video_path

    if not uploaded_video_path:
        return "No video uploaded"

    return Response(
        generate_violation_stream(uploaded_video_path),
        mimetype='text/event-stream'
    )


# ---------------- VIOLATIONS DASHBOARD ----------------
@app.route("/violations")
def violations_page():
    return render_template("violations.html")


# ---------------- API ----------------
@app.route("/api/violations")
def api_violations():
    return jsonify(load_violations())


if __name__ == "__main__":
    print("🚀 Running: http://127.0.0.1:5000")
    app.run(debug=True)