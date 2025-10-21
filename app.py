# app.py
from flask import Flask, request, redirect, url_for, send_file, render_template_string
import os
from video_io import process_video

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

INDEX_HTML = """
<!doctype html>
<title>Vehicle Counting Demo</title>
<h1>Upload video to count vehicles</h1>
<form method=post enctype=multipart/form-data action="/upload">
  <input type=file name=video>
  <input type=submit value=Upload>
</form>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return "No file part", 400
    file = request.files["video"]
    if file.filename == "":
        return "No selected file", 400
    in_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(in_path)

    out_video = os.path.join(app.config["OUTPUT_FOLDER"], f"out_{file.filename}")
    out_csv = os.path.join(app.config["OUTPUT_FOLDER"], f"{os.path.splitext(file.filename)[0]}_counts.csv")

    # Process (synchronously) - có thể mất thời gian tùy video
    res = process_video(in_path, output_path=out_video, csv_path=out_csv, display=False)
    return f"Done. <a href='/download_video?path={out_video}'>Download video</a> | <a href='/download_csv?path={out_csv}'>Download CSV</a>"

@app.route("/download_video")
def download_video():
    path = request.args.get("path")
    return send_file(path, as_attachment=True)

@app.route("/download_csv")
def download_csv():
    path = request.args.get("path")
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
