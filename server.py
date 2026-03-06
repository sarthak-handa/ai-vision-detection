from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import time
import os

app = Flask(__name__)
CORS(app)

model = YOLO("PPE_detection_YOLO-main/PPE_detection_YOLO-main/ppe.pt")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return """
    <h2>Helmet Detection</h2>

    <input type="file" id="imageInput"/>
    <button onclick="uploadImage()">Upload</button>

    <p id="status" style="white-space: pre-line; font-weight: bold;"></p>

    <h3>Result</h3>
    <img id="resultImage" width="600"/>

    <script>
    async function uploadImage() {

        const fileInput = document.getElementById("imageInput")
        const status = document.getElementById("status")
        const resultImage = document.getElementById("resultImage")

        if (!fileInput.files.length) {
            alert("Please select an image")
            return
        }

        const formData = new FormData()
        formData.append("image", fileInput.files[0])

        status.innerText = "Processing..."

        try {
            const response = await fetch("/detect-helmet", {
                method: "POST",
                body: formData
            })

            const data = await response.json()

            // --- STEP 15: Updated Status Display ---
            status.innerText = 
                "⏱ Detection time: " + data.time_taken + " seconds\\n" +
                "🟢 Helmets: " + data.helmets + "\\n" +
                "🔴 Violations: " + data.violations

            resultImage.src = data.image_url + "?t=" + new Date().getTime()
        } catch (error) {
            status.innerText = "Error processing the image."
            console.error(error)
        }
    }
    </script>
    """

@app.route("/detect-helmet", methods=["POST"])
def detect_helmet():
    file = request.files["image"]
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    start = time.time()

    results = model(input_path)[0]
    img = cv2.imread(input_path)

    # --- STEP 13: Added Counters ---
    helmet_count = 0
    no_helmet_count = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        score = f"{round(confidence*100)}%"

        if "Hardhat" in label:
            helmet_count += 1
            text = f"HELMET {score}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, text, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if "NO-Hardhat" in label:
            no_helmet_count += 1
            text = f"NO HELMET {score}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, text, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    output_path = os.path.join(OUTPUT_FOLDER, "result.jpg")
    cv2.imwrite(output_path, img)

    end = time.time()

    # --- STEP 14: Return Statistics ---
    return jsonify({
        "time_taken": round(end - start, 2),
        "image_url": "/result",
        "helmets": helmet_count,
        "violations": no_helmet_count
    })

@app.route("/result")
def result():
    return send_file("outputs/result.jpg", mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(port=5000)