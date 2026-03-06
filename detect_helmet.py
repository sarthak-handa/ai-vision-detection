from ultralytics import YOLO
import cv2

# Load PPE model
model = YOLO("PPE_detection_YOLO-main/PPE_detection_YOLO-main/ppe.pt")

image_path = "uploads/test.jpg"

results = model(image_path)[0]

img = cv2.imread(image_path)

for box in results.boxes:

    cls = int(box.cls[0])
    label = model.names[cls]

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # HELMET DETECTED
    if "Hardhat" in label:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, "HELMET", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # NO HELMET
    if "NO-Hardhat" in label:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
        cv2.putText(img, "NO HELMET", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

output_path = "outputs/result.jpg"
cv2.imwrite(output_path, img)

print("Helmet safety detection complete:", output_path)