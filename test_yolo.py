import cv2
import time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Camera not opened")
    exit()

print("YOLO started")

frame_count = 0
prev_time = time.time()

# store last detections
last_boxes = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (416, 320))

    frame_count += 1

    # run detection every 2nd frame
    if frame_count % 2 == 0:

        results = model(frame, conf=0.4, verbose=False)
        last_boxes = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls = int(box.cls[0])
                name = model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                last_boxes.append((x1, y1, x2, y2, name))

    # draw last detections EVERY frame
    for (x1, y1, x2, y2, name) in last_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, name, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # FPS
    curr = time.time()
    fps = 1 / (curr - prev_time)
    prev_time = curr

    cv2.putText(frame, f"FPS: {int(fps)}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("BlindAid", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()