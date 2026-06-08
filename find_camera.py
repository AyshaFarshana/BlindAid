import cv2

print("Starting camera test...")

cap = cv2.VideoCapture(1)  # change index later if needed

print("Camera object created")

if not cap.isOpened():
    print("Camera not opened")
    exit()

print("Camera opened successfully")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("BlindAid Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
