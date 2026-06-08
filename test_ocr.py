import cv2
from ocr_module import run_ocr

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if ret:
    text = run_ocr(frame)
    print("Detected Text:", text)
else:
    print("Camera failed")

cap.release()
