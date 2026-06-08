import cv2
import easyocr
import pyttsx3
import numpy as np

# Load image
image_path = 'test_image.jpg'
image = cv2.imread(image_path)

# --- STRONGEST PREPROCESSING FOR PRINTED TEXT ---

# 1) Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2) Slight Gaussian blur (reduces noise)
blur = cv2.GaussianBlur(gray, (3, 3), 0)

# 3) Sharpen (boosts printed text edges)
kernel = np.array([[0, -1, 0],
                   [-1, 6, -1],
                   [0, -1, 0]])
sharp = cv2.filter2D(blur, -1, kernel)

# 4) Otsu threshold (best for clean printed text)
_, thresh = cv2.threshold(
    sharp, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# 5) Light dilation (strengthens thin characters)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
processed = cv2.dilate(thresh, kernel, iterations=1)

# OCR
reader = easyocr.Reader(['en'])
result = reader.readtext(processed)

text = ' '.join([res[1] for res in result])
print("\nDetected text:\n", text)

# TTS
engine = pyttsx3.init()
engine.say(text)
engine.runAndWait()
