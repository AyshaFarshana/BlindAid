import cv2
import easyocr
import numpy as np

# Initialize OCR reader ONCE
reader = easyocr.Reader(['en'], gpu=False)

def run_ocr(frame):
    """
    Input: OpenCV frame (BGR)
    Output: detected text (string)
    """

    # 1. Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)

    # 4. Otsu Threshold
    _, thresh = cv2.threshold(
        sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 5. Dilation
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.dilate(thresh, kernel, iterations=1)

    # 6. OCR
    results = reader.readtext(processed)

    # 7. Extract text
    texts = [res[1] for res in results]
    detected_text = " ".join(texts)

    return detected_text
