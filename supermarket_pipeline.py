# supermarket_pipeline.py
import threading
import time
from collections import deque, defaultdict

import cv2
import numpy as np
import easyocr
import pyttsx3
from ultralytics import YOLO

# Optional: speech recognition (voice commands)
import speech_recognition as sr

# ---------------------- CONFIG ----------------------
VIDEO_URL = "http://192.168.29.30:8080/video"   # <<=== change this to your /video URL
YOLO_MODEL = "yolov8n.pt"                       # lightweight model
OCR_LANGS = ["en"]                              # EasyOCR languages
TTS_RATE = 170                                  # words per minute for pyttsx3
ANNOUNCE_MIN_INTERVAL = 2.0                     # seconds between identical announcements
LABEL_TTL = 6.0                                 # seconds to keep a label "alive" so it isn't reannounced
DETECTION_CONFIDENCE = 0.25                     # YOLO confidence threshold (0-1)
OCR_MIN_WORD_LEN = 2                            # ignore tiny OCR words
# ----------------------------------------------------

# Initialize components
print("Loading YOLO model...")
model = YOLO(YOLO_MODEL)

print("Loading EasyOCR reader (this might take time)...")
reader = easyocr.Reader(OCR_LANGS, gpu=False)  # set gpu=True if you have GPU configured

print("Initializing TTS...")
tts = pyttsx3.init()
tts.setProperty("rate", TTS_RATE)

def speak(text):
    """Non-blocking TTS wrapper that blocks while speaking to avoid overlap."""
    try:
        tts.say(text)
        tts.runAndWait()
    except Exception as e:
        print("TTS error:", e)

# Storage for event-based announcements
last_announced_time = defaultdict(lambda: 0.0)  # label -> timestamp
label_history = {}  # label -> {'first_seen': ts, 'last_seen': ts, 'count': n}
lock = threading.Lock()

# For voice commands we keep recent detections (center preference)
recent_detections = deque(maxlen=8)  # items: dict with keys 'label','box','ts','ocr_text'

# Speech recognition thread: sets commands to a queue
command_queue = deque(maxlen=4)

def speech_listener():
    """Background thread to listen for voice commands and push them to command_queue."""
    recognizer = sr.Recognizer()
    mic = None
    try:
        mic = sr.Microphone()
    except Exception as e:
        print("Microphone not available or PyAudio missing. Voice commands disabled. Error:", e)
        return

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.0)
    print("Voice listener ready. Say: 'What is this?' or 'Is this <name>?'")

    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
            try:
                text = recognizer.recognize_google(audio)  # requires internet
                text = text.lower().strip()
                print("[Voice command recognized]:", text)
                with lock:
                    command_queue.append(text)
            except sr.UnknownValueError:
                # Nothing recognized - ignore quietly
                continue
            except sr.RequestError as e:
                print("Speech recognition request failed:", e)
                # Avoid spamming; sleep and continue
                time.sleep(1.0)
        except Exception:
            # listening timeout or mic error; just continue listening
            continue

def pick_focused_detection(detections):
    """
    Choose the detection nearest to the center of the frame (useful for 'What is this?' queries).
    Each detection is (box, ocr_text).
    box is (x1, y1, x2, y2)
    """
    if not detections:
        return None

    # assume frame center (we will get frame size before calling)
    # caller must pass detections with frame size context; here we simply choose detection with box closest to center
    # For simplicity, caller should pass detections relative to same frame
    centers = []
    for item in detections:
        (x1, y1, x2, y2), ocr_text, score = item
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append((cx, cy, item))
    # For now, caller will find true center and compute distance externally; we'll return last detection if needed
    return centers

def run_pipeline():
    cap = cv2.VideoCapture(VIDEO_URL)
    if not cap.isOpened():
        print("Failed to open camera stream:", VIDEO_URL)
        return

    # Get frame size for focus calculations
    ret, frame = cap.read()
    if not ret:
        print("Failed to read initial frame from camera.")
        cap.release()
        return
    height, width = frame.shape[:2]
    frame_center = (width / 2, height / 2)

    last_frame_time = time.time()
    print("Starting main loop. Press Q in the window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            # Try again a few times
            time.sleep(0.5)
            continue

        # YOLO detection (we let model handle preprocessing)
        results = model(frame, conf=DETECTION_CONFIDENCE, save=False)
        # results[0] is the prediction for this frame
        boxes = []
        try:
            det = results[0].boxes
            # Each box has .xyxy, .conf, .cls
            for b in det:
                xyxy = b.xyxy[0].cpu().numpy() if hasattr(b.xyxy[0], "cpu") else b.xyxy[0].numpy()
                conf = float(b.conf[0]) if hasattr(b.conf[0], "item") else float(b.conf)
                cls_id = int(b.cls[0]) if hasattr(b.cls[0], "item") else int(b.cls)
                # Keep boxes with reasonable confidence
                if conf < DETECTION_CONFIDENCE:
                    continue
                x1, y1, x2, y2 = map(int, xyxy)
                boxes.append((x1, y1, x2, y2, conf, cls_id))
        except Exception:
            # Fallback parsing for older ultralytics returns
            for r in results:
                if hasattr(r, 'boxes'):
                    for b in r.boxes:
                        try:
                            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                            conf = float(b.conf[0])
                            cls_id = int(b.cls[0])
                            boxes.append((x1, y1, x2, y2, conf, cls_id))
                        except Exception:
                            continue

        frame_ann = frame.copy()
        current_frame_detections = []

        # For each box: crop -> OCR -> determine label
        for (x1, y1, x2, y2, conf, cls_id) in boxes:
            # enforce box inside frame
            x1c = max(0, x1); y1c = max(0, y1)
            x2c = min(width - 1, x2); y2c = min(height - 1, y2)
            if x2c - x1c < 10 or y2c - y1c < 10:
                continue

            crop = frame[y1c:y2c, x1c:x2c]
            # Preprocessing for OCR (you already have a good pipeline; if you want, integrate it here)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Use EasyOCR on the crop
            try:
                ocr_result = reader.readtext(gray, detail=0, paragraph=False)
            except Exception as e:
                print("OCR error:", e)
                ocr_result = []

            # Join OCR words into a single string, filter short words
            words = [w.strip() for w in ocr_result if len(w.strip()) >= OCR_MIN_WORD_LEN]
            text_label = " ".join(words).strip()

            # If OCR found nothing, use coarse YOLO class name if available (e.g., 'bottle', 'chair')
            label_to_announce = None
            if text_label:
                label_to_announce = text_label
            else:
                # try to extract class name from YOLO's class id
                try:
                    coco_name = model.model.names[cls_id] if hasattr(model, "model") else None
                except Exception:
                    coco_name = None
                label_to_announce = coco_name if coco_name else "object"

            # Draw box + label
            display_label = label_to_announce if label_to_announce else "object"
            cv2.rectangle(frame_ann, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)
            cv2.putText(frame_ann, display_label, (x1c, y1c - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Save detection for event-based logic and speech queries
            ts = time.time()
            current_frame_detections.append(((x1c, y1c, x2c, y2c), label_to_announce, conf, ts, text_label))
            with lock:
                recent_detections.appendleft({
                    "box": (x1c, y1c, x2c, y2c),
                    "label": label_to_announce,
                    "ocr_text": text_label,
                    "ts": ts
                })

        # EVENT-BASED ANNOUNCEMENTS (B)
        now = time.time()
        announced_now = []
        for det in current_frame_detections:
            box, label_to_announce, conf, ts, ocr_text = det
            if not label_to_announce:
                continue
            # Normalize label (lowercase)
            canon_label = label_to_announce.strip().lower()

            # Announce if not announced recently
            if now - last_announced_time[canon_label] > LABEL_TTL:
                # Announce new detection
                # Prefer OCR text for announcement if available
                if ocr_text:
                    announce_text = f"{ocr_text}"
                else:
                    announce_text = canon_label

                # Make TTS in separate thread so detection loop doesn't block long
                threading.Thread(target=speak, args=(announce_text,), daemon=True).start()
                last_announced_time[canon_label] = now
                announced_now.append(canon_label)

            # Update label history
            with lock:
                if canon_label not in label_history:
                    label_history[canon_label] = {"first_seen": now, "last_seen": now, "count": 1}
                else:
                    label_history[canon_label]["last_seen"] = now
                    label_history[canon_label]["count"] += 1

        # Show annotated frame
        cv2.imshow("BlindAid - Supermarket Mode (Press Q to quit)", frame_ann)

        # Handle voice commands from queue
        with lock:
            if command_queue:
                cmd = command_queue.popleft()
            else:
                cmd = None

        if cmd:
            handle_command(cmd, frame, width, height)

        # Exit on Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def handle_command(cmd_text, current_frame, frame_w, frame_h):
    """
    Interpret voice command and answer using recent_detections.
    Supports:
      - "what is this"
      - "is this <name>"
      - "what is on the left/right"
    """
    cmd = cmd_text.lower()
    print("Handling command:", cmd)

    # If we have no recent detections, say so
    with lock:
        detections_snapshot = list(recent_detections)

    if not detections_snapshot:
        speak("I don't see any object.")
        return

    # Choose the detection nearest the center of the frame as the focused object
    frame_center = (frame_w / 2, frame_h / 2)
    best = None
    best_dist = 1e9
    for det in detections_snapshot:
        (x1, y1, x2, y2) = det["box"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        dist = ((cx - frame_center[0]) ** 2 + (cy - frame_center[1]) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best = det

    focused = best

    # Handle common queries
    if "what is this" in cmd or "what is that" in cmd or cmd.strip() == "what's this":
        # Prefer OCR text if present
        if focused["ocr_text"]:
            speak(focused["ocr_text"])
        else:
            speak(focused["label"])
        return

    if cmd.startswith("is this"):
        # expected format: "is this <name>"
        parts = cmd.split()
        if len(parts) >= 3:
            query = " ".join(parts[2:]).strip()
            # compare with OCR text or label
            ocr_text = focused["ocr_text"].lower() if focused["ocr_text"] else ""
            label_text = focused["label"].lower() if focused["label"] else ""
            if query in ocr_text or query in label_text:
                speak("Yes.")
            else:
                speak("No. It looks like " + (ocr_text or label_text))
            return

    # Fallbacks
    speak("I didn't understand. Try: 'what is this' or 'is this shampoo'.")

# Start speech listener thread
listener_thread = threading.Thread(target=speech_listener, daemon=True)
listener_thread.start()

# Run the main pipeline
if __name__ == "__main__":
    run_pipeline()
