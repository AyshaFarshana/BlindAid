import cv2
import time
import os
import threading
import speech_recognition as sr
from ultralytics import YOLO
from mode_manager import ModeManager
from gtts import gTTS
from playsound import playsound

# ==============================
# MODEL
# ==============================
model = YOLO("yolov8n.pt")

# ==============================
# MODE MANAGER
# ==============================
mode_manager = ModeManager()

# ==============================
# TARGET OBJECT
# ==============================
target_label = "cell phone"

# ==============================
# NON-BLOCKING SPEECH
# ==============================
def speak(text):
    def run():
        print("Speaking:", text)
        try:
            filename = "temp_speech.mp3"
            tts = gTTS(text=text, lang="en")
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except:
            pass

    threading.Thread(target=run, daemon=True).start()


# ==============================
# VOICE MODE SELECTION
# ==============================
def select_mode_once():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    print("Say road or market")

    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)

            command = recognizer.recognize_google(audio, language="en-IN").lower()
            print("Heard:", command)

            speak(f"I heard {command}")

            if "road" in command:
                mode_manager.switch_mode("road")
                speak("Road mode activated")
                break

            elif "market" in command:
                mode_manager.switch_mode("market")
                speak("Market mode activated")
                break

        except:
            print("Listening again...")


# ==============================
# TARGET CHANGE
# ==============================
def listen_for_target_change():
    global target_label

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)

        command = recognizer.recognize_google(audio, language="en-IN").lower()
        print("Heard:", command)

        speak(f"I heard {command}")

        if "find" in command:

            if "phone" in command:
                target_label = "cell phone"
            elif "bottle" in command:
                target_label = "bottle"
            elif "remote" in command:
                target_label = "remote"
            elif "cup" in command:
                target_label = "cup"
            elif "book" in command:
                target_label = "book"
            else:
                speak("Object not recognized")
                return

            speak(f"Searching for {target_label}")

    except:
        pass


# ==============================
# START SYSTEM
# ==============================
select_mode_once()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Camera not opened")
    exit()

# ==============================
# WINDOW SETTINGS
# ==============================
cv2.namedWindow("BlindAid", cv2.WINDOW_NORMAL)

window_width = 960
window_height = 540

cv2.resizeWindow("BlindAid", window_width, window_height)

screen_width = 1920
x = int((screen_width - window_width) / 2)

y = 120

cv2.moveWindow("BlindAid", x, y)

print("Blind Aid running...")
speak("Blind Aid system started")

# ==============================
# CONFIG
# ==============================
PRIORITY_CLASSES = [
    "person","car","bus",
    "truck","motorcycle",
    "bicycle","traffic light"
]

COOLDOWN = 2
spoken_objects = {}
last_detections = []
frame_count = 0
DETECTION_INTERVAL = 8

# ==============================
# MAIN LOOP
# ==============================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = time.time()

    # Resize for YOLO
    small_frame = cv2.resize(frame,(256,192))

    # Scale factors for correct box placement
    scale_x = frame.shape[1] / 256
    scale_y = frame.shape[0] / 192

    if frame_count % DETECTION_INTERVAL == 0:
        results = model(small_frame,conf=0.4,verbose=False)

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:

                cls = int(box.cls[0])
                name = model.names[cls]

                x1,y1,x2,y2 = map(int,box.xyxy[0])

                # SCALE BOX BACK TO ORIGINAL FRAME
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                detections.append({"class":name,"box":(x1,y1,x2,y2)})

        last_detections = detections
    else:
        detections = last_detections

    current_mode = mode_manager.get_mode()
    message = None
    object_name = None

    if current_mode == "road":

        filtered = [d for d in detections if d["class"] in PRIORITY_CLASSES]

        if filtered:
            closest = max(filtered,key=lambda d:(d["box"][2]-d["box"][0])*(d["box"][3]-d["box"][1]))
            x1,y1,x2,y2 = closest["box"]
            object_name = closest["class"]

            center_x = (x1+x2)//2

            if center_x < frame.shape[1]*0.33:
                direction = "left"
            elif center_x > frame.shape[1]*0.66:
                direction = "right"
            else:
                direction = "center"

            message = f"{object_name} on your {direction}"

    elif current_mode == "market":

        if frame_count % 60 == 0:
            listen_for_target_change()

        if detections:

            targets = [d for d in detections if d["class"] == target_label]

            if targets:
                closest = max(targets,key=lambda d:(d["box"][2]-d["box"][0])*(d["box"][3]-d["box"][1]))
                x1,y1,x2,y2 = closest["box"]
                object_name = target_label

                center_x = (x1+x2)//2
                box_area = (x2-x1)*(y2-y1)

                if center_x < frame.shape[1]*0.25:
                    message = "Move left"
                elif center_x > frame.shape[1]*0.75:
                    message = "Move right"
                elif box_area < 6000:
                    message = "Move closer"
                else:
                    message = "Correct product selected"

            else:
                wrong_object = detections[0]["class"]
                object_name = "wrong_item"
                message = f"This is {wrong_object}. Not the target item"

        else:
            object_name = "nothing"
            message = "Scanning for product"

    if message and object_name:

        if object_name not in spoken_objects:
            speak(message)
            spoken_objects[object_name] = current_time

        elif current_time - spoken_objects[object_name] > COOLDOWN:
            speak(message)
            spoken_objects[object_name] = current_time

    for d in detections:
        x1,y1,x2,y2 = d["box"]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,d["class"],(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

    cv2.putText(frame,
                f"MODE: {current_mode.upper()}",
                (10,25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,0,255),
                2)

    cv2.imshow("BlindAid",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()