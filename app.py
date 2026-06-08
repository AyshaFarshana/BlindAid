from flask import Flask, render_template, Response, request
import cv2
import time
import os
import threading
import speech_recognition as sr
from ultralytics import YOLO
from gtts import gTTS
from playsound import playsound

status_text="System ready"
state_data={"state":"ready"}

app = Flask(__name__)

# ================= MODEL =================

model = YOLO("yolov8n.pt")

# ================= GLOBAL =================

mode = "road"
target_label = "cell phone"

frame_global = None

spoken_objects = {}

COOLDOWN = 2

PRIORITY_CLASSES = [
    "person","car","bus",
    "truck","motorcycle",
    "bicycle","traffic light"
]


# ================= SPEECH =================

def speak(text):

    global status_text

    status_text = text
    state_data["state"] = "speaking"

    def run():
        try:
            print("Speak:", text)

            filename = "temp.mp3"

            tts = gTTS(text=text, lang="en")
            tts.save(filename)

            playsound(filename)

            os.remove(filename)

        except:
            pass

        state_data["state"] = "ready"

    threading.Thread(target=run, daemon=True).start()

def voice_listener():

    global mode, target_label, spoken_objects

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    print("Voice control ready")

    while True:

        try:

            state_data["state"]="listening"

            with mic as source:
                audio = recognizer.listen(source)

            command = recognizer.recognize_google(audio)
            state_data["state"]="ready"

            print("Heard:", command)

            # ---------- MODE ----------

            if "road" in command:

                mode = "road"
                spoken_objects = {}
                speak("road mode activated")

            elif "market" in command:

                mode = "market"
                spoken_objects = {}
                speak("market mode activated")

            # ---------- TARGET ----------

            if "find" in command:

                if "phone" in command:
                    target_label = "cell phone"

                elif "bottle" in command:
                    target_label = "bottle"

                elif "book" in command:
                    target_label = "book"

                elif "cup" in command:
                    target_label = "cup"

                elif "remote" in command:
                    target_label = "remote"

                else:
                    speak("object not recognized")
                    continue

                spoken_objects = {}

                speak(f"searching for {target_label}")

        except:
            pass

# ================= DETECTION LOOP =================

def detection_loop():

    global frame_global, mode, target_label, spoken_objects

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Camera not opened")
        return

    frame_count = 0
    DETECTION_INTERVAL = 6

    last_detections = []

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        frame_count += 1
        current_time = time.time()

        small = cv2.resize(frame,(256,192))

        scale_x = frame.shape[1] / 256
        scale_y = frame.shape[0] / 192

        if frame_count % DETECTION_INTERVAL == 0:

            results = model(small,conf=0.4,verbose=False)

            detections = []

            for r in results:

                if r.boxes is None:
                    continue

                for box in r.boxes:

                    cls = int(box.cls[0])
                    name = model.names[cls]

                    x1,y1,x2,y2 = map(int,box.xyxy[0])

                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)

                    detections.append({
                        "class": name,
                        "box": (x1,y1,x2,y2)
                    })

            last_detections = detections

        else:
            detections = last_detections


        message = None
        object_name = None


        # ========= ROAD MODE =========

        if mode == "road":

            filtered = [
                d for d in detections
                if d["class"] in PRIORITY_CLASSES
            ]

            if filtered:
                frame_center=frame.shape[1]//2

                closest = min(
            filtered,
            key=lambda d: abs(
                ((d["box"][0] + d["box"][2]) // 2) - frame_center
            )
        )


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


        # ========= MARKET MODE =========

        elif mode == "market":

            if detections:

                targets = [
                    d for d in detections
                    if d["class"] == target_label
                ]

                if targets:

                    closest = max(
                        targets,
                        key=lambda d:
                        (d["box"][2]-d["box"][0])*
                        (d["box"][3]-d["box"][1])
                    )

                    x1,y1,x2,y2 = closest["box"]

                    object_name = target_label

                    center_x = (x1+x2)//2
                    area = (x2-x1)*(y2-y1)

                    if center_x < frame.shape[1]*0.25:
                        message = "move left"

                    elif center_x > frame.shape[1]*0.75:
                        message = "move right"

                    elif area < 6000:
                        message = "move closer"

                    else:
                        message = "correct product selected"

                else:

                    wrong = detections[0]["class"]

                    object_name = "wrong"

                    message = f"this is {wrong}"


        # ========= SPEAK =========

        if message and object_name:

            if object_name not in spoken_objects:

                speak(message)
                spoken_objects[object_name] = current_time

            elif current_time - spoken_objects[object_name] > COOLDOWN:

                speak(message)
                spoken_objects[object_name] = current_time


        # ========= DRAW =========

        for d in detections:

            x1,y1,x2,y2 = d["box"]

            cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2),
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                d["class"],
                (x1,y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                1
            )


        cv2.putText(
            frame,
            f"MODE: {mode}",
            (10,25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,0,255),
            2
        )

        frame_global = frame


threading.Thread(
    target=detection_loop,
    daemon=True
).start()

threading.Thread(
    target=voice_listener,
    daemon=True
).start()

# ================= STREAM =================

def generate():

    global frame_global

    while True:

        if frame_global is None:
            time.sleep(0.01)
            continue

        ret, buffer = cv2.imencode(".jpg", frame_global)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )


# ================= ROUTES =================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route("/mode", methods=["POST"])
def change_mode():

    global mode, spoken_objects

    mode = request.form["mode"]

    spoken_objects = {}

    speak(mode + " mode activated")

    return "OK"


@app.route("/target", methods=["POST"])
def change_target():

    global target_label, spoken_objects

    target_label = request.form["target"]

    spoken_objects = {}

    speak("searching " + target_label)

    return "OK"

@app.route("/state")
def state():
    return state_data["state"]


if __name__ == "__main__":
    app.run(debug=False, threaded=True, use_reloader=False)