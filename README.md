# BlindAid

**BlindAid Video Demonstartion:** https://drive.google.com/file/d/1M08rQ0on-cPAfXxGeOUcVfCFtAyiEt51/view?usp=sharing 
Blind Aid is an intelligent assistive application designed to help visually impaired individuals understand and interact with their surroundings. The system combines Artificial Intelligence, Computer Vision, Optical Character Recognition (OCR), and Voice Assistance to provide real-time environmental awareness and guidance.

The application captures live video through a webcam, detects surrounding objects, reads printed text, and provides audio feedback to the user.

---

**Problem Statement**

Visually impaired individuals often face difficulties in identifying objects, reading printed text, and navigating unfamiliar environments independently. Blind Aid addresses these challenges by providing real-time scene understanding and voice-based assistance.

---

**Objectives**

- Detect and identify surrounding objects in real time.
- Read printed text from the environment using OCR.
- Provide voice-based feedback to users.
- Assist users in navigation and object identification.
- Improve accessibility through AI-powered assistance.

---

**Key Features**

**Real-Time Object Detection**

The system uses YOLOv8 to identify multiple objects through a live webcam feed and provide spoken descriptions.

**OCR Text Reading**

EasyOCR is used to extract printed text from images and display or speak the detected content.

**Voice Assistance**

The application converts detected information into speech, allowing hands-free interaction.

**Road Mode**

Designed to assist users in identifying obstacles and important objects in outdoor environments.

**Market Mode**

Allows users to locate and identify specific products and objects in shopping environments.

**Web-Based Interface**

The project includes a Flask-based web interface for easy interaction and monitoring.

---

**Technologies Used**

**Frontend**

- HTML
- CSS
- JavaScript

**Backend**

- Python
- Flask

**Artificial Intelligence & Computer Vision**

- YOLOv8
- OpenCV
- EasyOCR

**Voice Processing**

- gTTS
- Speech Recognition

---

**System Workflow**

1. Capture live video through webcam.
2. Process frames using YOLOv8 object detection.
3. Extract text using EasyOCR when required.
4. Generate voice output for detected objects and text.
5. Provide user assistance through Road Mode and Market Mode.

---

**Project Modules**

**Object Detection Module**

Detects surrounding objects using the YOLOv8 model.

**OCR Module**

Extracts and reads printed text from the environment.

**Voice Output Module**

Converts system responses into speech.

**Mode Management Module**

Handles switching between Road Mode and Market Mode.

**User Interface Module**

Provides a web-based interface using Flask.

---

**Applications**

- Assistive technology for visually impaired individuals.
- Educational demonstrations of AI and Computer Vision.
- Smart accessibility solutions.
- Human-computer interaction research.

---

**Future Enhancements**

- GPS-based navigation assistance.
- Currency recognition.
- Face recognition for known contacts.
- Mobile application support.
- Multilingual voice assistance.

---

**Conclusion**

Blind Aid demonstrates how Artificial Intelligence and Computer Vision can be utilized to improve accessibility for visually impaired individuals. By integrating object detection, OCR, and voice assistance into a single platform, the system enables users to interact with their environment more independently and confidently.
