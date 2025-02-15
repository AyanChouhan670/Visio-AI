from flask import Flask, render_template, Response
from facial_emotion_recognition import EmotionRecognition
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import threading
import logging

# Initialize Flask app
app = Flask(__name__)

# Initialize logger
logging.basicConfig(level=logging.INFO)

# Video Stream Class with threading
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for performance
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# Initialize VideoStream instead of using global `cap`
video_stream = VideoStream().start()

# Initialize EmotionRecognition
er = EmotionRecognition(device='cpu')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/face_detection')
def face_detection():
    return Response(gen_face_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/air_canvas')
def air_canvas():
    return Response(gen_air_canvas(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_recognition')
def emotion_recognition():
    return Response(gen_emotion_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# Graceful shutdown for releasing the camera
@app.route('/shutdown')
def shutdown():
    video_stream.stop()
    return "Camera and stream stopped successfully."

# Generator for Face Detection
def gen_face_detection():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    while True:
        frame = video_stream.read()
        if frame is None:
            logging.error("Error: Unable to capture video")
            break
        
        frame = cv2.flip(frame, 1)  # Flip for mirror effect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        num_faces = len(faces)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display face count messages
        if num_faces == 1:
            cv2.putText(frame, "SINGLE FACE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        elif num_faces == 2:
            cv2.putText(frame, "DOUBLE FACES DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        elif num_faces > 2:
            cv2.putText(frame, "MULTIPLE FACES DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logging.error("Error: Encoding frame failed")
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Generator for Air Canvas (drawing based on hand tracking)
def gen_air_canvas():
    # Initialize deque for storing points for each color
    bpoints = [deque(maxlen=1024)]
    gpoints = [deque(maxlen=1024)]
    rpoints = [deque(maxlen=1024)]
    ypoints = [deque(maxlen=1024)]

    # Initialize index for each color
    blue_index = green_index = red_index = yellow_index = 0

    # Define color values
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    colorIndex = 0

    # Initialize white canvas for drawing
    paintWindow = np.ones((471, 636, 3)) + 255

    # Draw color selection buttons on canvas
    paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
    paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)
    paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)
    paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)
    paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 2)

    # Add text labels on the color buttons
   # Add text labels for buttons on the paint window
    cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.namedWindow('paint', cv2.WINDOW_AUTOSIZE)

    # Initialize Mediapipe hands module for hand tracking
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    # Frame skip for performance (process every 2nd frame)
    frame_skip = 2
    frame_count = 0

    while True:
        frame = video_stream.read()  # Read the current frame
        if frame is None:
            logging.error("Error: Unable to capture video")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally for mirror effect
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for Mediapipe processing

        # Draw the color selection buttons on the live frame
        frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
        frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
        frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
        frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
        frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)

        cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        # Get hand landmarks and draw on the canvas based on finger positions
        results = hands.process(framergb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Index finger tip coordinates (landmark #8)
                index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_finger_tip.x * 640)
                y = int(index_finger_tip.y * 480)

                # Check if the fingertip is within one of the color selection buttons
                if 40 <= x <= 140 and 1 <= y <= 65:
                    bpoints = [deque(maxlen=1024)]
                    gpoints = [deque(maxlen=1024)]
                    rpoints = [deque(maxlen=1024)]
                    ypoints = [deque(maxlen=1024)]
                    blue_index = green_index = red_index = yellow_index = 0
                    paintWindow[67:, :, :] = 255  # Clear the canvas
                elif 160 <= x <= 255 and 1 <= y <= 65:
                    colorIndex = 0  # Blue
                elif 275 <= x <= 370 and 1 <= y <= 65:
                    colorIndex = 1  # Green
                elif 390 <= x <= 485 and 1 <= y <= 65:
                    colorIndex = 2  # Red
                elif 505 <= x <= 600 and 1 <= y <= 65:
                    colorIndex = 3  # Yellow
                else:
                    # Add points to the corresponding color deque based on the index finger position
                    if colorIndex == 0:
                        bpoints[blue_index].appendleft((x, y))
                    elif colorIndex == 1:
                        gpoints[green_index].appendleft((x, y))
                    elif colorIndex == 2:
                        rpoints[red_index].appendleft((x, y))
                    elif colorIndex == 3:
                        ypoints[yellow_index].appendleft((x, y))

                mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

        # Draw lines on the frame for all points in each color deque
        for i in range(len(bpoints)):
            for j in range(1, len(bpoints[i])):
                if bpoints[i][j - 1] is None or bpoints[i][j] is None:
                    continue
                cv2.line(frame, bpoints[i][j - 1], bpoints[i][j], colors[0], 2)
                cv2.line(paintWindow, bpoints[i][j - 1], bpoints[i][j], colors[0], 2)

        for i in range(len(gpoints)):
            for j in range(1, len(gpoints[i])):
                if gpoints[i][j - 1] is None or gpoints[i][j] is None:
                    continue
                cv2.line(frame, gpoints[i][j - 1], gpoints[i][j], colors[1], 2)
                cv2.line(paintWindow, gpoints[i][j - 1], gpoints[i][j], colors[1], 2)

        for i in range(len(rpoints)):
            for j in range(1, len(rpoints[i])):
                if rpoints[i][j - 1] is None or rpoints[i][j] is None:
                    continue
                cv2.line(frame, rpoints[i][j - 1], rpoints[i][j], colors[2], 2)
                cv2.line(paintWindow, rpoints[i][j - 1], rpoints[i][j], colors[2], 2)

        for i in range(len(ypoints)):
            for j in range(1, len(ypoints[i])):
                if ypoints[i][j - 1] is None or ypoints[i][j] is None:
                    continue
                cv2.line(frame, ypoints[i][j - 1], ypoints[i][j], colors[3], 2)
                cv2.line(paintWindow, ypoints[i][j - 1], ypoints[i][j], colors[3], 2)

        # Show the paint window in a separate window for reference
        cv2.imshow("Paint", paintWindow)

        # Encode the frame to a JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logging.error("Error: Encoding frame failed")
            continue
        frame = buffer.tobytes()

        # Yield the frame as a multipart response for streaming
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Generator for Emotion Recognition
def gen_emotion_recognition():
    frame_skip = 5  # Process every 5th frame for emotion recognition
    frame_count = 0

    while True:
        frame = video_stream.read()
        if frame is None:
            logging.error("Error: Unable to capture video")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame = cv2.flip(frame, 1)
        emotions = er.recognise_emotion(frame, return_type='BGR')

        ret, buffer = cv2.imencode('.jpg', emotions)
        if not ret:
            logging.error("Error: Encoding frame failed")
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Main app execution
if __name__ == '__main__':
    try:
        app.run(debug=False, threaded=True)
    finally:
        video_stream.stop()
