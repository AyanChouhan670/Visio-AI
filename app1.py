
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize color points for different colors
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# Indexes to track color points in specific arrays
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# Kernel for dilation
kernel = np.ones((5, 5), np.uint8)

# Color palette (BGR)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Create the paint window with buttons for CLEAR, BLUE, GREEN, RED, YELLOW
paintWindow = np.ones((471, 636, 3)) * 255
cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)
cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)
cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)
cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 2)
cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip frame horizontally
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw buttons on the frame
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

    # Get hand landmark prediction
    result = hands.process(framergb)

    # Post-process hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
            landmarks = [[int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])] for lm in hand_landmarks.landmark]

            fore_finger = landmarks[8]
            thumb = landmarks[4]
            cv2.circle(frame, tuple(fore_finger), 3, (0, 255, 0), -1)

            if thumb[1] - fore_finger[1] < 30:
                bpoints.append(deque(maxlen=512))
                blue_index += 1
                gpoints.append(deque(maxlen=512))
                green_index += 1
                rpoints.append(deque(maxlen=512))
                red_index += 1
                ypoints.append(deque(maxlen=512))
                yellow_index += 1

            elif fore_finger[1] <= 65:
                if 40 <= fore_finger[0] <= 140:  # Clear Button
                    bpoints = [deque(maxlen=512)]
                    gpoints = [deque(maxlen=512)]
                    rpoints = [deque(maxlen=512)]
                    ypoints = [deque(maxlen=512)]
                    blue_index = green_index = red_index = yellow_index = 0
                    paintWindow[67:, :, :] = 255
                elif 160 <= fore_finger[0] <= 255:
                    colorIndex = 0  # Blue
                elif 275 <= fore_finger[0] <= 370:
                    colorIndex = 1  # Green
                elif 390 <= fore_finger[0] <= 485:
                    colorIndex = 2  # Red
                elif 505 <= fore_finger[0] <= 600:
                    colorIndex = 3  # Yellow
            else:
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(fore_finger)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(fore_finger)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(fore_finger)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(fore_finger)

    # Draw the lines on frame and paint window
    points = [bpoints, gpoints, rpoints, ypoints]
    for i, color_points in enumerate(points):
        for point_list in color_points:
            for k in range(1, len(point_list)):
                if point_list[k - 1] is None or point_list[k] is None:
                    continue
                cv2.line(frame, point_list[k - 1], point_list[k], colors[i], 2)
                cv2.line(paintWindow, point_list[k - 1], point_list[k], colors[i], 2)

    # Display the output frame and paint window
    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    # Quit on pressing 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
