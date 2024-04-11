import cv2
import numpy as np
from subprocess import call

# Load pre-trained face detection model
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

font = cv2.FONT_HERSHEY_SIMPLEX

# List of names corresponding to IDs in the recognizer
names = ['', 'hedi']  # Index 0 is empty, IDs start from 1

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Main loop
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Predict the ID of the face
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # Check if confidence is less than a threshold (you may need to adjust this)
        if confidence < 100:
            name = names[id]
            confidence_text = "  {0}%".format(round(100 - confidence))
        else:
            name = "unknown"
            confidence_text = "  {0}%".format(round(100 - confidence))

        # Draw rectangle around the face and display name and confidence
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, name, (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, confidence_text, (x+5, y+h-5), font, 1, (255, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
