import cv2

# Initialize the camera
cam = cv2.VideoCapture(0)  # Use 0 for the default webcam, you can try different numbers if you have multiple cameras
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

# Make sure 'haarcascade_frontalface_default.xml' is in the same folder as this code
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# For each person, enter one numeric face id (must enter number start from 1, this is the label of person 1)
face_id = input('\n Enter user id and press <return>: ')

print("\n [INFO] Initializing face capture. Look at the camera and wait...")

# Initialize individual sampling face count
count = 0

# Start detecting your face and take 30 pictures
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

        cv2.imshow('image', img)

    # Press 'ESC' for exiting video
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 30:  # Take 30 face samples and stop video
        break

# Cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
