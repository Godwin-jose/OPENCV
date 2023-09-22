import cv2 as cv
import numpy as np
import joblib

def process(image):
    # Resize the image to 48x48 pixels (assuming it's a square image)
    resized_image = cv.resize(image, (48, 48))

    # Flatten the image (convert it to a 1D array)
    flattened_image = resized_image.flatten()

    # Reshape for prediction
    new = flattened_image.reshape(1, -1)  # Reshape for prediction

    # Predict using the model
    result = model.predict(new)

    return result

# Load the pre-trained face detection Haar Cascade classifier
face_cascade = cv.CascadeClassifier("face.xml")

# Load the pre-trained expression recognition model
model = joblib.load("face.joblib")

cap = cv.VideoCapture(0)

while True:
    istrue, frame = cap.read()

    # Process the frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert to grayscale

    # Perform face detection
    face_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)

    for (x, y, w, h) in face_rect:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        # Crop the detected face region and use the process function
        detected_face = gray[y:y+h, x:x+w]
        result = process(detected_face)  # Predict emotion
        cv.imshow("face",detected_face)
        if result[0]==0:
            print("predicted emotion:Happy")
        else:
            print("predicted emotion:Sad")

        # Print predicted emotion

    # Display the resulting frame
    cv.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close the OpenCV window
cap.release()
cv.destroyAllWindows()
