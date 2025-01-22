import cv2
import numpy as np
from tf_keras.models import load_model

# Load the face recognition model
model = load_model('face_recognition_model.h5')

# Load target names (assuming you saved them separately during training)
target_names = lfw_people.target_names

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Define the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert the frame to grayscale (model expects grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract the region of interest (the face)
        face = gray[y:y+h, x:x+w]
        
        # Resize the face to match the input size of the model (50x37, or the shape used during training)
        face_resized = cv2.resize(face, (50, 37))
        
        # Reshape the face to match the model input (add batch size and channel dimensions)
        face_input = face_resized.reshape(1, 50, 37, 1)
        
        # Normalize pixel values to [0, 1] (if needed)
        face_input = face_input.astype('float32') / 255.0
        
        # Predict the face using the model
        predictions = model.predict(face_input)
        predicted_class = np.argmax(predictions[0])
        predicted_name = target_names[predicted_class]
        
        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
