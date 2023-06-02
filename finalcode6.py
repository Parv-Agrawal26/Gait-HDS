import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model("gait_detection_model.h5")

# Set up the video capture
cap = cv2.VideoCapture(0)  # Use the webcam (change the index if you have multiple cameras)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 128))
    normalized = resized / 255.0
    input_data = np.expand_dims(normalized, axis=-1)  # Add an extra dimension for the channel
    
    # Perform inference with the model
    prediction = model.predict(np.array([input_data]))
    probability = prediction[0][0]

    # Set a thershold for classification
    thershold = 0.5
    
    # Display the result
    if probability >= thershold:
        label = "Human"
        accuracy = probability
    else:
        label = "Non-Human"
        accuracy = 1 - probability
    
    cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Accuracy: {accuracy:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Real-time Human Detection", frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
