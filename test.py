import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Load the trained model
model = load_model("asl_model.keras")

# Define the class labels (assumes folder names are alphabetically sorted A-Z)
class_labels = sorted(os.listdir("Data"))  # Modify if needed based on your dataset folder

# Set parameters
img_size = (64, 64)  # Should match the input size used for training

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for convenience (mirror effect)
    frame = cv2.flip(frame, 1)

    # Draw a rectangle for hand region (adjust as needed for better positioning)
    x1, y1, x2, y2 = 200, 100, 400, 300
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Crop the hand region and resize it
    hand_img = frame[y1:y2, x1:x2]
    hand_img = cv2.resize(hand_img, img_size)
    hand_img = img_to_array(hand_img) / 255.0  # Normalize
    hand_img = np.expand_dims(hand_img, axis=0)

    # Get prediction from the model
    prediction = model.predict(hand_img)
    predicted_class = np.argmax(prediction)
    predicted_letter = class_labels[predicted_class]

    # Display prediction on the frame
    cv2.putText(frame, f'Prediction: {predicted_letter}', (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("ASL Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
