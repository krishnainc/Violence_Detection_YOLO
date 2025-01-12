import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('violence_detection_final_model.keras')

# Define the input size for the model
IMG_SIZE = (224, 224)

# Open the webcam (0 for default camera, or replace with the video file path)
cap = cv2.VideoCapture(0)  # Use "0" for the primary webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit.")

# Process the live video feed
while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize and preprocess the frame
    resized_frame = cv2.resize(frame, IMG_SIZE)
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension

    # Make predictions
    prediction = model.predict(input_frame, verbose=0)[0][0]
    label = "Violent" if prediction > 0.5 else "Non-Violent"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Display the prediction on the frame
    color = (0, 0, 255) if label == "Violent" else (0, 255, 0)  # Red for violent, green for non-violent
    cv2.putText(frame, f"{label} ({confidence * 100:.2f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show the video feed
    cv2.imshow('Violence Detection', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
