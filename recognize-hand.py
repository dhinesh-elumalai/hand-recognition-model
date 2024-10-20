import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('./model/hand_sign_model.h5')

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load hand_signs from the file
with open('hand_signs.json', 'r') as f:
    hand_signs = json.load(f)

# Labels for hand signs
labels = {i: sign for i, sign in enumerate(hand_signs)}

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Define the region of interest (ROI) coordinates
x, y, w, h = 600, 200, 700, 700  # Adjust these values as needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw the rectangle on the frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Extract the ROI
    roi = frame[y:y + h, x:x + w]

    # Preprocess the ROI
    img = cv2.resize(roi, (64, 64))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Predict the hand sign
    prediction = model.predict(img)
    sign = labels[np.argmax(prediction)]

    # Display the prediction
    cv2.putText(frame, sign, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    print('Accuracy Level', prediction)
    cv2.imshow('Hand Sign Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()