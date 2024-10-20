import cv2
import os
import json
import time

# Get hand sign names from user
hand_signs = []
num_signs = int(input("Enter the number of hand signs: "))
for i in range(num_signs):
    sign_name = input(f"Enter the name for hand sign {i+1}: ")
    hand_signs.append(sign_name)
    os.makedirs(f'data/{sign_name}', exist_ok=True)

# Load existing hand signs if the file exists
if os.path.exists('hand_signs.json'):
    with open('hand_signs.json', 'r') as f:
        existing_hand_signs = json.load(f)
else:
    existing_hand_signs = []

# Append new hand signs to the existing list
existing_hand_signs.extend(hand_signs)

# Save the updated list to the file
with open('hand_signs.json', 'w') as f:
    json.dump(existing_hand_signs, f)
# Capture images from webcam
cap = cv2.VideoCapture(0)

# Wait for 2 seconds to allow the camera to warm up
time.sleep(2)

# Define the region of interest (ROI) coordinates
x, y, w, h = 600, 200, 700, 700  # Adjust these values as needed

for sign in hand_signs:
    print(f'Collecting images for {sign}')
    for img_num in range(100):  # Collect 100 images per sign
        ret, frame = cap.read()
        if not ret:
            break

        # Draw the rectangle on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the ROI
        roi = frame[y:y + h, x:x + w]

        # Display the frame with the rectangle
        cv2.imshow('frame', frame)

        # Save the ROI
        cv2.imwrite(f'data/{sign}/{img_num}.jpg', roi)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()