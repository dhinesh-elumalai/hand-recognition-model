# Hand Sign Recognition

This project aims to recognize hand signs using a Convolutional Neural Network (CNN) trained on images of different hand signs. The project includes scripts for data collection, model training, and real-time recognition.

## Dependencies

Ensure you have the following libraries installed:

- `opencv-python`
- `tensorflow`
- `numpy`
- `json`

You can install these dependencies using the following command:

```sh
pip install -r requirements.txt
```

## Data Collection

The `create-data-set.py` script captures images from the webcam and saves them in directories corresponding to each hand sign. The script also updates a JSON file with the names of the hand signs.

### Usage

```sh
python create-data-set.py
```

### Description

- Prompts the user to input the names of hand signs.
- Captures 100 images for each hand sign from the webcam.
- Saves the images in the `data` directory.
- Updates the `hand_signs.json` file with the names of the hand signs.

## Model Training

The `train-model.py` script trains a CNN on the collected images and saves the trained model to disk.

### Usage

```sh
python train-model.py
```

### Description

- Loads images from the `data` directory.
- Trains a CNN on the images.
- Saves the trained model to `hand_sign_model.h5`.

## Real-Time Recognition

The `recognize-hand.py` script uses the trained model to recognize hand signs in real-time from the webcam.

### Usage

```sh
python recognize-hand.py
```

### Description

- Loads the trained model from `hand_sign_model.h5`.
- Captures video from the webcam.
- Recognizes hand signs in a defined region of interest (ROI) on the screen.
- Displays the recognized hand sign on the screen.

## Notes

- Adjust the region of interest (ROI) coordinates in the scripts based on your specific requirements.
- Ensure the `hand_signs.json` file is present in the same directory as the scripts.
- The rectangle drawn on the screen helps the user position their hand within the ROI for data collection and recognition.
