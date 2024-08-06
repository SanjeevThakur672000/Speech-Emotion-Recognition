# Speech Emotion Recognition 

This project implements a Speech Emotion Recognition (SER) system using Convolutional Neural Networks (CNN) in Python. The system is designed to classify audio recordings into various emotion categories, leveraging deep learning techniques to achieve accurate predictions. The goal is to enable machines to understand human emotions from speech, which can have applications in customer service, mental health monitoring, and more.

Drive link: https://drive.google.com/drive/folders/125lBC2oJI8uwR-M9jI6ztETjhRhYrcqL?usp=sharing

## Table of Contents

- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Feature Extraction](#feature-extraction)
- [Data Augmentation](#data-augmentation)
- [Model](#model)
- [Detailed Steps](#detailed-steps)

## Prerequisites

Ensure you have Python 3.7 installed. You can download it from [Python's official website](https://www.python.org/downloads/release/python-370/).

## Dataset

This project uses a combination of four well-known datasets for speech emotion recognition:

1. **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**
   - Source: [RAVDESS on Zenodo](https://zenodo.org/record/1188976)
   - Description: This dataset contains 1440 audio files from 24 professional actors (12 female, 12 male) vocalizing two lexically-matched statements in a neutral North American accent.

2. **CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)**
   - Source: [CREMA-D on GitHub](https://github.com/CheyneyComputerScience/CREMA-D)
   - Description: CREMA-D includes facial and vocal emotional expressions in sentences spoken in a range of basic emotions by a diverse group of 91 actors.

3. **TESS (Toronto Emotional Speech Set)**
   - Source: [TESS on TSpace](https://tspace.library.utoronto.ca/handle/1807/24487)
   - Description: A set of 200 target words spoken in the carrier phrase "Say the word _____" by two actresses (aged 26 and 64 years) in seven emotions.

4. **SAVEE (Surrey Audio-Visual Expressed Emotion)**
   - Source: [SAVEE Database](http://kahlan.eps.surrey.ac.uk/savee/)
   - Description: This database was recorded from four male actors in seven different emotions, with 480 British English utterances in total.

These datasets provide a diverse range of emotional speech samples, helping to create a robust and generalizable model for speech emotion recognition.

Note: While we've provided Kaggle paths in our code, we recommend accessing these datasets from their original sources for the most up-to-date versions and to ensure compliance with their respective usage terms.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/speech-emotion-recognition-cnn.git
    cd speech-emotion-recognition-cnn
    ```

2. **Install the required packages using pip for flask web application:**

    ```bash
    pip install Flask==2.2.5
    pip install joblib==1.2.0
    pip install Keras==2.4.3
    pip install librosa==0.8.1
    pip install matplotlib==3.5.0
    pip install numpy==1.19.5
    pip install pandas==1.3.5
    pip install Pillow==9.4.0
    pip install requests==2.31.0
    pip install scikit-learn==0.24.2
    pip install scipy==1.7.3
    pip install seaborn==0.11.2
    pip install soundfile==0.12.1
    pip install tensorflow==2.5.0
    pip install urllib3==2.0.7
    pip install Werkzeug==2.2.3
    ```

3. **Alternatively, install all dependencies using the `requirements.txt` file:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Open and run the Jupyter notebook:**

    This will start the Jupyter server and open a web browser. In the browser:

    - Navigate to and click on speech_emotion_recognition_cnn.ipynb
    - Run each cell in the notebook sequentially

    The notebook will:

    - Preprocess the audio data
    - Train the CNN model
    - Evaluate its performance

2. **Run the Flask web application:**

    ```bash
    python app.py
    ```

    The web application will allow users to upload audio files and get emotion predictions.

## Project Structure

- `app.py`: The Flask web application script.
- `requirements.txt`: A file listing all the required dependencies.
- `static/styles.css`: CSS file for styling the web application.
- `templates/index.html`: HTML file for the web application's front-end.
- `best_model1_weights.h5`: Pre-trained model weights.
- `CNN_model.json`: Model architecture in JSON format.
- `encoder2.pickle`: Label encoder for the emotion classes.
- `scaler2.pickle`: Scaler for feature normalization.
- `notebook/speech-emotion-recognition-CNN.ipynb`: Jupyter Notebook containing exploratory data analysis and model training steps.
 https://drive.google.com/drive/folders/125lBC2oJI8uwR-M9jI6ztETjhRhYrcqL?usp=sharing

## Feature Extraction

Feature extraction is a crucial step in the Speech Emotion Recognition process. In this project, we use the following techniques:

- **Mel-frequency cepstral coefficients (MFCCs):** MFCCs are widely used in speech and audio processing. They provide a compact representation of the power spectrum of an audio signal, capturing the important characteristics of the sound.
- **Chroma Features:** These capture the energy distribution across different pitch classes, highlighting the harmonic and tonal content of the audio.
- **Mel Spectrogram:** This provides a time-frequency representation of the audio signal, offering a detailed view of how energy is distributed across different frequencies over time.

These features are extracted using the `librosa` library.

## Data Augmentation

Data augmentation helps in improving the model's robustness and generalization by artificially increasing the size of the dataset. In this project, we use the following data augmentation techniques:

- **Noise Injection:** Adding random noise to the audio signals.
- **Pitch Shifting:** Shifting the pitch of the audio signals up or down.
- **Time Stretching:** Speeding up or slowing down the audio signals.

These techniques are implemented using `librosa`.

## Model

The model architecture is a Convolutional Neural Network (CNN) designed to capture spatial patterns in the audio features. The architecture includes:

- **Convolutional Layers:** These layers apply convolution operations to extract local patterns.
- **Pooling Layers:** These layers reduce the dimensionality of the feature maps.
- **Fully Connected Layers:** These layers perform the final classification based on the features extracted by the convolutional layers.

The model is implemented using `Keras` and trained with a categorical cross-entropy loss function and the Adam optimizer. The model architecture is saved in `CNN_model.json`, and the best weights are saved in `best_model1_weights.h5`.

## Detailed Steps

1. **Data Preprocessing:**
   - Load audio files using `librosa`
   - Extract features such as MFCCs, Chroma features, and Mel spectrogram
   - Normalize the features using the scaler saved in `scaler2.pickle`

2. **Model Architecture:**
   - Define the CNN architecture using `Keras`
   - Save the model architecture in `CNN_model.json`
   - Compile the model with appropriate loss function and optimizer

3. **Training the Model:**
   - Split the data into training and validation sets
   - Train the model on the training set
   - Save the best model weights in `best_model1_weights.h5`
   - Monitor performance on the validation set

4. **Evaluation:**
   - Evaluate the model's performance on a test set
   - Generate accuracy and loss metrics
   - Visualize the results using `matplotlib` and `seaborn`

5. **Web Application:**
   - Use Flask to create a web application for uploading audio files
   - Load the pre-trained model and scaler
   - Predict emotions from uploaded audio files
   - Display the predicted emotion on the web interface styled with `styles.css`
